#!/usr/bin/env python3
"""Which classes get flipped to which? Confusion-style matrices of the selected label flips.

For a (model, dataset, tag, seed) cell and a deployment budget n, the flipped samples are those whose
selected label (argmax of <cell>/select_flips/<n>.npy) differs from the true one (select_flips/true.npy).
Each matrix is row-normalized: entry (i, j) = share of the true-class-i samples (of the whole
distillation set) that were flipped to class j. The diagonal is the share left unflipped, so it is
masked by default to keep the colour scale readable on the off-diagonal. Matrices are averaged over seeds.

The selection is shared by the single_user and federated deployments (the federated one only shards it
across workers), so the single_user select_flips is read for both.

  <out>/<campaign>/<model>_<dataset>_<tag>/
    flip_matrix_<budget>.{png,pdf}     one matrix per budget
    flip_matrix_all_budgets.{png,pdf}  every budget side by side
  <out>/flip_matrices.csv              long format: budget, seed, true class, new class, count, share

Usage:
  python scripts/show_results/plot_flip_matrix.py --tags baseline eps_16_255
  python scripts/show_results/plot_flip_matrix.py --campaign flip_frozen --budgets 500 2500 5000
  python scripts/show_results/plot_flip_matrix.py --counts      # annotate with sample counts instead of %
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from modules.base_utils import plot_style  # noqa: E402
from modules.base_utils.plot_style import plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from modules.generate_labels_trigger_joint.gen_configs import SOURCE_LABEL, TARGET_LABEL  # noqa: E402
from plot_paper_main_campaign_user import (  # noqa: E402
    EXP_BASE,
    FLIP_FROZEN_EXP_BASE,
    DATASETS,
    SEEDS,
    DEPLOY_BUDGETS,
    DATASET_DISPLAY,
    MODEL_DISPLAY,
    _branch_run_dir,
)

CAMPAIGNS = {"jolt": EXP_BASE, "flip_frozen": FLIP_FROZEN_EXP_BASE}
CAMPAIGN_DISPLAY = {"jolt": "JOLT", "flip_frozen": "FLIP, frozen JOLT trigger"}
SOURCE_COLOR, TARGET_COLOR = "tab:blue", "tab:green"  # tab: palette; not red, which the heatmap already uses
CLASS_NAMES = {
    "cifar": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
    "svhn": [str(d) for d in range(10)],
}


def flip_counts(select_flips_dir, budget):
    """(K, K) counts[true class, selected class] over the distillation set, or None if missing."""
    sel_path, true_path = Path(select_flips_dir) / f"{budget}.npy", Path(select_flips_dir) / "true.npy"
    if not (sel_path.exists() and true_path.exists()):
        return None
    sel, true = np.load(sel_path), np.load(true_path)
    k = true.shape[1]
    counts = np.zeros((k, k), dtype=np.int64)
    np.add.at(counts, (true.argmax(axis=1), sel.argmax(axis=1)), 1)
    return counts


def collect(exp_base, model, dataset, tag, seeds, budgets):
    """{budget: {seed: counts}}"""
    out = {}
    for budget in budgets:
        for seed in seeds:
            cell_dir = _branch_run_dir(model, dataset, tag, seed, "single_user", exp_base=exp_base).parent
            c = flip_counts(cell_dir / "select_flips", budget)
            if c is not None:
                out.setdefault(budget, {})[seed] = c
    return out


def mean_share(per_seed):
    """Row-normalized matrix averaged over seeds, plus the mean counts."""
    counts = np.mean([c for c in per_seed.values()], axis=0)
    shares = np.mean([c / c.sum(axis=1, keepdims=True) for c in per_seed.values()], axis=0)
    return shares * 100, counts


def display_vmax(share, mask_diag):
    shown = np.array(share, dtype=float)
    if mask_diag:
        np.fill_diagonal(shown, np.nan)
    return float(np.nanmax(shown)) if np.isfinite(shown).any() and np.nanmax(shown) > 0 else 1.0


def draw(ax, share, counts, names, annotate_counts, mask_diag, title, vmax, fontsize=9):
    """Matrix with the attack's source row (blue) and target column (red) flagged."""
    k = len(names)
    shown = np.array(share, dtype=float)
    if mask_diag:
        np.fill_diagonal(shown, np.nan)
    cmap = plt.get_cmap("Reds").copy()
    cmap.set_bad("#ececec")
    im = ax.imshow(shown, cmap=cmap, vmin=0, vmax=vmax)
    for (i, j), v in np.ndenumerate(shown):
        if np.isnan(v):
            continue
        val = counts[i, j] if annotate_counts else v
        if val == 0:
            continue
        txt = f"{val:.0f}" if annotate_counts else (f"{val:.1f}" if val < 10 else f"{val:.0f}")
        ax.text(j, i, txt, ha="center", va="center", fontsize=fontsize,
                color="white" if v > 0.6 * vmax else "black")
    ax.set_xticks(range(k))
    ax.set_xticklabels(names, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(k))
    ax.set_yticklabels(names)
    for tick in ax.get_yticklabels():
        if tick.get_text() == names[SOURCE_LABEL]:
            tick.set_color(SOURCE_COLOR)
            tick.set_fontweight("bold")
    for tick in ax.get_xticklabels():
        if tick.get_text() == names[TARGET_LABEL]:
            tick.set_color(TARGET_COLOR)
            tick.set_fontweight("bold")
    # outlines: whole source row, whole target column
    ax.add_patch(Rectangle((-0.5, SOURCE_LABEL - 0.5), k, 1, fill=False, edgecolor=SOURCE_COLOR, linewidth=2.2,
                           clip_on=False))
    ax.add_patch(Rectangle((TARGET_LABEL - 0.5, -0.5), 1, k, fill=False, edgecolor=TARGET_COLOR, linewidth=2.2,
                           clip_on=False))
    ax.set_xticks(np.arange(-0.5, k, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, k, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Flipped to")
    ax.set_ylabel("True class")
    if title:
        ax.set_title(title)
    return im


def role_legend(fig, names, **kwargs):
    handles = [
        Line2D([], [], color=SOURCE_COLOR, linewidth=2.5, label=f"Source class ({names[SOURCE_LABEL]})"),
        Line2D([], [], color=TARGET_COLOR, linewidth=2.5, label=f"Target class ({names[TARGET_LABEL]})"),
    ]
    fig.legend(handles=handles, frameon=False, ncol=2, **kwargs)


def save_constrained(fig, path_stem):
    """plot_style.save calls tight_layout, which would fight the constrained layout."""
    for ext in plot_style.FORMATS:
        fig.savefig(Path(path_stem).with_suffix(f".{ext}"), dpi=plot_style.DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", choices=list(CAMPAIGNS), default="jolt")
    parser.add_argument("--root", default=None, help="override the campaign's results root")
    parser.add_argument("--model", default="r32p")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--tags", nargs="+", default=["baseline", "eps_16_255"])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--budgets", nargs="+", type=int, default=[b for b in DEPLOY_BUDGETS if b > 0])
    parser.add_argument("--counts", action="store_true", help="annotate with mean sample counts instead of %%")
    parser.add_argument("--keep-diagonal", action="store_true", help="show the unflipped share on the diagonal")
    parser.add_argument("--no-title", action="store_true", help="omit titles (caption goes in the paper)")
    parser.add_argument("--out-dir", default="./plots_flip_matrix")
    args = parser.parse_args()

    plot_style.apply()
    exp_base = Path(args.root) if args.root else CAMPAIGNS[args.campaign]
    out_root = Path(args.out_dir)
    rows = []
    for dataset in args.datasets:
        names = CLASS_NAMES.get(dataset)
        for tag in args.tags:
            data = collect(exp_base, args.model, dataset, tag, args.seeds, args.budgets)
            if not data:
                print(f"[INFO] no select_flips for {args.campaign}/{args.model}/{dataset}/{tag} under {exp_base}")
                continue
            out_dir = out_root / args.campaign / f"{args.model}_{dataset}_{tag}"
            out_dir.mkdir(parents=True, exist_ok=True)
            title = (f"{CAMPAIGN_DISPLAY[args.campaign]} ({tag}) -- {MODEL_DISPLAY.get(args.model, args.model)} / "
                     f"{DATASET_DISPLAY.get(dataset, dataset)}")
            budgets = [b for b in args.budgets if b in data]
            stats = {b: mean_share(data[b]) for b in budgets}
            k = next(iter(stats.values()))[0].shape[0]
            labels = names if names and len(names) == k else [str(i) for i in range(k)]

            colorbar_label = "% of the true class" if args.keep_diagonal else "Flipped (% of the true class)"
            if args.counts:
                colorbar_label = "Samples (mean over seeds)"
            mask_diag = not args.keep_diagonal
            suptitle = None if args.no_title else title

            for b in budgets:
                fig, ax = plt.subplots(figsize=(7.5, 7.2), layout="constrained")
                im = draw(ax, *stats[b], labels, args.counts, mask_diag,
                          None if args.no_title else f"{title}\nbudget {b}", display_vmax(stats[b][0], mask_diag))
                fig.colorbar(im, ax=ax, label=colorbar_label, shrink=0.85)
                role_legend(fig, labels, loc="outside lower center")
                save_constrained(fig, out_dir / f"flip_matrix_{b}")

            # one colour scale across budgets so the panels are comparable
            vmax = max(display_vmax(stats[b][0], mask_diag) for b in budgets)
            ncols = min(4, len(budgets))
            nrows = (len(budgets) + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols + 1.2, 4.9 * nrows + 0.8), squeeze=False,
                                     layout="constrained")
            for i, b in enumerate(budgets):
                im = draw(axes[i // ncols][i % ncols], *stats[b], labels, args.counts, mask_diag, f"Budget {b}",
                          vmax, fontsize=7)
                if i % ncols:
                    axes[i // ncols][i % ncols].set_ylabel("")
            for i in range(len(budgets), nrows * ncols):
                axes[i // ncols][i % ncols].axis("off")
            fig.colorbar(im, ax=axes, label=colorbar_label, shrink=0.8)
            role_legend(fig, labels, loc="outside lower center")
            if suptitle:
                fig.suptitle(suptitle)
            save_constrained(fig, out_dir / "flip_matrix_all_budgets")

            for b in budgets:
                for seed, c in data[b].items():
                    sh = c / c.sum(axis=1, keepdims=True) * 100
                    for i in range(k):
                        for j in range(k):
                            rows.append({"campaign": args.campaign, "model": args.model, "dataset": dataset,
                                         "tag": tag, "budget": b, "seed": seed, "true_class": labels[i],
                                         "new_class": labels[j], "count": int(c[i, j]), "share_pct": sh[i, j]})
            # one-line text summary: where does each class's flipped mass go (largest budget)
            b = budgets[-1]
            share, _ = stats[b]
            print(f"[{dataset}/{tag}] budget {b} (source={labels[SOURCE_LABEL]}, target={labels[TARGET_LABEL]}), main destination of each class's flips:")
            for i in range(k):
                off = share[i].copy()
                off[i] = 0
                role = " [source]" if i == SOURCE_LABEL else " [target]" if i == TARGET_LABEL else ""
                print(f"    {labels[i]:>10s}{role}: {off.sum():5.1f}% flipped, mostly -> {labels[int(off.argmax())]} "
                      f"({off.max():.1f}%)")
            print(f"[INFO] Saved {out_dir}/")
    if rows:
        out_root.mkdir(parents=True, exist_ok=True)
        csv = out_root / "flip_matrices.csv"
        pd.DataFrame(rows).to_csv(csv, index=False)
        print(f"[INFO] Saved {csv}")


if __name__ == "__main__":
    main()
