#!/usr/bin/env python3
"""Per-seed view of the deployment results: ASR (or CTA) vs budget with one line per seed, to
see where the across-seed variance -- including the nonzero ASR at budget 0 -- comes from.

For each (model, dataset, tag), reading the same caccs.npy/paccs.npy final values as
plot_paper_main_campaign_user.py:
  <out>/<campaign>/<model>_<dataset>_<tag>/
    <metric>_vs_budget_per_seed.{png,pdf}            all aggregators, one panel each
    <metric>_vs_budget_per_seed_<branch>.{png,pdf}   one figure per aggregator
    <metric>_seed_heatmap.{png,pdf}                  seed x budget values, one panel per aggregator
  <out>/per_seed_results.csv                         long format, every cell read

Usage:
  python scripts/show_results/plot_asr_per_seed.py --tags baseline eps_16_255
  python scripts/show_results/plot_asr_per_seed.py --campaign flip_frozen --metric cta
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
from plot_paper_main_campaign_user import (  # noqa: E402
    EXP_BASE,
    FLIP_FROZEN_EXP_BASE,
    DATASETS,
    SEEDS,
    DEPLOY_BUDGETS,
    DATASET_DISPLAY,
    MODEL_DISPLAY,
    TABLE_AGG_DISPLAY,
    _branch_run_dir,
    get_final_value,
)

CAMPAIGNS = {"jolt": EXP_BASE, "flip_frozen": FLIP_FROZEN_EXP_BASE}
CAMPAIGN_DISPLAY = {"jolt": "JOLT", "flip_frozen": "FLIP, frozen JOLT trigger"}
# (branch key, branch, agg) -- "centralized" is the single_user deployment, as in the tables.
BRANCHES = [("centralized", "single_user", None)] + [
    (agg, "federated", agg) for agg in ("mean", "median", "krum", "trmean", "multikrum")
]
METRIC_FILE = {"asr": "paccs.npy", "cta": "caccs.npy"}


def collect(exp_base, campaign, model, dataset, tag, seeds, budgets):
    rows = []
    for key, branch, agg in BRANCHES:
        for seed in seeds:
            for budget in budgets:
                run = _branch_run_dir(model, dataset, tag, seed, branch, agg, exp_base=exp_base) / f"train_user_{budget}"
                cta, asr = (get_final_value(run / METRIC_FILE[m]) for m in ("cta", "asr"))
                if np.isnan(cta) and np.isnan(asr):
                    continue
                rows.append({"campaign": campaign, "model": model, "dataset": dataset, "tag": tag, "branch": key,
                             "seed": seed, "budget": budget, "cta": cta * 100, "asr": asr * 100})
    return pd.DataFrame(rows)


def seed_colors(seeds):
    cmap = plt.get_cmap("tab10" if len(seeds) <= 10 else "tab20")
    return {s: cmap(i % cmap.N) for i, s in enumerate(seeds)}


def draw_per_seed(ax, df, metric, budgets, colors):
    """Budgets on evenly spaced positions (0 ... 5000 would crush the small budgets on a linear axis)."""
    x = {b: i for i, b in enumerate(budgets)}
    for seed, g in df.groupby("seed"):
        g = g.sort_values("budget")
        ax.plot([x[b] for b in g["budget"]], g[metric], color=colors[seed], linewidth=1.2,
                marker="o", markersize=4, alpha=0.85, label=f"seed {seed}")
    mean = df.groupby("budget")[metric].mean().reindex(budgets)
    ax.plot(range(len(budgets)), mean.values, color="black", linewidth=3, marker="s", markersize=6, label="mean")
    ax.set_xticks(range(len(budgets)))
    ax.set_xticklabels([str(b) for b in budgets], rotation=45)


def plot_cell(df, out_dir, title, metric, budgets, seeds):
    colors = seed_colors(seeds)
    ylabel = plot_style.label(metric)
    present = [(key, TABLE_AGG_DISPLAY.get(key, key.upper())) for key, _, _ in BRANCHES if key in set(df["branch"])]

    # one panel per aggregator
    ncols = 3
    nrows = (len(present) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.6 * nrows), squeeze=False, sharey=True)
    for i, (key, name) in enumerate(present):
        ax = axes[i // ncols][i % ncols]
        draw_per_seed(ax, df[df["branch"] == key], metric, budgets, colors)
        plot_style.style_axes(ax, "Deployment budget (poisoned samples)", ylabel if i % ncols == 0 else "", name)
    for i in range(len(present), nrows * ncols):
        axes[i // ncols][i % ncols].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=True, fontsize=10)
    fig.suptitle(title)
    plot_style.save(fig, out_dir / f"{metric}_vs_budget_per_seed")

    # one figure per aggregator (paper subfigures)
    for key, name in present:
        fig, ax = plt.subplots(figsize=plot_style.FIGSIZE)
        draw_per_seed(ax, df[df["branch"] == key], metric, budgets, colors)
        plot_style.style_axes(ax, "Deployment budget (poisoned samples)", ylabel, f"{title} -- {name}")
        ax.legend(frameon=True, fontsize=9, loc="best", ncol=2)
        plot_style.save(fig, out_dir / f"{metric}_vs_budget_per_seed_{key}")

    # seed x budget heatmap: is it always the same seeds that score high?
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 0.42 * len(seeds) * nrows + 1.8 * nrows),
                             squeeze=False, layout="constrained")
    vmax = 100
    for i, (key, name) in enumerate(present):
        ax = axes[i // ncols][i % ncols]
        grid = df[df["branch"] == key].pivot_table(index="seed", columns="budget", values=metric).reindex(
            index=seeds, columns=budgets)
        im = ax.imshow(grid.values, cmap="Reds", vmin=0, vmax=vmax, aspect="auto")
        for (r, c), v in np.ndenumerate(grid.values):
            if not np.isnan(v):
                ax.text(c, r, f"{v:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if v > 0.6 * vmax else "black")
        ax.set_xticks(range(len(budgets)))
        ax.set_xticklabels([str(b) for b in budgets], rotation=45)
        ax.set_yticks(range(len(seeds)))
        ax.set_yticklabels([f"seed {s}" for s in seeds] if i % ncols == 0 else [])
        if i // ncols == nrows - 1 or i + ncols >= len(present):
            ax.set_xlabel("Deployment budget (poisoned samples)")
        ax.set_title(name)
    for i in range(len(present), nrows * ncols):
        axes[i // ncols][i % ncols].axis("off")
    fig.colorbar(im, ax=axes, label=ylabel, shrink=0.8)
    fig.suptitle(title)
    # saved directly (not plot_style.save): tight_layout would fight the constrained layout
    for ext in plot_style.FORMATS:
        path = out_dir / f"{metric}_seed_heatmap.{ext}"
        fig.savefig(path, dpi=plot_style.DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", choices=list(CAMPAIGNS), default="jolt")
    parser.add_argument("--root", default=None, help="override the campaign's results root (like --jolt-root)")
    parser.add_argument("--model", default="r32p")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--tags", nargs="+", default=["baseline", "eps_16_255"])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--budgets", nargs="+", type=int, default=DEPLOY_BUDGETS)
    parser.add_argument("--metric", choices=list(METRIC_FILE), default="asr")
    parser.add_argument("--out-dir", default="./plots_per_seed")
    args = parser.parse_args()

    plot_style.apply()
    exp_base = Path(args.root) if args.root else CAMPAIGNS[args.campaign]
    out_root = Path(args.out_dir)
    frames = []
    for dataset in args.datasets:
        for tag in args.tags:
            df = collect(exp_base, args.campaign, args.model, dataset, tag, args.seeds, args.budgets)
            if df.empty:
                print(f"[INFO] no results for {args.campaign}/{args.model}/{dataset}/{tag} under {exp_base}")
                continue
            frames.append(df)
            out_dir = out_root / args.campaign / f"{args.model}_{dataset}_{tag}"
            out_dir.mkdir(parents=True, exist_ok=True)
            title = (f"{CAMPAIGN_DISPLAY[args.campaign]} ({tag}) -- {MODEL_DISPLAY.get(args.model, args.model)} / "
                     f"{DATASET_DISPLAY.get(dataset, dataset)}")
            plot_cell(df, out_dir, title, args.metric, args.budgets, args.seeds)
            zero = df[df["budget"] == 0].groupby("branch")[args.metric]
            print(f"[{dataset}/{tag}] {args.metric} at budget 0 per aggregator (mean / min / max over seeds):")
            for key, s in zero:
                print(f"    {key:12s} {s.mean():6.1f} / {s.min():6.1f} / {s.max():6.1f}   n={len(s)}")
            print(f"[INFO] Saved {out_dir}/")
    if frames:
        out_root.mkdir(parents=True, exist_ok=True)
        csv = out_root / "per_seed_results.csv"
        pd.concat(frames).to_csv(csv, index=False)
        print(f"[INFO] Saved {csv}")


if __name__ == "__main__":
    main()
