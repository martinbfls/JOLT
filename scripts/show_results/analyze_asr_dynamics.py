#!/usr/bin/env python3
"""Per-epoch ASR/CTA dynamics for a trigger campaign -- the view that tells a backdoor which
SURVIVES the victim's training from one that merely peaks mid-run and is then unlearned.

plot_warp_multibudget.py answers "what is the final ASR per budget". This one answers "did the
ASR get there and stay, or did it spike and collapse", and gives the two series that discriminate
the usual explanations for a collapse:

  - the victim trains on CLEAN images with FLIPPED labels (the trigger is only ever applied to
    the ASR test set), so the flipped examples are label noise. A network explains label noise
    with a shared feature while the learning rate is high -- which is what the trigger rides --
    and memorizes it example by example once the LR drops. Symptom: ASR collapses AT an LR
    milestone while the TRAIN accuracy keeps climbing.
  - under a robust aggregator, the poisoned workers can instead be progressively filtered out.
    Symptom: cos(poisoned gradient, honest gradient) decays toward 0 as the ASR falls, and the
    centralized branch (a no-op `mean` aggregator) does NOT show the same collapse.

Reads, per (tag, seed, budget) and per branch:
  <exp-root>/<exp-base-prefix>_<tag>/<model>/<dataset>/1vs0/<gen-agg>/seed<seed>/
      train_user_<budget>/                               centralized branch (1vs0, mean)
      <deploy-agg>_<P>vs<H>/train_user_<budget>/         federated branch
          logs/federated_train_user_metrics.json         asr, cta, train acc/loss, lr, grad cosine
          paccs.npy / caccs.npy                          fallback: per-epoch (acc, loss), asr/cta only

Outputs under --out-dir:
  asr_vs_epoch_<tag>_<branch>.{png,pdf}   one panel per budget; ASR solid, CTA dashed, mean over
                                          seeds thick, each seed thin, LR drops as vertical lines
  transience.{png,pdf}                    peak ASR vs final ASR per (tag, budget) -- the headline
  diagnosis_<tag>_<branch>.{png,pdf}      ASR, train accuracy and cos(poisoned, honest) on one
                                          epoch axis, for the budget with the largest collapse
  dynamics.csv                            long format, every epoch of every cell read
  transience.{csv,md}                     per (tag, branch, budget): final/peak ASR, peak epoch,
                                          collapse, mean +/- std over seeds
and prints the transience table, flagging the cells whose ASR drops by more than
--transient-threshold points between its peak and the last epoch.

Usage (from repo root; needs no Slurm and no GPU):
  python scripts/show_results/analyze_asr_dynamics.py \\
      --variant-tags hybrid_eps0p25_s8p0 hybrid_eps0p0627_s8p0 --seeds 0 1 2
  python scripts/show_results/analyze_asr_dynamics.py --branch both   # federated AND centralized
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modules.base_utils import plot_style  # noqa: E402
from modules.base_utils.plot_style import plt  # noqa: E402

# Series pulled out of the tracker JSON. ExperimentTracker stores raw values (plot_style.scale is
# applied at plot time only), so the [0, 1] accuracies need the x100 here.
JSON_SERIES = {
    "asr": ("asr", 100.0),
    "cta": ("cta", 100.0),
    "train_acc": ("acc", 100.0),
    "train_loss": ("loss", 1.0),
    "lr": ("lr", 1.0),
    "grad_cos": ("grad_cosine_sim_poisoned_honest", 1.0),
}


def cell_dir(args, tag, seed):
    return (
        Path(args.exp_root)
        / f"{args.exp_base_prefix}_{tag}"
        / args.model_flag
        / args.dataset
        / "1vs0"
        / args.gen_agg_method
        / f"seed{seed}"
    )


def run_dir(args, tag, seed, budget, branch):
    base = cell_dir(args, tag, seed)
    if branch == "centralized":
        return base / f"train_user_{budget}"
    deploy = f"{args.deploy_agg_method}_{args.deploy_num_poisoned}vs{args.deploy_num_honests}"
    return base / deploy / f"train_user_{budget}"


def read_series(path):
    """Per-epoch series for one run, as {name: (epochs, values)}; empty dict when unreadable."""
    metrics_path = path / "logs" / "federated_train_user_metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            data = {}
        out = {}
        for name, (key, scale) in JSON_SERIES.items():
            points = data.get(key) or []
            if points:
                out[name] = (
                    np.array([p["step"] for p in points], dtype=int),
                    np.array([p["value"] for p in points], dtype=float) * scale,
                )
        if "asr" in out:
            return out

    # Fallback: the raw per-epoch (acc, loss) arrays train_user saves. No lr/train acc/grad cosine.
    out = {}
    for name, fname in (("asr", "paccs.npy"), ("cta", "caccs.npy")):
        npy = path / fname
        if not npy.exists() or npy.stat().st_size == 0:
            continue
        try:
            arr = np.load(npy, allow_pickle=True)
        except Exception:  # noqa: BLE001 -- a truncated file from a killed job
            continue
        if arr.size == 0:
            continue
        values = arr[:, 0] if arr.ndim > 1 else arr
        out[name] = (np.arange(1, len(values) + 1), np.asarray(values, dtype=float) * 100.0)
    return out


def collect(args, branches):
    rows, missing = [], []
    for tag in args.variant_tags:
        for branch in branches:
            for seed in args.seeds:
                for budget in args.budgets:
                    path = run_dir(args, tag, seed, budget, branch)
                    series = read_series(path)
                    if "asr" not in series:
                        missing.append((tag, branch, seed, budget, path))
                        continue
                    epochs = series["asr"][0]
                    record = {"tag": tag, "branch": branch, "seed": seed, "budget": budget,
                              "epoch": epochs}
                    for name, (steps, values) in series.items():
                        # every series is logged once per epoch by the same callback, but a run
                        # killed mid-write could still truncate one of them
                        record[name] = values if len(values) == len(epochs) else np.nan
                    rows.append(pd.DataFrame(record))
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return df, missing


def lr_drop_epochs(df_cell):
    """Epochs where the learning rate changed, read from the run itself rather than assumed."""
    if "lr" not in df_cell or df_cell["lr"].isna().all():
        return []
    g = df_cell.sort_values("epoch")
    lr = g["lr"].to_numpy(float)
    epochs = g["epoch"].to_numpy()
    return [int(e) for e, prev, cur in zip(epochs[1:], lr[:-1], lr[1:]) if cur < prev * 0.99]


def summarize(df, args):
    """Per (tag, branch, seed, budget): final ASR, peak ASR and when it peaked."""
    records = []
    for (tag, branch, seed, budget), g in df.groupby(["tag", "branch", "seed", "budget"]):
        g = g.sort_values("epoch")
        asr = g["asr"].to_numpy(float)
        peak_i = int(np.nanargmax(asr))
        rec = {
            "tag": tag, "branch": branch, "seed": seed, "budget": budget,
            "n_epochs": len(asr),
            "asr_final": asr[-1],
            "asr_peak": asr[peak_i],
            "epoch_peak": int(g["epoch"].to_numpy()[peak_i]),
            "asr_collapse": asr[peak_i] - asr[-1],
            "cta_final": g["cta"].to_numpy(float)[-1] if "cta" in g else np.nan,
        }
        for name in ("train_acc", "grad_cos"):
            rec[f"{name}_final"] = g[name].to_numpy(float)[-1] if name in g and not g[name].isna().all() else np.nan
        rec["lr_drops"] = lr_drop_epochs(g)
        records.append(rec)
    return pd.DataFrame(records)


def aggregate(per_seed, args):
    """mean/std across seeds, reindexed so every requested (tag, branch, budget) shows up."""
    cols = ["asr_final", "asr_peak", "epoch_peak", "asr_collapse", "cta_final",
            "train_acc_final", "grad_cos_final"]
    agg = per_seed.groupby(["tag", "branch", "budget"])[cols].agg(["mean", "std", "count"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    return agg.reset_index()


def fmt(row, col, n_seeds):
    mean, std, count = row[f"{col}_mean"], row[f"{col}_std"], row[f"{col}_count"]
    if pd.isna(mean):
        return "--"
    std = 0.0 if pd.isna(std) else std
    flag = "" if count == n_seeds else f" (n={int(count)})"
    return f"{mean:.1f} +/- {std:.1f}{flag}"


def write_table(agg, args, out_dir):
    n_seeds = len(args.seeds)
    header = ["tag", "branch", "budget", "ASR final (%)", "ASR peak (%)", "peak epoch",
              "collapse (pts)", "CTA final (%)", "transient?"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for _, row in agg.iterrows():
        collapse = row["asr_collapse_mean"]
        flag = "--" if pd.isna(collapse) else ("YES" if collapse > args.transient_threshold else "no")
        lines.append("| " + " | ".join([
            str(row["tag"]), str(row["branch"]), str(row["budget"]),
            fmt(row, "asr_final", n_seeds), fmt(row, "asr_peak", n_seeds),
            fmt(row, "epoch_peak", n_seeds), fmt(row, "asr_collapse", n_seeds),
            fmt(row, "cta_final", n_seeds), flag,
        ]) + " |")
    table = "\n".join(lines)
    (out_dir / "transience.md").write_text(table + "\n")
    agg.to_csv(out_dir / "transience.csv", index=False)
    return table


def draw_curves(df, args, tag, branch, out_dir, colors):
    cell = df[(df["tag"] == tag) & (df["branch"] == branch)]
    if cell.empty:
        return []
    budgets = [b for b in args.budgets if b in set(cell["budget"])]
    ncols = min(4, len(budgets))
    nrows = int(np.ceil(len(budgets) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), squeeze=False)

    for ax, budget in zip(axes.flat, budgets):
        sub = cell[cell["budget"] == budget]
        for seed, g in sub.groupby("seed"):
            g = g.sort_values("epoch")
            ax.plot(g["epoch"], g["asr"], color=colors["asr"], linewidth=0.8, alpha=0.3)
            ax.plot(g["epoch"], g["cta"], color=colors["cta"], linewidth=0.8, alpha=0.3)
        mean = sub.groupby("epoch")[["asr", "cta"]].mean().sort_index()
        ax.plot(mean.index, mean["asr"], color=colors["asr"], linewidth=2.2, label="ASR")
        ax.plot(mean.index, mean["cta"], color=colors["cta"], linewidth=2.2, linestyle="--", label="CTA")
        for drop in lr_drop_epochs(sub[sub["seed"] == sub["seed"].iloc[0]]):
            ax.axvline(drop, color="gray", linestyle=":", linewidth=1.2)
        ax.set_ylim(0, 100)
        plot_style.style_axes(ax, "Epoch", "Accuracy (%)", title=f"budget {budget}")

    for ax in axes.flat[len(budgets):]:
        ax.axis("off")
    axes.flat[0].legend(loc="best")
    fig.suptitle(f"{tag} -- {branch} ({args.model_flag}/{args.dataset}, {len(args.seeds)} seeds; "
                 "dotted = LR drop)")
    return plot_style.save(fig, out_dir / f"asr_vs_epoch_{tag}_{branch}")


def draw_transience(agg, args, out_dir):
    """Peak vs final ASR per (tag, branch, budget) -- the gap IS the problem, so show it directly."""
    rows = agg.dropna(subset=["asr_final_mean"])
    if rows.empty:
        return []
    labels = [f"{r['tag']}\n{r['branch']}\nb={r['budget']}" for _, r in rows.iterrows()]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(max(8.0, 0.85 * len(rows)), 5.5))
    ax.bar(x - 0.2, rows["asr_peak_mean"], width=0.4, label="peak ASR",
           color="tab:orange", yerr=rows["asr_peak_std"].fillna(0), capsize=3)
    ax.bar(x + 0.2, rows["asr_final_mean"], width=0.4, label="final ASR",
           color="tab:red", yerr=rows["asr_final_std"].fillna(0), capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.set_ylim(0, 100)
    ax.legend()
    plot_style.style_axes(ax, "", "ASR (%)", title="Peak vs final ASR (a tall gap = transient backdoor)")
    return plot_style.save(fig, out_dir / "transience")


def draw_diagnosis(df, per_seed, args, tag, branch, out_dir):
    """For the worst-collapsing budget: ASR, train accuracy and cos(poisoned, honest) together."""
    cells = per_seed[(per_seed["tag"] == tag) & (per_seed["branch"] == branch)]
    if cells.empty or cells["asr_collapse"].isna().all():
        return []
    budget = int(cells.groupby("budget")["asr_collapse"].mean().idxmax())
    sub = df[(df["tag"] == tag) & (df["branch"] == branch) & (df["budget"] == budget)]
    if sub.empty:
        return []
    mean = sub.groupby("epoch").mean(numeric_only=True).sort_index()

    fig, ax = plt.subplots(figsize=plot_style.FIGSIZE)
    ax.plot(mean.index, mean["asr"], color="tab:red", label="ASR")
    if "train_acc" in mean and not mean["train_acc"].isna().all():
        ax.plot(mean.index, mean["train_acc"], color="tab:green", label="train accuracy")
    for drop in lr_drop_epochs(sub[sub["seed"] == sub["seed"].iloc[0]]):
        ax.axvline(drop, color="gray", linestyle=":", linewidth=1.2)
    ax.set_ylim(0, 100)
    plot_style.style_axes(ax, "Epoch", "Accuracy (%)")

    handles, labels = ax.get_legend_handles_labels()
    if "grad_cos" in mean and not mean["grad_cos"].isna().all():
        ax2 = ax.twinx()
        ax2.plot(mean.index, mean["grad_cos"], color="tab:blue", linestyle="-.",
                 label="cos(poisoned, honest)")
        ax2.set_ylabel("cos(poisoned, honest)")
        ax2.set_ylim(-1, 1)
        h2, l2 = ax2.get_legend_handles_labels()
        handles, labels = handles + h2, labels + l2
    ax.legend(handles, labels, loc="best")
    ax.set_title(f"{tag} -- {branch}, budget {budget} (worst collapse)")
    return plot_style.save(fig, out_dir / f"diagnosis_{tag}_{branch}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp-root", default="experiments/federated_experiments")
    parser.add_argument("--exp-base-prefix", default="threat_model_direct_trigger_joint")
    parser.add_argument("--variant-tags", nargs="+",
                        default=["hybrid_eps0p25_s8p0", "hybrid_eps0p0627_s8p0"])
    parser.add_argument("--model-flag", default="r32p")
    parser.add_argument("--dataset", default="cifar")
    parser.add_argument("--gen-agg-method", default="mean")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--budgets", nargs="+", type=int,
                        default=[0, 150, 300, 500, 1000, 2000, 2500, 5000])
    parser.add_argument("--branch", choices=["federated", "centralized", "both"], default="both",
                        help="which deployment to read; 'both' is what tells an aggregator "
                             "effect from a plain optimization one")
    parser.add_argument("--deploy-agg-method", default="multikrum")
    parser.add_argument("--deploy-num-poisoned", type=int, default=3)
    parser.add_argument("--deploy-num-honests", type=int, default=7)
    parser.add_argument("--transient-threshold", type=float, default=20.0,
                        help="ASR points between peak and final above which a cell is flagged")
    parser.add_argument("--out-dir", default="experiments/federated_experiments/asr_dynamics")
    args = parser.parse_args()

    branches = ["federated", "centralized"] if args.branch == "both" else [args.branch]
    df, missing = collect(args, branches)
    if df.empty:
        print("[ABORT] no readable run found. Looked for e.g.:")
        for _, _, _, _, path in missing[:3]:
            print(f"  {path}")
        raise SystemExit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "dynamics.csv", index=False)

    per_seed = summarize(df, args)
    agg = aggregate(per_seed, args)
    table = write_table(agg, args, out_dir)

    plot_style.apply()
    colors = {"asr": "tab:red", "cta": "tab:blue"}
    written = list(draw_transience(agg, args, out_dir))
    for tag in args.variant_tags:
        for branch in branches:
            written += draw_curves(df, args, tag, branch, out_dir, colors)
            written += draw_diagnosis(df, per_seed, args, tag, branch, out_dir)

    n_cells = len(per_seed)
    print(f"[SUMMARY] {n_cells} finished cell(s) over {len(args.variant_tags)} tag(s) x "
          f"{len(branches)} branch(es) x {len(args.seeds)} seed(s) x {len(args.budgets)} budget(s)")
    has_lr = per_seed["lr_drops"].map(bool).any()
    if not has_lr:
        print("[NOTE] no learning-rate series found (cells read from the .npy fallback): the LR "
              "drops cannot be marked, and train accuracy / gradient cosine are unavailable.")
    print(table)
    print(f"\n[FLAG] 'transient' = peak ASR exceeds final ASR by more than "
          f"{args.transient_threshold:.0f} points, averaged over seeds.")

    if missing:
        print(f"\n[MISSING] {len(missing)} cell(s) with no results yet:")
        for tag, branch, seed, budget, path in missing[:15]:
            print(f"  {tag} {branch} seed{seed} budget{budget}: {path}")
        if len(missing) > 15:
            print(f"  ... and {len(missing) - 15} more")

    print("\n[OK] written:")
    for path in [out_dir / "dynamics.csv", out_dir / "transience.md",
                 out_dir / "transience.csv", *written]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
