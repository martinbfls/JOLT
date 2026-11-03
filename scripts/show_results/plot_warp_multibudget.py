#!/usr/bin/env python3
"""ASR/CTA vs flip budget for the warp multi-budget federated campaign
(orchestrate_slurm/run_warp_multibudget_vs_multikrum_slurm.sh): one line per VARIANT_TAG,
mean +/- std across seeds, on the federated deploy branch only (that script does not re-deploy
the centralized one).

Reads the deploy cells that script writes:
  <exp-root>/<exp-base-prefix>_<tag>/<model>/<dataset>/1vs0/<gen-agg>/seed<seed>/
      <deploy-agg>_<P>vs<H>/train_user_<budget>/
          caccs.npy / paccs.npy                    final value, via plot_paper_main_campaign_user's
                                                   own loader, so the numbers match the paper figures
          logs/federated_train_user_metrics.json    fallback (already in %) when the .npy are missing

Outputs under --out-dir:
  asr_cta_vs_budget.{png,pdf}   both panels side by side -- the figure to look at first
  asr_vs_budget.{png,pdf}, cta_vs_budget.{png,pdf}
  per_seed.csv                  long format, every (tag, seed, budget) cell actually read
  summary.{md,csv}              mean +/- std per (tag, budget)
and prints the summary table plus any missing cells.

Usage (from repo root, with the JOLT conda env active):
  python scripts/show_results/plot_warp_multibudget.py \\
      --variant-tags warp_s8_lpips0p0 warp_s16_lpips0p0 \\
      --seeds 0 1 2 3 4 --budgets 0 150 300 500 1000 2000 2500 5000
  # with the individual seeds drawn as thin lines behind the mean:
  python scripts/show_results/plot_warp_multibudget.py --per-seed
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from modules.base_utils import plot_style  # noqa: E402
from modules.base_utils.plot_style import plt  # noqa: E402
from plot_paper_main_campaign_user import get_final_value  # noqa: E402

METRICS = ("asr", "cta")


def deploy_run_dir(args, tag, seed, budget):
    """The federated train_user directory for one (tag, seed, budget) cell."""
    return (
        Path(args.exp_root)
        / f"{args.exp_base_prefix}_{tag}"
        / args.model_flag
        / args.dataset
        / "1vs0"
        / args.gen_agg_method
        / f"seed{seed}"
        / f"{args.deploy_agg_method}_{args.deploy_num_poisoned}vs{args.deploy_num_honests}"
        / f"train_user_{budget}"
    )


def read_final(run_dir):
    """(cta, asr) in %, plus which file they came from; NaNs when the run has not finished."""
    cta = get_final_value(run_dir / "caccs.npy") * 100
    asr = get_final_value(run_dir / "paccs.npy") * 100
    if not (np.isnan(cta) and np.isnan(asr)):
        return cta, asr, "npy"

    metrics_path = run_dir / "logs" / "federated_train_user_metrics.json"
    if not metrics_path.exists():
        return np.nan, np.nan, None
    try:
        with open(metrics_path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return np.nan, np.nan, None
    # ExperimentTracker stores the RAW value, i.e. the [0, 1] accuracy clf_eval returns
    # (plot_style.scale is applied at plot time only) -- so this needs the same x100 as the .npy
    cta_log, asr_log = data.get("cta", []), data.get("asr", [])
    return (
        cta_log[-1]["value"] * 100 if cta_log else np.nan,
        asr_log[-1]["value"] * 100 if asr_log else np.nan,
        "json",
    )


def collect(args):
    """Long-format rows for every cell that exists, plus the list of the ones that do not."""
    rows, missing = [], []
    for tag in args.variant_tags:
        for seed in args.seeds:
            for budget in args.budgets:
                run_dir = deploy_run_dir(args, tag, seed, budget)
                cta, asr, source = read_final(run_dir)
                if source is None:
                    missing.append((tag, seed, budget, run_dir))
                    continue
                rows.append(
                    {"tag": tag, "seed": seed, "budget": budget, "cta": cta, "asr": asr, "source": source}
                )
    return pd.DataFrame(rows), missing


def summarize(df, args):
    """mean/std/n per (tag, budget), reindexed so every requested cell shows up (NaN if absent)."""
    index = pd.MultiIndex.from_product([args.variant_tags, args.budgets], names=["tag", "budget"])
    if df.empty:
        return pd.DataFrame(index=index).reset_index()
    agg = df.groupby(["tag", "budget"])[list(METRICS)].agg(["mean", "std", "count"])
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    return agg.reindex(index).reset_index()


def tag_colors(tags):
    cmap = plt.get_cmap("tab10")
    return {tag: cmap(i % cmap.N) for i, tag in enumerate(tags)}


def draw_metric(ax, df, summary, metric, args, colors):
    """Budgets on evenly spaced positions: 0 ... 5000 on a linear axis crushes the small ones."""
    x = {budget: i for i, budget in enumerate(args.budgets)}
    for tag in args.variant_tags:
        cell = summary[summary["tag"] == tag].set_index("budget").reindex(args.budgets)
        mean, std = cell[f"{metric}_mean"].to_numpy(float), cell[f"{metric}_std"].to_numpy(float)
        positions = np.arange(len(args.budgets))

        if args.per_seed and not df.empty:
            for _, seed_df in df[df["tag"] == tag].groupby("seed"):
                seed_df = seed_df.sort_values("budget")
                ax.plot([x[b] for b in seed_df["budget"]], seed_df[metric], color=colors[tag],
                        linewidth=0.9, alpha=0.35, zorder=1)

        ax.plot(positions, mean, color=colors[tag], marker="o", markersize=5, label=tag, zorder=3)
        std = np.nan_to_num(std)  # a single seed has no std; draw the line without a band
        ax.fill_between(positions, mean - std, mean + std, color=colors[tag], alpha=0.18, zorder=2)

    ax.set_xticks(range(len(args.budgets)))
    ax.set_xticklabels([str(b) for b in args.budgets], rotation=45)
    plot_style.style_axes(ax, "Flip budget", plot_style.label(metric))
    ax.set_ylim(0, 100)


def fmt_cell(row, metric, n_seeds):
    mean, std, count = row[f"{metric}_mean"], row[f"{metric}_std"], row[f"{metric}_count"]
    if pd.isna(mean):
        return "--"
    std = 0.0 if pd.isna(std) else std
    flag = "" if count == n_seeds else f" (n={int(count)})"
    return f"{mean:.2f} +/- {std:.2f}{flag}"


def write_summary_table(summary, args, out_dir):
    n_seeds = len(args.seeds)
    header = ["tag", "budget", "CTA (%)", "ASR (%)"]
    lines_md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for _, row in summary.iterrows():
        lines_md.append(
            "| " + " | ".join([
                str(row["tag"]), str(row["budget"]),
                fmt_cell(row, "cta", n_seeds), fmt_cell(row, "asr", n_seeds),
            ]) + " |"
        )
    table_md = "\n".join(lines_md)
    (out_dir / "summary.md").write_text(table_md + "\n")
    summary.to_csv(out_dir / "summary.csv", index=False)
    return table_md


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp-root", default="experiments/federated_experiments")
    parser.add_argument("--exp-base-prefix", default="threat_model_direct_trigger_joint")
    parser.add_argument("--variant-tags", nargs="+", default=["warp_s8_lpips0p0", "warp_s16_lpips0p0"])
    parser.add_argument("--model-flag", default="r32p")
    parser.add_argument("--dataset", default="cifar")
    parser.add_argument("--gen-agg-method", default="mean", help="the agg_method the GEN cell was created under")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--budgets", nargs="+", type=int, default=[0, 150, 300, 500, 1000, 2000, 2500, 5000])
    parser.add_argument("--deploy-agg-method", default="multikrum")
    parser.add_argument("--deploy-num-poisoned", type=int, default=3)
    parser.add_argument("--deploy-num-honests", type=int, default=7)
    parser.add_argument("--per-seed", action="store_true", help="also draw each seed as a thin line")
    parser.add_argument("--out-dir", default="experiments/federated_experiments/warp_multibudget_summary")
    args = parser.parse_args()

    df, missing = collect(args)
    deploy_tag = f"{args.deploy_agg_method}_{args.deploy_num_poisoned}vs{args.deploy_num_honests}"
    if df.empty:
        print(f"[ABORT] no finished cell found under {args.exp_root}/{args.exp_base_prefix}_<tag>/.../{deploy_tag}/")
        print(f"        looked for {len(missing)} cells, e.g. {missing[0][3] if missing else '(none)'}")
        raise SystemExit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.sort_values(["tag", "seed", "budget"]).to_csv(out_dir / "per_seed.csv", index=False)

    summary = summarize(df, args)
    table_md = write_summary_table(summary, args, out_dir)

    plot_style.apply()
    colors = tag_colors(args.variant_tags)
    title = f"{args.model_flag}/{args.dataset}, {deploy_tag}, {len(args.seeds)} seeds"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric in zip(axes, METRICS):
        draw_metric(ax, df, summary, metric, args, colors)
    axes[0].legend(title="variant", loc="best")
    fig.suptitle(title)
    written = plot_style.save(fig, out_dir / "asr_cta_vs_budget")

    for metric in METRICS:
        fig, ax = plt.subplots(figsize=plot_style.SMALL_FIGSIZE)
        draw_metric(ax, df, summary, metric, args, colors)
        ax.legend(title="variant", loc="best")
        ax.set_title(title)
        written += plot_style.save(fig, out_dir / f"{metric}_vs_budget")

    print(f"[SUMMARY] {len(df)} finished cells over {len(args.variant_tags)} tag(s) x "
          f"{len(args.seeds)} seed(s) x {len(args.budgets)} budget(s), branch={deploy_tag}")
    sources = df["source"].value_counts().to_dict()
    if sources.get("json"):
        print(f"[NOTE] {sources['json']} cell(s) read from the tracker JSON (caccs/paccs .npy missing)")
    print(table_md)

    if missing:
        print(f"\n[MISSING] {len(missing)} cell(s) with no results yet:")
        for tag, seed, budget, run_dir in missing[:20]:
            print(f"  {tag} seed{seed} budget{budget}: {run_dir}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more (see per_seed.csv for what WAS read)")

    print("\n[OK] written:")
    for path in [out_dir / "per_seed.csv", out_dir / "summary.md", out_dir / "summary.csv", *written]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
