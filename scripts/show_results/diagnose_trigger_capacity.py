#!/usr/bin/env python3
"""Is the attack limited by the TRIGGER's capacity, by the label flips' ability to INDUCE it, or
by the AGGREGATOR? The three call for opposite work, and the existing plots cannot tell them
apart because they only show the deployed ASR.

The three measurements this script puts side by side, per (variant_tag, seed):

  expert ASR         GEN phase, `expert_asr` / `retrain_expert_asr` in
                     <cell>/gen_labels_trigger_joint/logs/generate_labels_trigger_joint_metrics.json
                     -- an expert trained WITH triggered images. It is the trigger's own ceiling:
                     how well this perturbation can backdoor this architecture at all.
  centralized ASR    <cell>/train_user_<budget>/, aggregator `mean` over 1 worker, i.e. a no-op.
                     The victim trains on CLEAN images with FLIPPED LABELS, so this measures
                     whether the flips can reproduce the trigger's feature at that budget.
  federated ASR      <cell>/<deploy-agg>_<P>vs<H>/train_user_<budget>/, the real deployment.

and the two gaps between them, which is where the diagnosis lives:

  expert -> centralized   the INDUCIBILITY gap. Large = the trigger works, but label flips on
                          clean data cannot reproduce the feature it rides. More trigger
                          expressivity makes this WORSE, not better: a more complex perturbation
                          is harder to induce, not easier.
  centralized -> federated  the AGGREGATION gap. Large = the flips do induce the backdoor, but
                          the robust aggregator filters the poisoned clients. The trigger is not
                          the lever here -- at deploy time the aggregator never even sees it
                          (federated_train_user poisons labels only) -- the flip budget and the
                          gradient-matching terms are.

A low expert ASR, and only that, means the trigger itself is the bottleneck and capacity is
worth adding.

Usage (from repo root, with the JOLT conda env active):
  python scripts/show_results/diagnose_trigger_capacity.py \\
      --variant-tags hybrid_eps0p0627_s8p0 --seeds 0 1 2 --budgets 150 500 1500

Outputs under --out-dir: verdict.md (read this first), per_seed.csv, gaps.{png,pdf}.
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
from plot_warp_multibudget import read_final  # noqa: E402

# An expert ASR below this means the perturbation does not reliably backdoor the architecture
# even when the victim is trained on it directly -- the one case where capacity is the problem.
EXPERT_ASR_FLOOR = 60.0
# Gaps wider than this (in ASR points) are called out as the dominant bottleneck.
GAP_THRESHOLD = 20.0


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


def read_gen_metrics(args, tag, seed, tail):
    """(expert_asr_final, expert_asr_peak, retrain_expert_asr_final) in %, NaN when absent.

    expert_asr is noisy from step to step (a fresh checkpoint is drawn each one), so the final
    value is the mean of the last `tail` points rather than the last point. retrain_expert_asr
    comes from expert_retrain_interval: a fresh expert actually trained against the current
    trigger, which is the trustworthier of the two when the run produced it."""
    path = cell_dir(args, tag, seed) / "gen_labels_trigger_joint" / "logs" / "generate_labels_trigger_joint_metrics.json"
    if not path.exists():
        return np.nan, np.nan, np.nan
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return np.nan, np.nan, np.nan

    def series(key):
        # ExperimentTracker stores the raw [0, 1] fraction; plot_style.scale is applied at plot
        # time only, so the x100 is ours to do -- same convention as plot_warp_multibudget.
        return np.array([p["value"] for p in data.get(key, [])], dtype=float) * 100

    expert = series("expert_asr")
    retrain = series("retrain_expert_asr")
    return (
        float(expert[-tail:].mean()) if expert.size else np.nan,
        float(expert.max()) if expert.size else np.nan,
        float(retrain[-tail:].mean()) if retrain.size else np.nan,
    )


def collect(args):
    rows, missing = [], []
    for tag in args.variant_tags:
        for seed in args.seeds:
            base = cell_dir(args, tag, seed)
            expert_final, expert_peak, retrain_final = read_gen_metrics(args, tag, seed, args.tail)
            deploy = f"{args.deploy_agg_method}_{args.deploy_num_poisoned}vs{args.deploy_num_honests}"
            for budget in args.budgets:
                cen_cta, cen_asr, cen_src = read_final(base / f"train_user_{budget}")
                fed_cta, fed_asr, fed_src = read_final(base / deploy / f"train_user_{budget}")
                if cen_src is None and fed_src is None and np.isnan(expert_final):
                    missing.append((tag, seed, budget, base))
                    continue
                rows.append({
                    "tag": tag, "seed": seed, "budget": budget,
                    "expert_asr": expert_final, "expert_asr_peak": expert_peak,
                    "retrain_expert_asr": retrain_final,
                    "centralized_asr": cen_asr, "centralized_cta": cen_cta,
                    "federated_asr": fed_asr, "federated_cta": fed_cta,
                    # The trigger's ceiling: the retrained expert when the run produced one
                    # (a real model trained on triggered images), else the matched expert.
                    "ceiling_asr": retrain_final if not np.isnan(retrain_final) else expert_final,
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["gap_inducibility"] = df["ceiling_asr"] - df["centralized_asr"]
        df["gap_aggregation"] = df["centralized_asr"] - df["federated_asr"]
        df = add_budget0_control(df, args)
    return df, missing


def add_budget0_control(df, args):
    """ASR net of the budget-0 cell, per (tag, seed).

    At budget 0 no label is flipped, so whatever ASR the trigger reaches there is NOT a backdoor:
    it is the trigger acting as a plain (universal) adversarial perturbation on the victim. Only
    the increment above it is attributable to the flips. Without this control a variant whose
    perturbation is simply more adversarial reads as a better backdoor, which is the opposite of
    what this campaign is trying to rank."""
    if args.baseline_budget not in args.budgets:
        df["federated_asr_net"] = np.nan
        df["centralized_asr_net"] = np.nan
        return df
    base = (
        df[df["budget"] == args.baseline_budget]
        .set_index(["tag", "seed"])[["centralized_asr", "federated_asr"]]
        .rename(columns=lambda c: c + "_base")
    )
    df = df.join(base, on=["tag", "seed"])
    df["federated_asr_net"] = df["federated_asr"] - df["federated_asr_base"]
    df["centralized_asr_net"] = df["centralized_asr"] - df["centralized_asr_base"]
    return df


def verdict_for(cell, baseline_budget=None):
    """The bottleneck for one (tag, budget) row of means, as (verdict, what to do)."""
    ceiling, cen, fed = cell["ceiling_asr"], cell["centralized_asr"], cell["federated_asr"]
    if baseline_budget is not None and cell["budget"] == baseline_budget:
        # No label is flipped here, so "the flips failed to induce it" is vacuously true and the
        # usual verdicts say nothing. What this row IS, is the control the others subtract.
        return (
            "control (no flips)",
            f"the trigger alone reaches {fed:.0f}% federated / {cen:.0f}% centralized with zero "
            "flipped labels -- that much of the attack is a plain adversarial perturbation, not "
            "an induced backdoor, and every other row's `net` column is measured against it",
        )
    if np.isnan(ceiling):
        return "no GEN metrics", "run the GEN phase, or point --exp-root at the right tree"
    if ceiling < EXPERT_ASR_FLOOR:
        return (
            "TRIGGER CAPACITY",
            f"the trigger only reaches {ceiling:.0f}% ASR on a victim trained directly on it -- "
            "this is the one case where adding expressivity is the right move",
        )
    if np.isnan(cen) and np.isnan(fed):
        return "no deploy results", "the GEN phase ran but no train_user cell finished yet"
    gap_ind = ceiling - cen if not np.isnan(cen) else np.nan
    gap_agg = cen - fed if not (np.isnan(cen) or np.isnan(fed)) else np.nan
    if not np.isnan(gap_ind) and gap_ind > GAP_THRESHOLD and (np.isnan(gap_agg) or gap_ind >= gap_agg):
        return (
            "INDUCIBILITY",
            f"the trigger reaches {ceiling:.0f}% but label flips only induce {cen:.0f}% "
            f"({gap_ind:.0f} pts lost) -- spend expressivity on perturbations flips can "
            "reproduce (low-frequency, crop/flip-invariant), not on raw capacity",
        )
    if not np.isnan(gap_agg) and gap_agg > GAP_THRESHOLD:
        return (
            "AGGREGATION",
            f"flips induce {cen:.0f}% centralized but only {fed:.0f}% federated "
            f"({gap_agg:.0f} pts lost) -- the aggregator never sees the trigger, so the levers "
            "are the flip budget and L_gradmatch/L_match/L_consistency",
        )
    return (
        "no dominant bottleneck",
        f"ceiling {ceiling:.0f}% -> centralized {cen:.0f}% -> federated {fed:.0f}%, both gaps "
        f"under {GAP_THRESHOLD:.0f} pts at this budget",
    )


def write_verdict(df, args, out_dir):
    cols = ["ceiling_asr", "expert_asr", "retrain_expert_asr", "centralized_asr", "federated_asr",
            "centralized_asr_net", "federated_asr_net", "gap_inducibility", "gap_aggregation"]
    means = df.groupby(["tag", "budget"])[cols].mean().reset_index()

    lines = [
        "# Trigger bottleneck diagnosis",
        "",
        f"`{args.model_flag}/{args.dataset}`, {len(args.seeds)} seed(s), deploy branch "
        f"`{args.deploy_agg_method}_{args.deploy_num_poisoned}vs{args.deploy_num_honests}`. "
        "ASR in %, mean across seeds.",
        "",
        f"`net` columns are the ASR minus the same (tag, seed)'s budget-{args.baseline_budget} "
        "cell: at that budget no label is flipped, so whatever ASR the trigger reaches there is "
        "the trigger acting as a plain adversarial perturbation, not an induced backdoor. **Rank "
        "variants on `net`, not on the raw ASR.**",
        "",
        "| tag | budget | trigger ceiling | centralized | federated | cen. net | fed. net | inducibility gap | aggregation gap | bottleneck |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    def fmt(v):
        return "--" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.1f}"

    details = []
    for _, row in means.iterrows():
        name, action = verdict_for(row, baseline_budget=args.baseline_budget)
        lines.append(
            f"| {row['tag']} | {row['budget']} | {fmt(row['ceiling_asr'])} | "
            f"{fmt(row['centralized_asr'])} | {fmt(row['federated_asr'])} | "
            f"{fmt(row['centralized_asr_net'])} | {fmt(row['federated_asr_net'])} | "
            f"{fmt(row['gap_inducibility'])} | {fmt(row['gap_aggregation'])} | **{name}** |"
        )
        details.append(f"- **{row['tag']}**, budget {row['budget']}: **{name}** -- {action}")

    lines += ["", "## What each verdict means for the next experiment", ""] + details
    lines += [
        "",
        "## Why the trigger cannot fix an AGGREGATION verdict",
        "",
        "At deploy time `federated_train_user` builds each worker's dataset from CLEAN images with "
        "flipped labels; the trigger is only ever applied to the ASR test set. The robust "
        "aggregator therefore never sees a triggered gradient, and no trigger -- however "
        "expressive -- evades it directly. The only indirect path is a trigger that needs FEWER "
        "flips to be induced, which leaves the poisoned clients' gradients closer to the honest "
        "ones: read the low-budget rows above, not the high-budget ones.",
        "",
        f"Thresholds: trigger ceiling below {EXPERT_ASR_FLOOR:.0f}% = capacity-limited; a gap wider "
        f"than {GAP_THRESHOLD:.0f} ASR points = dominant bottleneck. The ceiling is "
        "`retrain_expert_asr` (a fresh expert trained against the current trigger) when the run "
        f"produced one, else `expert_asr` averaged over its last {args.tail} steps.",
        "",
    ]
    text = "\n".join(lines)
    (out_dir / "verdict.md").write_text(text)
    means.to_csv(out_dir / "verdict.csv", index=False)
    return text, means


def draw(means, args, out_dir):
    """One grouped bar panel per tag: the three ASRs side by side, so the gaps are the thing you
    see rather than something to subtract by hand."""
    plot_style.apply()
    tags = list(means["tag"].unique())
    fig, axes = plt.subplots(1, len(tags), figsize=(max(6.0, 5.0 * len(tags)), 4.6), squeeze=False)
    stages = [("ceiling_asr", "Trigger ceiling\n(expert trained on it)"),
              ("centralized_asr", "Centralized\n(flips, no aggregator)"),
              ("federated_asr", "Federated\n(real deployment)")]
    colors = ["tab:gray", "tab:blue", "tab:red"]

    for ax, tag in zip(axes[0], tags):
        cell = means[means["tag"] == tag].set_index("budget").reindex(args.budgets)
        width = 0.8 / len(stages)
        positions = np.arange(len(args.budgets))
        for i, ((key, label), color) in enumerate(zip(stages, colors)):
            ax.bar(positions + i * width - 0.4 + width / 2, cell[key].to_numpy(float),
                   width=width, color=color, label=label)
        ax.set_xticks(positions)
        ax.set_xticklabels([str(b) for b in args.budgets])
        ax.set_ylim(0, 100)
        plot_style.style_axes(ax, "Flip budget", "ASR (%)", title=tag)
        ax.axhline(EXPERT_ASR_FLOOR, color="black", linestyle=":", linewidth=1)

    # Figure-level, below the panels: an in-axes legend lands on top of the bars exactly in the
    # panels worth looking at (the ones where the first bar is tall).
    # Anchored just BELOW the figure: plot_style.save calls tight_layout() itself (so a rect
    # passed here would be discarded) and then saves with bbox_inches="tight", which grows the
    # canvas to include the legend instead of letting it land on the bars or the tick labels.
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0),
               ncol=len(stages), fontsize=8, frameon=False)
    fig.suptitle(
        f"{args.model_flag}/{args.dataset} -- where the ASR is lost "
        f"({len(args.seeds)} seeds, {args.deploy_agg_method}_"
        f"{args.deploy_num_poisoned}vs{args.deploy_num_honests})"
    )
    return plot_style.save(fig, out_dir / "gaps")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-root", default="experiments/federated_experiments")
    ap.add_argument("--exp-base-prefix", default="threat_model_direct_trigger_joint")
    ap.add_argument("--variant-tags", nargs="+", required=True)
    ap.add_argument("--model-flag", default="r32p")
    ap.add_argument("--dataset", default="cifar")
    ap.add_argument("--gen-agg-method", default="mean")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--budgets", nargs="+", type=int, default=[150, 500, 1500])
    ap.add_argument("--deploy-agg-method", default="multikrum")
    ap.add_argument("--deploy-num-poisoned", type=int, default=3)
    ap.add_argument("--deploy-num-honests", type=int, default=7)
    ap.add_argument("--tail", type=int, default=50,
                    help="expert_asr is averaged over its last this-many optimization steps")
    ap.add_argument("--baseline-budget", type=int, default=0,
                    help="the no-flip control subtracted to get the `net` columns; must be in "
                         "--budgets for them to be computed (they are NaN otherwise)")
    ap.add_argument("--out-dir", default="experiments/federated_experiments/trigger_diagnosis")
    args = ap.parse_args()

    df, missing = collect(args)
    if df.empty:
        print(f"[ABORT] nothing read under {args.exp_root}/{args.exp_base_prefix}_<tag>/")
        if missing:
            print(f"        looked for {len(missing)} cells, e.g. {missing[0][3]}")
        raise SystemExit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.sort_values(["tag", "seed", "budget"]).to_csv(out_dir / "per_seed.csv", index=False)
    text, means = write_verdict(df, args, out_dir)
    written = draw(means, args, out_dir)

    print(text)
    if missing:
        print(f"[MISSING] {len(missing)} cell(s) with neither GEN nor deploy results:")
        for tag, seed, budget, base in missing[:10]:
            print(f"  {tag} seed{seed} budget{budget}: {base}")
    print("[OK] written:")
    for path in [out_dir / "verdict.md", out_dir / "verdict.csv", out_dir / "per_seed.csv", *written]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
