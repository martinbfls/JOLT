#!/usr/bin/env python3
"""Paper-ready training / generation curves, mean +- std over seeds, from the per-run metrics
ExperimentTracker writes (experiments/<run>/logs/<module>_metrics.json).

One figure per metric, one line per (campaign, tag): JOLT tags are solid, the FLIP
frozen-trigger campaign (gen_configs_flip_frozen_trigger.py) dashed, in the same color as the tag.

  # trigger/label generation: expert ASR, losses, trigger norm along the optimization
  python scripts/show_results/plot_training_curves.py --stage gen --tags baseline eps_16_255
  # user training: CTA/ASR per epoch for one deployment cell, JOLT vs FLIP with the same trigger
  python scripts/show_results/plot_training_curves.py --stage train_user --agg multikrum --budget 1000 \\
      --campaigns jolt flip_frozen
  # experts trained on the frozen trigger
  python scripts/show_results/plot_training_curves.py --stage train_expert --campaigns flip_frozen
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modules.base_utils import plot_style  # noqa: E402
from modules.base_utils.plot_style import plt  # noqa: E402
from modules.generate_labels_trigger_joint import gen_configs_flip_frozen_trigger as flip_frozen  # noqa: E402
from modules.generate_labels_trigger_joint.gen_configs_paper_main_campaign import (  # noqa: E402
    EXP_BASE as JOLT_EXP_BASE,
    DEPLOY_SINGLE_USER_AGG_METHOD,
    FEDERATED_TAG,
    SEEDS,
    bootstrap_cell_name,
    cell_name,
)

CAMPAIGNS = {"jolt": JOLT_EXP_BASE, "flip_frozen": flip_frozen.EXP_BASE}
CAMPAIGN_LABEL = {"jolt": "JOLT", "flip_frozen": "FLIP, frozen JOLT trigger"}
CAMPAIGN_LINESTYLE = {"jolt": "-", "flip_frozen": "--"}
TAG_LABEL = {"baseline": r"$\epsilon$=1.0", "eps_16_255": r"$\epsilon$=16/255",
             "eps_0p50_lpips_0p1": r"$\epsilon$=0.5, LPIPS=0.1"}
TAG_COLOR = {"baseline": "tab:blue", "eps_16_255": "tab:orange", "eps_0p50_lpips_0p1": "tab:green"}
X_LABEL = {"gen": "Optimization step", "train_user": "Epoch", "train_expert": "Epoch"}
DEFAULT_METRICS = {
    "gen": ["expert_asr", "retrain_expert_asr", "matching_term", "L_bd_mean", "g_loss", "delta_l2", "cos_target"],
    "train_user": ["asr", "cta", "grad_cosine_sim_poisoned_honest"],
    "train_expert": ["asr", "cta"],
}


def run_dir(stage, campaign, model, dataset, tag, seed, agg=None, budget=None):
    base = CAMPAIGNS[campaign]
    cell = base / cell_name(model, dataset, tag, seed)
    if stage == "gen":
        return cell / ("gen_labels_trigger_joint" if campaign == "jolt" else "gen_labels")
    if stage == "train_expert":
        if campaign == "jolt":  # shared bootstrap expert, independent of the tag
            return base / bootstrap_cell_name(model, dataset, seed)
        return base / flip_frozen.expert_cell_name(model, dataset, tag, seed)
    branch = cell / DEPLOY_SINGLE_USER_AGG_METHOD if agg == "centralized" else cell / FEDERATED_TAG / agg
    return branch / f"train_user_{budget}"


def load_metrics(directory):
    files = sorted((directory / "logs").glob("*_metrics.json"))
    if not files:
        return {}
    with open(files[0]) as f:
        data = json.load(f)
    return {k: (np.array([p["step"] for p in v]), np.array([p["value"] for p in v], dtype=float))
            for k, v in data.items() if isinstance(v, list) and v and isinstance(v[0], dict)}


def mean_std(runs, metric):
    """Mean/std across seeds on the steps common to every seed that logged `metric`."""
    series = [r[metric] for r in runs if metric in r]
    if not series:
        return None
    steps = series[0][0]
    for s, _ in series[1:]:
        steps = np.intersect1d(steps, s)
    if len(steps) == 0:
        return None
    vals = np.stack([v[np.searchsorted(s, steps)] for s, v in series]) * plot_style.scale(metric)
    return steps, vals.mean(0), vals.std(0), len(series)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=list(X_LABEL), required=True)
    parser.add_argument("--campaigns", nargs="+", choices=list(CAMPAIGNS), default=["jolt"])
    parser.add_argument("--model", default="r32p")
    parser.add_argument("--datasets", nargs="+", default=["cifar"])
    parser.add_argument("--tags", nargs="+", default=["baseline", "eps_16_255"])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--metrics", nargs="+", default=None, help="default depends on --stage")
    parser.add_argument("--agg", default="centralized",
                        help="train_user only: 'centralized' (single_user branch) or a federated agg_method")
    parser.add_argument("--budget", type=int, default=1000, help="train_user only")
    parser.add_argument("--smooth", type=int, default=None,
                        help="rolling-mean window (default: automatic for dense per-step curves)")
    parser.add_argument("--out-dir", default="./plots_training_curves")
    args = parser.parse_args()

    plot_style.apply()
    metrics = args.metrics or DEFAULT_METRICS[args.stage]
    tags = args.tags if not (args.stage == "train_expert" and args.campaigns == ["jolt"]) else args.tags[:1]

    for dataset in args.datasets:
        runs = {}
        for campaign in args.campaigns:
            for tag in tags:
                loaded = [load_metrics(run_dir(args.stage, campaign, args.model, dataset, tag, seed,
                                               args.agg, args.budget)) for seed in args.seeds]
                runs[(campaign, tag)] = [r for r in loaded if r]
                print(f"[{dataset}] {campaign}/{tag}: {len(runs[(campaign, tag)])}/{len(args.seeds)} seeds with metrics")

        cell = f"{args.stage}_{args.model}_{dataset}" + (
            f"_{args.agg}_{args.budget}" if args.stage == "train_user" else "")
        for metric in metrics:
            fig, ax = plt.subplots(figsize=plot_style.FIGSIZE)
            drawn = 0
            for (campaign, tag), tag_runs in runs.items():
                stats = mean_std(tag_runs, metric)
                if stats is None:
                    continue
                steps, mean, std, n = stats
                window = args.smooth or plot_style.auto_window(len(steps))
                mean, std = plot_style.smooth(mean, window), plot_style.smooth(std, window)
                color = TAG_COLOR.get(tag, f"C{drawn}")
                jolt_only_expert = args.stage == "train_expert" and campaign == "jolt"
                label = CAMPAIGN_LABEL[campaign] if jolt_only_expert else (
                    f"{CAMPAIGN_LABEL[campaign]} ({TAG_LABEL.get(tag, tag)})")
                ax.plot(steps, mean, color=color, linestyle=CAMPAIGN_LINESTYLE[campaign], label=f"{label}, n={n}")
                ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
                drawn += 1
            if not drawn:
                plt.close(fig)
                continue
            plot_style.style_axes(ax, X_LABEL[args.stage], plot_style.label(metric),
                                  f"{args.model} / {dataset}" + (
                                      f" / {args.agg}, budget {args.budget}" if args.stage == "train_user" else ""))
            ax.legend(frameon=True, fontsize=10, loc="best")
            for path in plot_style.save(fig, Path(args.out_dir) / f"{cell}_{metric}"):
                print(f"[INFO] Saved {path}")


if __name__ == "__main__":
    main()
