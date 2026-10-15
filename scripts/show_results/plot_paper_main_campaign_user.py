#!/usr/bin/env python3
"""
Plots + LaTeX tables + trigger visuals for the paper_main_campaign TRAIN_USER results:

  experiments/federated_experiments/threat_model_direct_trigger_joint_paper_main_campaign/
    {model_flag}/{dataset}/{tag}/seed{seed}/
      gen_labels_trigger_joint/trigger/                <- the generated trigger (.pt + .png)
      mean/train_user_{budget}/                         <- single_user branch (1v0/mean)
      federated_3vs7/{agg}/train_user_{budget}/         <- federated branch (3v7/agg)

Usage:
  python scripts/show_results/plot_paper_main_campaign_user.py
  python scripts/show_results/plot_paper_main_campaign_user.py --model r32p \
      --tags baseline eps_16_255 --datasets cifar svhn
  # + FLIP re-run with each JOLT trigger frozen (gen_configs_flip_frozen_trigger.py):
  python scripts/show_results/plot_paper_main_campaign_user.py --flip-frozen-tags baseline eps_16_255
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modules.base_utils.datasets import load_dataset, pick_poisoner
from modules.generate_labels_trigger_joint.gen_configs_paper_main_campaign import (
    EXP_BASE,
    MODEL_FLAGS,
    DATASETS,
    SEEDS,
    DEPLOY_BUDGETS,
    DEPLOY_SINGLE_USER_AGG_METHOD,
    FEDERATED_TAG,
    DEPLOY_AGG_METHODS_FEDERATED,
    GEN_NUM_POISONED,
    GEN_NUM_HONESTS,
    SOURCE_LABEL,
    TARGET_LABEL,
    GEN_INIT,
    cell_name,
)
from modules.generate_labels_trigger_joint.gen_configs_flip_frozen_trigger import (
    EXP_BASE as FLIP_FROZEN_EXP_BASE,
)

# =========================
# Style (Paper-ready)
# =========================
plt.rcParams.update(
    {
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "lines.linewidth": 2.5,
    }
)

BRANCH_COLORS = {
    "single_user": "black",
    "federated_mean": "tab:blue",
    "federated_median": "tab:orange",
    "federated_krum": "tab:green",
    "federated_trmean": "tab:red",
    "federated_multikrum": "tab:purple",
}

JOLT_LABEL = "JOLT"

BRANCH_LABELS = {
    "single_user": JOLT_LABEL,
    "federated_mean": JOLT_LABEL,
    "federated_median": JOLT_LABEL,
    "federated_krum": JOLT_LABEL,
    "federated_trmean": JOLT_LABEL,
    "federated_multikrum": JOLT_LABEL,
}

# Filename suffix per branch, matching the LaTeX subfigure grid's expected {dataset}_{suffix}.png.
BRANCH_FILE_SUFFIX = {
    "single_user": "centralized",
    "federated_mean": "mean",
    "federated_median": "median",
    "federated_krum": "krum",
    "federated_trmean": "trmean",
    "federated_multikrum": "multikrum",
}

# FLIP baseline (its own 1xs sinusoidal trigger) comparison series, paired against its JOLT
# counterpart branch on the same plot (dotted instead of solid; see _branch_linestyle).
FLIP_LABEL = "FLIP"
FLIP_ATTACK = "backdoor"
FLIP_POISONER_FLAG = "1xs"
for _agg in DEPLOY_AGG_METHODS_FEDERATED:
    BRANCH_COLORS[f"flip_{_agg}"] = BRANCH_COLORS[f"federated_{_agg}"]
    BRANCH_LABELS[f"flip_{_agg}"] = FLIP_LABEL
    BRANCH_FILE_SUFFIX[f"flip_{_agg}"] = f"federated_{_agg}"

BRANCH_COLORS["flip_single_user"] = BRANCH_COLORS["single_user"]
BRANCH_LABELS["flip_single_user"] = FLIP_LABEL
BRANCH_FILE_SUFFIX["flip_single_user"] = BRANCH_FILE_SUFFIX["single_user"]

# "centralized" is a synthetic series read from the single_user branch, not a real aggregator.
CENTRALIZED_SERIES = "centralized"
TABLE_AGG_ORDER = ["mean", "krum", "trmean", "multikrum"]
TABLE_COLUMNS_WITH_CENTRALIZED = [CENTRALIZED_SERIES] + TABLE_AGG_ORDER
TABLE_AGG_DISPLAY = {
    CENTRALIZED_SERIES: "CENTRALIZED",
    "mean": "MEAN",
    "median": "CW-MEDIAN",
    "krum": "KRUM",
    "trmean": "TRMEAN",
    "multikrum": "MULTIKRUM",
}

# Tag -> table block name.
TAG_DISPLAY_NAMES = {
    "baseline": "Optimized Trigger ($\\varepsilon=1.0$) -- \\jolt{}",
    "eps_0p50_lpips_0p1": "Optimized Trigger ($\\varepsilon=0.5$, LPIPS$=0.1$) -- \\jolt{}",
    "eps_16_255": "Optimized Trigger ($\\varepsilon=16/255$) -- \\jolt{}",
}

# FLIP re-run with each tag's JOLT trigger frozen (gen_configs_flip_frozen_trigger.py).
FLIP_FROZEN_TAG_DISPLAY_NAMES = {
    tag: name.replace("-- \\jolt{}", "-- FLIP") for tag, name in TAG_DISPLAY_NAMES.items()
}

# FLIP has no stealth-regularized variants, so it gets exactly one table block.
FLIP_BLOCK_NAME = f"Sinusoidal Trigger -- {FLIP_LABEL}"

DATASET_DISPLAY = {"cifar": "CIFAR-10", "svhn": "SVHN"}
MODEL_DISPLAY = {"r32p": "ResNet-32", "convnext_micro": "ConvNeXt-Micro"}

# =========================
# Multi-tag overlay style: plot_cta_vs_pta_per_branch overlays every requested JOLT tag plus
# FLIP on the same per-aggregator figure, each getting its own shade of BRANCH_COLORS's hue.
TAG_SHORT_DISPLAY = {
    "baseline": f"{JOLT_LABEL} ($\\epsilon$=1.0)",
    "eps_16_255": f"{JOLT_LABEL} ($\\epsilon$=16/255)",
    "eps_0p50_lpips_0p1": f"{JOLT_LABEL} ($\\epsilon$=0.5, LPIPS=0.1)",
}
TAG_LINESTYLES = {
    "baseline": "-",
    "eps_16_255": "--",
    "eps_0p50_lpips_0p1": "-.",
}
TAG_MARKERS = {
    "baseline": "o",
    "eps_16_255": "s",
    "eps_0p50_lpips_0p1": "^",
}
# Fallback cycles for any tag not listed above, so a new/unknown tag still renders distinctly
# instead of crashing or silently colliding with an existing one.
_FALLBACK_LINESTYLES = ["-", "--", "-."]
_FALLBACK_MARKERS = ["o", "s", "^", "D", "v"]
FLIP_LINESTYLE = ":"
FLIP_MARKER = "D"  # filled diamond -- 'x' has no fill, so it can't take the best-point black edge


def _tag_short_display(tag):
    return TAG_SHORT_DISPLAY.get(tag, f"{JOLT_LABEL} ({tag})")


def _tag_linestyle(tag, idx=0):
    return TAG_LINESTYLES.get(tag, _FALLBACK_LINESTYLES[idx % len(_FALLBACK_LINESTYLES)])


def _tag_marker(tag, idx=0):
    return TAG_MARKERS.get(tag, _FALLBACK_MARKERS[idx % len(_FALLBACK_MARKERS)])


def _lighten(color, amount):
    """Blend `color` toward white by `amount` in [0, 1] (0 = unchanged, 1 = white)."""
    r, g, b = mcolors.to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def _darken(color, amount):
    """Blend `color` toward black by `amount` in [0, 1] (0 = unchanged, 1 = black)."""
    r, g, b = mcolors.to_rgb(color)
    return (r * (1 - amount), g * (1 - amount), b * (1 - amount))


def _tag_color(base_color, tag_idx):
    """Per-tag shade of an aggregator's base color: the first tag (tag_idx=0, e.g. baseline)
    keeps the base color, later tags are progressively shaded away from it. Combined with
    linestyle+marker, this keeps close/overlapping curves on the same per-aggregator figure
    legible without giving up the aggregator's own hue (so the figure still reads as "this is
    MEAN" etc. across the whole set of per-aggregator plots). Darkens by default; for an already
    near-black base (single_user's color), darkening would be a no-op, so those shade by
    lightening instead."""
    if tag_idx == 0:
        return base_color
    amount = min(0.5, 0.3 * tag_idx)
    r, g, b = mcolors.to_rgb(base_color)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return _lighten(base_color, amount) if luminance < 0.25 else _darken(base_color, amount)


def _flip_color(base_color):
    """FLIP gets a clearly lighter/washed-out tint of the same aggregator hue."""
    return _lighten(base_color, 0.55)


# =========================
# Utils
# =========================
def get_final_value(npy_path):
    try:
        if not os.path.exists(npy_path):
            return np.nan
        if os.path.getsize(npy_path) == 0:
            return np.nan

        data = np.load(npy_path, allow_pickle=True)

        if data.size == 0:
            return np.nan

        if data.ndim > 1:
            return float(data[-1][0])

        return float(data[-1])

    except Exception:
        return np.nan


# =========================
# COMPUTE
# =========================
def _branch_run_dir(model_flag, dataset, tag, seed, branch, agg=None, exp_base=EXP_BASE):
    cell_dir = exp_base / cell_name(model_flag, dataset, tag, seed)
    if branch == "single_user":
        return cell_dir / DEPLOY_SINGLE_USER_AGG_METHOD
    return cell_dir / FEDERATED_TAG / agg


def compute_cta_pta_mean_var(
    model_flag,
    dataset,
    tag,
    branch,
    budgets,
    seeds,
    agg=None,
    cta_file="caccs.npy",
    pta_file="paccs.npy",
    exp_base=EXP_BASE,
):
    """Averages caccs.npy/paccs.npy's final value across `seeds` for each budget; `exp_base` overridable via --jolt-root."""
    records = []

    for budget in budgets:
        cta_vals, pta_vals = [], []
        for seed in seeds:
            run_dir = (
                _branch_run_dir(model_flag, dataset, tag, seed, branch, agg, exp_base=exp_base) / f"train_user_{budget}"
            )
            cta_path = run_dir / cta_file
            pta_path = run_dir / pta_file
            if not (cta_path.exists() and pta_path.exists()):
                continue
            cta_vals.append(get_final_value(cta_path))
            pta_vals.append(get_final_value(pta_path))

        records.append(
            {
                "dataset": dataset,
                "branch": branch,
                "agg": agg,
                "budget": budget,
                "model": model_flag,
                "cta_mean": np.mean(cta_vals) if cta_vals else np.nan,
                "cta_var": np.var(cta_vals) if cta_vals else np.nan,
                "pta_mean": np.mean(pta_vals) if pta_vals else np.nan,
                "pta_var": np.var(pta_vals) if pta_vals else np.nan,
            }
        )

    return pd.DataFrame.from_records(records)


def _flip_run_dir(flip_root, model_flag, dataset, agg, seed, budget):
    tag_suffix = FEDERATED_TAG.split("_", 1)[-1]  # "federated_3vs7" -> "3vs7"
    return (
        Path(flip_root)
        / model_flag
        / f"{tag_suffix}_FLIP"
        / dataset
        / FLIP_ATTACK
        / agg
        / FLIP_POISONER_FLAG
        / str(seed)
        / str(budget)
    )


def _flip_centralized_run_dir(flip_root, model_flag, dataset, seed, budget):
    """Centralized (1v0, undefended) counterpart of _flip_run_dir."""
    return (
        Path(flip_root)
        / model_flag
        / f"{GEN_NUM_POISONED}vs{GEN_NUM_HONESTS}"
        / dataset
        / FLIP_ATTACK
        / DEPLOY_SINGLE_USER_AGG_METHOD
        / FLIP_POISONER_FLAG
        / str(seed)
        / str(budget)
    )


def compute_cta_pta_mean_var_flip(
    flip_root,
    model_flag,
    dataset,
    agg,
    budgets,
    seeds,
    cta_file="caccs.npy",
    pta_file="paccs.npy",
):
    """Same as compute_cta_pta_mean_var, but reads the FLIP baseline's own results tree."""
    records = []

    for budget in budgets:
        cta_vals, pta_vals = [], []
        for seed in seeds:
            run_dir = _flip_run_dir(flip_root, model_flag, dataset, agg, seed, budget)
            cta_path = run_dir / cta_file
            pta_path = run_dir / pta_file
            if not (cta_path.exists() and pta_path.exists()):
                continue
            cta_vals.append(get_final_value(cta_path))
            pta_vals.append(get_final_value(pta_path))

        records.append(
            {
                "dataset": dataset,
                "branch": f"flip_{agg}",
                "agg": agg,
                "budget": budget,
                "model": model_flag,
                "cta_mean": np.mean(cta_vals) if cta_vals else np.nan,
                "cta_var": np.var(cta_vals) if cta_vals else np.nan,
                "pta_mean": np.mean(pta_vals) if pta_vals else np.nan,
                "pta_var": np.var(pta_vals) if pta_vals else np.nan,
            }
        )

    return pd.DataFrame.from_records(records)


def compute_cta_pta_mean_var_flip_centralized(
    flip_root,
    model_flag,
    dataset,
    budgets,
    seeds,
    cta_file="caccs.npy",
    pta_file="paccs.npy",
):
    """Centralized counterpart of compute_cta_pta_mean_var_flip, reading _flip_centralized_run_dir
    instead (FLIP's own 1xs-trigger results in the undefended single-victim setting)."""
    records = []

    for budget in budgets:
        cta_vals, pta_vals = [], []
        for seed in seeds:
            run_dir = _flip_centralized_run_dir(flip_root, model_flag, dataset, seed, budget)
            cta_path = run_dir / cta_file
            pta_path = run_dir / pta_file
            if not (cta_path.exists() and pta_path.exists()):
                continue
            cta_vals.append(get_final_value(cta_path))
            pta_vals.append(get_final_value(pta_path))

        records.append(
            {
                "dataset": dataset,
                "branch": "flip_single_user",
                "agg": None,
                "budget": budget,
                "model": model_flag,
                "cta_mean": np.mean(cta_vals) if cta_vals else np.nan,
                "cta_var": np.var(cta_vals) if cta_vals else np.nan,
                "pta_mean": np.mean(pta_vals) if pta_vals else np.nan,
                "pta_var": np.var(pta_vals) if pta_vals else np.nan,
            }
        )

    return pd.DataFrame.from_records(records)


def collect_dataset(
    model_flag,
    dataset,
    tag,
    budgets=DEPLOY_BUDGETS,
    seeds=SEEDS,
    flip_root=None,
    exp_base=EXP_BASE,
):
    """{branch_key: DataFrame} for single_user + every federated aggregator, one dataset; adds flip_{agg} entries when flip_root is given."""
    all_data = {
        "single_user": compute_cta_pta_mean_var(
            model_flag,
            dataset,
            tag,
            "single_user",
            budgets,
            seeds,
            exp_base=exp_base,
        ),
    }
    if flip_root:
        all_data["flip_single_user"] = compute_cta_pta_mean_var_flip_centralized(
            flip_root,
            model_flag,
            dataset,
            budgets,
            seeds,
        )
    for agg in DEPLOY_AGG_METHODS_FEDERATED:
        all_data[f"federated_{agg}"] = compute_cta_pta_mean_var(
            model_flag,
            dataset,
            tag,
            "federated",
            budgets,
            seeds,
            agg=agg,
            exp_base=exp_base,
        )
        if flip_root:
            all_data[f"flip_{agg}"] = compute_cta_pta_mean_var_flip(
                flip_root,
                model_flag,
                dataset,
                agg,
                budgets,
                seeds,
            )
    return all_data


def collect_flip_data(flip_root, model_flag, dataset, budgets=DEPLOY_BUDGETS, seeds=SEEDS):
    """{branch_key: DataFrame} for the FLIP baseline only, fetched once per dataset and overlaid on every tag's figure."""
    all_data = {
        "flip_single_user": compute_cta_pta_mean_var_flip_centralized(
            flip_root,
            model_flag,
            dataset,
            budgets,
            seeds,
        ),
    }
    for agg in DEPLOY_AGG_METHODS_FEDERATED:
        all_data[f"flip_{agg}"] = compute_cta_pta_mean_var_flip(
            flip_root,
            model_flag,
            dataset,
            agg,
            budgets,
            seeds,
        )
    return all_data


# =========================
# PLOT
# =========================
def annotate_key_points(df, x, y, score, color):
    budgets = df["budget"].values

    if np.all(np.isnan(score)):
        return

    max_score = np.nanmax(score)
    candidates = np.where(np.isclose(score, max_score, atol=1e-12))[0]
    idx_best = candidates[np.argmin(budgets[candidates])]

    x_range = max(np.nanmax(x) - np.nanmin(x), 1e-6)
    y_range = max(np.nanmax(y) - np.nanmin(y), 1e-6)
    dx = 0.03 * x_range
    dy = 0.03 * y_range

    if np.isnan(x[idx_best]) or np.isnan(y[idx_best]):
        return

    plt.text(
        x[idx_best] + dx,
        y[idx_best] + dy,
        str(int(budgets[idx_best])),
        fontsize=11,
        fontweight="bold",
        color=color,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, alpha=0.8),
        zorder=10,
    )


def _branch_linestyle(branch_key):
    """single_user is dashed, FLIP is dotted, everything else solid."""
    if branch_key == "single_user":
        return "--"
    if branch_key.startswith("flip_"):
        return ":"
    return "-"


def plot_cta_vs_pta(all_data, dataset, model_flag, tag, save_dir=None, filename=None):
    """One figure, one line per branch (single_user + each federated aggregator)."""
    plt.figure(figsize=(7.5, 6))
    any_data = False

    for branch_key, df in all_data.items():
        # subset= avoids dropping every row: single_user's "agg" column is always None.
        df = df.dropna(subset=["cta_mean", "cta_var", "pta_mean", "pta_var"])
        if df.empty:
            continue
        any_data = True

        df = df.sort_values("budget")

        x = df["pta_mean"].values * 100
        y = df["cta_mean"].values * 100

        xerr = np.sqrt(df["pta_var"].values) * 100
        yerr = np.sqrt(df["cta_var"].values) * 100

        color = BRANCH_COLORS.get(branch_key)
        linestyle = _branch_linestyle(branch_key)

        plt.plot(
            x,
            y,
            linestyle=linestyle,
            linewidth=2.2,
            color=color,
            alpha=0.85,
            label=BRANCH_LABELS.get(branch_key, branch_key),
            zorder=3,
        )

        plt.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="none",
            ecolor=color,
            elinewidth=1.2,
            capsize=3,
            alpha=0.35,
            zorder=1,
        )

        plt.scatter(x, y, s=45, color=color, edgecolors="none", zorder=4)

        score = x  # ASR
        max_score = np.nanmax(score)
        candidates = np.where(np.isclose(score, max_score, atol=1e-12))[0]
        budgets = df["budget"].values
        idx_best = candidates[np.argmin(budgets[candidates])]

        plt.scatter(
            x[idx_best],
            y[idx_best],
            s=95,
            color=color,
            edgecolor="black",
            linewidth=1.2,
            zorder=6,
        )

        annotate_key_points(df, x, y, score, color)

    if not any_data:
        plt.close()
        return False

    plt.xlabel("ASR (%)")
    plt.ylabel("CTA (%)")
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True, linestyle="--", alpha=0.25)
    plt.legend(frameon=True, fontsize=10, loc="lower left")
    plt.title(f"{model_flag} / {dataset} / tag={tag}")
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename or f"{dataset}.png")
        plt.savefig(path, dpi=300)
        print(f"[INFO] Saved plot: {path}")

    plt.close()
    return True


def _draw_tradeoff_series(df, color, linestyle, label=None, marker="o"):
    """Draws one branch's CTA-vs-ASR curve onto the current figure; returns False if df has no complete cell."""
    df = df.dropna(subset=["cta_mean", "cta_var", "pta_mean", "pta_var"])
    if df.empty:
        return False

    df = df.sort_values("budget")

    x = df["pta_mean"].values * 100
    y = df["cta_mean"].values * 100
    xerr = np.sqrt(df["pta_var"].values) * 100
    yerr = np.sqrt(df["cta_var"].values) * 100

    plt.plot(
        x,
        y,
        linestyle=linestyle,
        linewidth=2.2,
        color=color,
        alpha=0.85,
        label=label,
        marker=marker,
        markersize=5,
        markerfacecolor=color,
        markeredgecolor="none",
        zorder=3,
    )
    plt.errorbar(
        x,
        y,
        xerr=xerr,
        yerr=yerr,
        fmt="none",
        ecolor=color,
        elinewidth=1.2,
        capsize=3,
        alpha=0.35,
        zorder=1,
    )

    score = x  # ASR
    max_score = np.nanmax(score)
    candidates = np.where(np.isclose(score, max_score, atol=1e-12))[0]
    budgets = df["budget"].values
    idx_best = candidates[np.argmin(budgets[candidates])]

    plt.scatter(
        x[idx_best],
        y[idx_best],
        s=110,
        marker=marker,
        color=color,
        edgecolor="black",
        linewidth=1.2,
        zorder=6,
    )
    annotate_key_points(df, x, y, score, color)
    return True


def plot_cta_vs_pta_per_branch(tag_data, flip_data, dataset, model_flag, tags, save_dir=None):
    """One figure per aggregator, overlaying every requested tag plus the FLIP baseline. Returns the list of paths written."""
    saved = []

    branch_keys = ["single_user"] + [f"federated_{agg}" for agg in DEPLOY_AGG_METHODS_FEDERATED]
    flip_key_for_branch = {
        "single_user": "flip_single_user",
        **{f"federated_{agg}": f"flip_{agg}" for agg in DEPLOY_AGG_METHODS_FEDERATED},
    }
    agg_suffix_for_branch = {"single_user": CENTRALIZED_SERIES}
    agg_suffix_for_branch.update({f"federated_{agg}": agg for agg in DEPLOY_AGG_METHODS_FEDERATED})

    for branch_key in branch_keys:
        base_color = BRANCH_COLORS[branch_key]
        series = []
        for tag_idx, tag in enumerate(tags):
            df = tag_data.get(tag, {}).get(branch_key)
            if df is None:
                continue
            series.append(
                (
                    df,
                    _tag_color(base_color, tag_idx),
                    _tag_linestyle(tag, tag_idx),
                    _tag_short_display(tag),
                    _tag_marker(tag, tag_idx),
                )
            )

        flip_key = flip_key_for_branch[branch_key]
        flip_df = flip_data.get(flip_key)
        if flip_df is not None:
            series.append((flip_df, _flip_color(base_color), FLIP_LINESTYLE, FLIP_LABEL, FLIP_MARKER))

        plt.figure(figsize=(7.5, 6))
        any_drawn = False
        n_labeled = 0
        for df, series_color, linestyle, label, marker in series:
            drawn = _draw_tradeoff_series(df, series_color, linestyle, label=label, marker=marker)
            any_drawn = any_drawn or drawn
            n_labeled += int(drawn)

        if not any_drawn:
            plt.close()
            print(
                f"[WARNING] branch={branch_key!r} has no complete (cta,pta) cell yet for "
                f"{model_flag}/{dataset}/tags={tags} -- skipping its plot."
            )
            continue

        plt.xlabel("ASR (%)")
        plt.ylabel("CTA (%)")
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.grid(True, linestyle="--", alpha=0.25)
        if n_labeled > 1:
            plt.legend(frameon=True, fontsize=10, loc="lower left")
        agg_display = TABLE_AGG_DISPLAY.get(agg_suffix_for_branch[branch_key], branch_key)
        plt.title(
            f"{MODEL_DISPLAY.get(model_flag, model_flag)} / {DATASET_DISPLAY.get(dataset, dataset)} -- {agg_display}"
        )
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            suffix = BRANCH_FILE_SUFFIX.get(branch_key, branch_key)
            path = os.path.join(save_dir, f"{dataset}_{suffix}.png")
            plt.savefig(path, dpi=300)
            print(f"[INFO] Saved plot: {path}")
            saved.append(path)

        plt.close()

    return saved


def plot_metric_vs_budget(all_data, dataset, model_flag, tag, metric, ylabel, save_dir=None, filename=None):
    """Budget-vs-metric view, one line per branch."""
    plt.figure(figsize=(7.5, 6))
    any_data = False

    for branch_key, df in all_data.items():
        df = df.dropna(subset=[f"{metric}_mean"])
        if df.empty:
            continue
        any_data = True

        df = df.sort_values("budget")
        budgets = df["budget"].values
        means = df[f"{metric}_mean"].values * 100
        stds = np.sqrt(df[f"{metric}_var"].values) * 100
        color = BRANCH_COLORS.get(branch_key)
        linestyle = _branch_linestyle(branch_key)

        plt.plot(
            budgets,
            means,
            marker="o",
            linestyle=linestyle,
            color=color,
            alpha=0.85,
            markersize=5,
            label=BRANCH_LABELS.get(branch_key, branch_key),
            zorder=3,
        )
        plt.fill_between(budgets, means - stds, means + stds, color=color, alpha=0.15, zorder=1)

    if not any_data:
        plt.close()
        return False

    plt.xlabel("Deployment budget (poisoned samples)")
    plt.ylabel(ylabel)
    plt.ylim(0, 100)
    plt.grid(True, linestyle="--", alpha=0.25)
    plt.legend(frameon=True, fontsize=10, loc="best")
    plt.title(f"{ylabel} vs budget -- {model_flag} / {dataset} / tag={tag}")
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename or f"{metric}_vs_budget.png")
        plt.savefig(path, dpi=300)
        print(f"[INFO] Saved plot: {path}")

    plt.close()
    return True


# =========================
# TABLE
# =========================
def format_cell(cta, cta_var, asr, asr_var):
    if np.isnan(cta):
        return "XXX"
    cta_std = np.sqrt(cta_var) * 100
    asr_std = np.sqrt(asr_var) * 100
    return f"{cta * 100:.1f}$\\pm${cta_std:.1f}/{asr * 100:.1f}$\\pm${asr_std:.1f}"


def compute_best_second(block, budgets, series_labels):
    """Best/second-best budget per column, ranked by ASR."""
    best, second = {}, {}

    for series in series_labels:
        values = np.array([block.get((b, series), (np.nan,) * 4)[2] for b in budgets])
        valid_idx = np.where(~np.isnan(values))[0]

        if len(valid_idx) == 0:
            best[series] = second[series] = None
            continue

        sorted_idx = valid_idx[np.argsort(values[valid_idx])]
        best[series] = budgets[sorted_idx[-1]]
        second[series] = budgets[sorted_idx[-2] if len(sorted_idx) > 1 else sorted_idx[-1]]

    return best, second


def render_block(name, block, budgets, series_labels):
    best, second = compute_best_second(block, budgets, series_labels)
    lines = [f"\\multicolumn{{{1 + len(series_labels)}}}{{c}}{{\\textbf{{{name}}}}} \\\\", "\\midrule"]

    for b in budgets:
        row = [str(b)]
        for series in series_labels:
            cell = format_cell(*block.get((b, series), (np.nan,) * 4))
            if best[series] == b:
                cell = f"\\textbf{{{cell}}}"
            elif second[series] == b:
                cell = f"\\underline{{{cell}}}"
            row.append(cell)
        lines.append(" & ".join(row) + " \\\\")

    return "\n".join(lines)


def build_table(blocks, budgets, series_labels, caption, label):
    """`blocks` is block name -> {(budget, series): (cta_mean, cta_var, pta_mean, pta_var)}, one multicolumn block per tag/variant."""
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "% \\tiny",
        "% \\setlength{\\tabcolsep}{1pt}",
        "",
        f"\\caption{{{caption}}}",
        "\\vspace{2mm}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{@{}" + " ".join(["c"] * (1 + len(series_labels))) + "@{}}",
        "\\toprule",
        "Budget & " + " & ".join(TABLE_AGG_DISPLAY.get(s, str(s).upper()) for s in series_labels) + " \\\\",
    ]

    for name, block in blocks.items():
        lines.append("\\midrule")
        lines.append(render_block(name, block, budgets, series_labels))

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        f"\\label{{{label}}}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def build_dataset_table(
    model_flag,
    dataset,
    tags,
    budgets=DEPLOY_BUDGETS,
    seeds=SEEDS,
    include_centralized=True,
    exp_base=EXP_BASE,
    flip_root=None,
    flip_frozen_tags=(),
    flip_frozen_exp_base=FLIP_FROZEN_EXP_BASE,
):
    """One table for (model_flag, dataset): one block per tag with data, plus an optional CENTRALIZED column and a FLIP block.
    Each tag in `flip_frozen_tags` also gets, right after its JOLT block, a block for FLIP run with that tag's trigger frozen."""
    columns = TABLE_COLUMNS_WITH_CENTRALIZED if include_centralized else TABLE_AGG_ORDER

    def tag_block(tag, base):
        block = {}
        any_cell = False
        for series in columns:
            if series == CENTRALIZED_SERIES:
                df = compute_cta_pta_mean_var(
                    model_flag,
                    dataset,
                    tag,
                    "single_user",
                    budgets,
                    seeds,
                    exp_base=base,
                )
            else:
                df = compute_cta_pta_mean_var(
                    model_flag,
                    dataset,
                    tag,
                    "federated",
                    budgets,
                    seeds,
                    agg=series,
                    exp_base=base,
                )
            for _, row in df.iterrows():
                if not np.isnan(row["cta_mean"]):
                    any_cell = True
                block[(row["budget"], series)] = (
                    row["cta_mean"],
                    row["cta_var"],
                    row["pta_mean"],
                    row["pta_var"],
                )
        return block if any_cell else None

    blocks = {}
    for tag in tags:
        block = tag_block(tag, exp_base)
        if block is not None:
            blocks[TAG_DISPLAY_NAMES.get(tag, tag)] = block
        else:
            print(f"[INFO] tag={tag!r} has no data yet for {model_flag}/{dataset} -- skipped in table.")
        if tag in flip_frozen_tags:
            block = tag_block(tag, flip_frozen_exp_base)
            if block is not None:
                blocks[FLIP_FROZEN_TAG_DISPLAY_NAMES.get(tag, f"{tag} -- FLIP")] = block
            else:
                print(f"[INFO] FLIP frozen-trigger tag={tag!r} has no data yet for {model_flag}/{dataset} -- skipped in table.")

    if flip_root:
        flip_block = {}
        any_flip_cell = False
        for series in columns:
            if series == CENTRALIZED_SERIES:
                df = compute_cta_pta_mean_var_flip_centralized(flip_root, model_flag, dataset, budgets, seeds)
            else:
                df = compute_cta_pta_mean_var_flip(flip_root, model_flag, dataset, series, budgets, seeds)
            for _, row in df.iterrows():
                if not np.isnan(row["cta_mean"]):
                    any_flip_cell = True
                flip_block[(row["budget"], series)] = (
                    row["cta_mean"],
                    row["cta_var"],
                    row["pta_mean"],
                    row["pta_var"],
                )
        if any_flip_cell:
            blocks[FLIP_BLOCK_NAME] = flip_block
        else:
            print(f"[INFO] FLIP baseline has no data yet for {model_flag}/{dataset} -- skipped in table.")

    if not blocks:
        return None

    dataset_name = DATASET_DISPLAY.get(dataset, dataset)
    model_name = MODEL_DISPLAY.get(model_flag, model_flag)
    centralized_clause = " and the centralized (undefended, single-victim) deployment" if include_centralized else ""
    caption = (
        f"Impact of poisoning budget on ASR and CTA across robust aggregation rules{centralized_clause} on "
        f"{dataset_name} with a {model_name} model. Results are reported as CTA/ASR (in \\%) "
        f"with standard deviation. Bold and underlined values indicate the highest and "
        f"second-highest ASR."
    )
    label = f"tab:{dataset}_{model_flag}_budget_vs_asr"
    return build_table(blocks, budgets, columns, caption, label)


# =========================
# TRIGGER VISUALS
# =========================
def trigger_path_in(module_dir, model_flag, dataset):
    """Path to the .pt trigger run_module.py writes, matching gen_configs_paper_main_campaign.generate_cell()."""
    return (
        module_dir
        / "trigger"
        / f"opt_trig_direct_joint_{GEN_INIT}_{model_flag}_{dataset}_{GEN_NUM_POISONED}vs{GEN_NUM_HONESTS}.pt"
    )


def save_trigger_visual_in(module_dir, model_flag, dataset, label, sample_seed=0):
    """Renders clean-vs-poisoned side by side for one generated trigger and saves the PNG next to the .pt file."""
    trig_path = trigger_path_in(module_dir, model_flag, dataset)
    if not trig_path.exists():
        return None

    dataset_obj = load_dataset(dataset, train=True)
    indices = [i for i, (_, y) in enumerate(dataset_obj) if y == SOURCE_LABEL]
    idx = np.random.RandomState(sample_seed).choice(indices)
    img, _ = dataset_obj[idx]

    clean_img = img
    if isinstance(clean_img, torch.Tensor):
        clean_img = transforms.ToPILImage()(clean_img)

    poisoner = pick_poisoner("optimized", dataset, TARGET_LABEL, delta=str(trig_path))
    poisoned_img, _ = poisoner.poison((img, SOURCE_LABEL))
    if isinstance(poisoned_img, torch.Tensor):
        poisoned_img = transforms.ToPILImage()(poisoned_img)

    fig, axes = plt.subplots(1, 2, figsize=(6, 3.2))
    axes[0].imshow(clean_img)
    axes[0].set_title("Clean image", fontsize=11)
    axes[0].axis("off")
    axes[1].imshow(poisoned_img)
    axes[1].set_title(f"Poisoned ({label})", fontsize=11)
    axes[1].axis("off")
    plt.tight_layout()

    out_path = trig_path.with_suffix(".png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved trigger visual: {out_path}")
    return out_path


def save_all_trigger_visuals(model_flags, datasets, tags, seeds, exp_base=EXP_BASE):
    """One trigger visual per (model_flag, dataset, tag, seed)."""
    saved, missing = [], []
    for model_flag in model_flags:
        for dataset in datasets:
            for tag in tags:
                for seed in seeds:
                    module_dir = exp_base / cell_name(model_flag, dataset, tag, seed) / "gen_labels_trigger_joint"
                    out_path = save_trigger_visual_in(
                        module_dir,
                        model_flag,
                        dataset,
                        label=f"{tag}, seed{seed}",
                    )
                    if out_path is None:
                        missing.append((model_flag, dataset, tag, seed))
                    else:
                        saved.append(out_path)

    if missing:
        print(f"[INFO] {len(missing)} trigger(s) not found yet (skipped):")
        for model_flag, dataset, tag, seed in missing:
            module_dir = exp_base / cell_name(model_flag, dataset, tag, seed) / "gen_labels_trigger_joint"
            print(f"  {trigger_path_in(module_dir, model_flag, dataset)}")

    return saved


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="r32p")
    parser.add_argument(
        "--tags",
        nargs="+",
        default=["baseline", "eps_16_255"],
        help="JOLT tags to compare, in order. Each is overlaid on every per-aggregator "
        "trade-off plot (see plot_cta_vs_pta_per_branch) AND gets its own block in the "
        "LaTeX table, alongside the FLIP baseline (default: baseline eps_16_255; e.g. pass "
        "--tags baseline eps_16_255 eps_0p50_lpips_0p1 to add the stealth-regularized tag "
        "too, once it has finished training).",
    )
    parser.add_argument(
        "--table-tags",
        nargs="+",
        default=None,
        help="tags to include as table blocks, in order (default: same as --tags)",
    )
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--out-dir", default="./plots_paper_main_campaign_user")
    parser.add_argument("--csv-dir", default="./results_csv_paper_main_campaign_user")
    parser.add_argument("--table-dir", default="./tables_paper_main_campaign_user")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument(
        "--flip-root",
        default="out_neurips",
        help="root of the FLIP paper baseline's results tree (out_neurips/{model}/"
        "{tag_suffix}_FLIP/{dataset}/backdoor/{agg}/1xs/{seed}/{budget}/{caccs,paccs}.npy), "
        "overlaid per-aggregator as a comparison series on the per-branch trade-off plots; "
        "pass an empty string to disable",
    )
    parser.add_argument(
        "--jolt-root",
        default=None,
        help="override where OUR (JOLT) own results are read from, in place of the imported "
        "EXP_BASE ('experiments/federated_experiments/"
        "threat_model_direct_trigger_joint_paper_main_campaign') -- e.g. a cluster mount "
        "mirroring that same {model_flag}/{dataset}/{tag}/seed{seed}/... subtree under a "
        "different root, to avoid local<->cluster file transfers "
        "(e.g. /mnt/cluster/FLIP/out_iclr). "
        "Default: use EXP_BASE unchanged.",
    )
    parser.add_argument(
        "--flip-frozen-tags",
        nargs="*",
        default=[],
        help="table tags that also get a block for FLIP re-run with that tag's JOLT trigger frozen "
        "(gen_configs_flip_frozen_trigger.py), placed right after the tag's JOLT block",
    )
    parser.add_argument(
        "--flip-frozen-root",
        default=None,
        help=f"override where the FLIP frozen-trigger results are read from (default: {FLIP_FROZEN_EXP_BASE})",
    )
    parser.add_argument("--skip-trigger-visuals", action="store_true")
    parser.add_argument(
        "--no-centralized-column",
        action="store_true",
        help="drop the CENTRALIZED (single_user) column from the table, matching the "
        "federated-aggregators-only style of the original show_results.py table",
    )
    args = parser.parse_args()

    table_tags = args.table_tags or args.tags
    exp_base = Path(args.jolt_root) if args.jolt_root else EXP_BASE
    flip_frozen_exp_base = Path(args.flip_frozen_root) if args.flip_frozen_root else FLIP_FROZEN_EXP_BASE

    tags_suffix = "-".join(args.tags)
    out_root = Path(args.out_dir) / f"{args.model}_{tags_suffix}"
    csv_root = Path(args.csv_dir) / f"{args.model}_{tags_suffix}"
    table_root = Path(args.table_dir)
    table_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_trigger_visuals:
        print(f"\n=== [paper_main_campaign] Trigger visuals ({args.model}, tags={table_tags}) ===")
        save_all_trigger_visuals([args.model], args.datasets, table_tags, args.seeds, exp_base=exp_base)

    for dataset in args.datasets:
        print(f"\n=== [paper_main_campaign/user] {args.model} / {dataset} / tags={args.tags} ===")

        tag_data = {}
        for tag in args.tags:
            tag_data[tag] = collect_dataset(
                args.model,
                dataset,
                tag,
                seeds=args.seeds,
                flip_root=None,
                exp_base=exp_base,
            )
        flip_data = collect_flip_data(args.flip_root, args.model, dataset, seeds=args.seeds) if args.flip_root else {}

        csv_dir = csv_root / dataset
        csv_dir.mkdir(parents=True, exist_ok=True)
        for tag, all_data in tag_data.items():
            for branch_key, df in all_data.items():
                n_cta = df["cta_mean"].notna().sum()
                n_asr = df["pta_mean"].notna().sum()
                print(
                    f"   -> tag={tag:16s} {branch_key:24s} budgets with data: "
                    f"cta={n_cta}/{len(DEPLOY_BUDGETS)} asr={n_asr}/{len(DEPLOY_BUDGETS)}"
                )
                df.to_csv(csv_dir / f"{tag}__{branch_key}.csv", index=False)
        for branch_key, df in flip_data.items():
            n_cta = df["cta_mean"].notna().sum()
            n_asr = df["pta_mean"].notna().sum()
            print(
                f"   -> tag={'flip':16s} {branch_key:24s} budgets with data: "
                f"cta={n_cta}/{len(DEPLOY_BUDGETS)} asr={n_asr}/{len(DEPLOY_BUDGETS)}"
            )
            df.to_csv(csv_dir / f"flip__{branch_key}.csv", index=False)

        tag_suffix = FEDERATED_TAG.split("_", 1)[-1]  # "federated_3vs7" -> "3vs7"
        per_branch_dir = str(out_root / f"{dataset}_{tag_suffix}")
        saved_per_branch = plot_cta_vs_pta_per_branch(
            tag_data,
            flip_data,
            dataset,
            args.model,
            args.tags,
            save_dir=per_branch_dir,
        )
        ok_tradeoff = bool(saved_per_branch)

        # CTA/ASR-vs-budget overlay plots only support a single tag; use the first requested one.
        primary_tag = args.tags[0]
        primary_data = {**tag_data[primary_tag], **flip_data}
        save_dir = str(out_root / dataset)
        ok_cta = plot_metric_vs_budget(
            primary_data,
            dataset,
            args.model,
            primary_tag,
            metric="cta",
            ylabel="CTA (%)",
            save_dir=save_dir,
        )
        ok_asr = plot_metric_vs_budget(
            primary_data,
            dataset,
            args.model,
            primary_tag,
            metric="pta",
            ylabel="ASR (%)",
            save_dir=save_dir,
        )

        if not (ok_tradeoff or ok_cta or ok_asr):
            print(
                f"[WARNING] no data found under {exp_base / cell_name(args.model, dataset, primary_tag, '*')} -- "
                f"nothing plotted for {dataset}."
            )

        table = build_dataset_table(
            args.model,
            dataset,
            table_tags,
            seeds=args.seeds,
            include_centralized=not args.no_centralized_column,
            exp_base=exp_base,
            flip_root=args.flip_root or None,
            flip_frozen_tags=args.flip_frozen_tags,
            flip_frozen_exp_base=flip_frozen_exp_base,
        )
        if table is None:
            print(f"[WARNING] no data found for table blocks {table_tags} -- nothing written for {dataset}.")
        else:
            table_path = table_root / f"{args.model}_{dataset}.tex"
            table_path.write_text(table)
            print(f"[INFO] Saved table: {table_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
