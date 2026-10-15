import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torchvision import transforms

# Repo root needs to be on sys.path for `modules.*` to import.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modules.base_utils.datasets import load_dataset, pick_poisoner
from modules.generate_labels_trigger_joint.gen_configs import (
    EXP_BASE,
    MODEL_FLAGS,
    DATASETS,
    AGG_METHODS,
    SEEDS,
    BUDGETS,
    NUM_POISONED,
    NUM_HONESTS,
    SOURCE_LABEL,
    TARGET_LABEL,
    INIT,
    cell_name,
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


# =========================
# Colors -- keyed by the series label, passed to plot_cta_vs_pta's `colors` param.
# =========================
AGG_COLORS = {
    "mean": "tab:blue",
    "median": "tab:orange",
    "krum": "tab:green",
    "trmean": "tab:red",
    "multikrum": "tab:purple",
}

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
def _cta_pta_mean_var(run_dir_fn, seeds, cta_file, pta_file):
    """Averages caccs.npy/paccs.npy's final value across `seeds` for the cell `run_dir_fn(seed)` resolves to."""
    cta_vals, pta_vals = [], []
    for seed in seeds:
        run_dir = run_dir_fn(seed)
        cta_path = run_dir / cta_file
        pta_path = run_dir / pta_file
        if not (cta_path.exists() and pta_path.exists()):
            continue
        cta_vals.append(get_final_value(cta_path))
        pta_vals.append(get_final_value(pta_path))

    return (
        np.mean(cta_vals) if cta_vals else np.nan,
        np.var(cta_vals) if cta_vals else np.nan,
        np.mean(pta_vals) if pta_vals else np.nan,
        np.var(pta_vals) if pta_vals else np.nan,
    )


def compute_cta_pta_mean_var(
    model_flag,
    dataset,
    agg_method,
    budgets,
    seeds,
    cta_file="caccs.npy",
    pta_file="paccs.npy",
):
    """Reads federated_train_user's outputs for each (budget, seed) cell of the main campaign."""
    records = []

    for budget in budgets:
        run_dir_fn = lambda seed, budget=budget: (
            EXP_BASE / cell_name(model_flag, dataset, agg_method, seed) / f"train_user_{budget}"
        )
        cta_mean, cta_var, pta_mean, pta_var = _cta_pta_mean_var(
            run_dir_fn, seeds, cta_file, pta_file
        )
        records.append(
            {
                "dataset": dataset,
                "aggregator": agg_method,
                "budget": budget,
                "model": model_flag,
                "cta_mean": cta_mean,
                "cta_var": cta_var,
                "pta_mean": pta_mean,
                "pta_var": pta_var,
            }
        )

    return pd.DataFrame.from_records(records)


# =========================
# TRIGGER VISUALS
# =========================
def trigger_path_in(module_dir, model_flag, dataset, num_poisoned, num_honests, init="stripe"):
    """Path to the .pt trigger run_module.py writes inside a campaign's gen_labels_trigger_joint dir."""
    return (
        module_dir
        / "trigger"
        / f"opt_trig_direct_joint_{init}_{model_flag}_{dataset}_{num_poisoned}vs{num_honests}.pt"
    )


def check_epsilon_compliance(
    module_dir, model_flag, dataset, num_poisoned, num_honests, epsilon, init="stripe",
    tol=1e-6,
):
    """Checks the saved trigger's measured Linf norm against its configured epsilon; None if not saved yet."""
    trig_path = trigger_path_in(
        module_dir, model_flag, dataset, num_poisoned, num_honests, init=init,
    )
    if not trig_path.exists():
        return None

    delta = torch.load(trig_path, map_location="cpu")
    measured_linf = delta.abs().max().item()
    within_bound = measured_linf <= epsilon + tol

    status = "OK" if within_bound else "VIOLATION"
    print(
        f"[EPSILON CHECK] {trig_path.parent.parent.name}: measured max|delta|="
        f"{measured_linf:.6f} vs configured epsilon={epsilon:.6f} -- {status}"
    )
    return measured_linf, epsilon, within_bound


def save_trigger_visual_in(
    module_dir, model_flag, dataset, num_poisoned, num_honests, label, sample_seed=0,
    init="stripe",
):
    """Renders clean-vs-poisoned side by side for one trigger and saves the PNG next to the .pt file."""
    trig_path = trigger_path_in(
        module_dir, model_flag, dataset, num_poisoned, num_honests, init=init,
    )
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


def save_all_trigger_visuals(model_flags, datasets, agg_methods, seeds):
    """Main campaign wrapper: builds each cell's module_dir via gen_configs.py's cell_name/EXP_BASE."""
    saved, missing = [], []
    for model_flag in model_flags:
        for dataset in datasets:
            for agg_method in agg_methods:
                for seed in seeds:
                    module_dir = (
                        EXP_BASE
                        / cell_name(model_flag, dataset, agg_method, seed)
                        / "gen_labels_trigger_joint"
                    )
                    out_path = save_trigger_visual_in(
                        module_dir, model_flag, dataset, NUM_POISONED, NUM_HONESTS,
                        label=f"{agg_method}, seed{seed}", init=INIT,
                    )
                    if out_path is None:
                        missing.append((model_flag, dataset, agg_method, seed))
                    else:
                        saved.append(out_path)
    if missing:
        print(f"[INFO] {len(missing)} trigger(s) not found yet (skipped):")
        for model_flag, dataset, agg_method, seed in missing:
            module_dir = (
                EXP_BASE / cell_name(model_flag, dataset, agg_method, seed)
                / "gen_labels_trigger_joint"
            )
            print(
                f"  {trigger_path_in(module_dir, model_flag, dataset, NUM_POISONED, NUM_HONESTS, init=INIT)}"
            )
    return saved


def compute_best_second(block, budgets, series_labels):
    best = {}
    second = {}

    for series in series_labels:
        values = []

        for b in budgets:
            val = block.get((b, series), (np.nan,) * 4)
            values.append(val[2])  # ASR

        values = np.array(values)

        valid_idx = np.where(~np.isnan(values))[0]

        if len(valid_idx) == 0:
            best[series] = None
            second[series] = None
            continue

        sorted_idx = valid_idx[np.argsort(values[valid_idx])]

        best_idx = sorted_idx[-1]
        second_idx = sorted_idx[-2] if len(sorted_idx) > 1 else sorted_idx[-1]

        best[series] = budgets[best_idx]
        second[series] = budgets[second_idx]

    return best, second


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
    key_indices = [idx_best]

    x_range = max(np.nanmax(x) - np.nanmin(x), 1e-6)
    y_range = max(np.nanmax(y) - np.nanmin(y), 1e-6)

    dx = 0.03 * x_range
    dy = 0.03 * y_range

    directions = [
        (+dx, +dy),
        (+dx, -dy),
        (-dx, +dy),
        (-dx, -dy),
        (0, +1.5 * dy),
        (+1.5 * dx, 0),
    ]

    for k, i in enumerate(key_indices):
        if np.isnan(x[i]) or np.isnan(y[i]):
            continue

        dx_i, dy_i = directions[k % len(directions)]

        plt.text(
            x[i] + dx_i,
            y[i] + dy_i,
            str(int(budgets[i])),
            fontsize=11,
            fontweight="bold",
            color=color,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, alpha=0.8),
            zorder=10,
        )


def plot_cta_vs_pta(all_data, dataset, save_dir=None, colors=None, filename=None):
    """One figure, one line per entry of `all_data` (series label -> DataFrame with budget/cta/pta columns)."""
    colors = colors or {}

    plt.figure(figsize=(7.5, 6))

    for series_label, df in all_data.items():
        df = df.dropna()
        if df.empty:
            continue

        df = df.sort_values("budget")

        x = df["pta_mean"].values * 100
        y = df["cta_mean"].values * 100

        xerr = np.sqrt(df["pta_var"].values) * 100
        yerr = np.sqrt(df["cta_var"].values) * 100

        color = colors.get(series_label, None)

        plt.plot(
            x, y, linestyle="-", linewidth=2.2, color=color, alpha=0.85,
            label=series_label, zorder=3,
        )

        plt.errorbar(
            x, y, xerr=xerr, yerr=yerr, fmt="none", ecolor=color,
            elinewidth=1.2, capsize=3, alpha=0.35, zorder=1,
        )

        plt.scatter(x, y, s=45, color=color, edgecolors="none", zorder=4)

        score = x  # ASR

        max_score = np.nanmax(score)
        candidates = np.where(np.isclose(score, max_score, atol=1e-12))[0]
        budgets = df["budget"].values
        idx_best = candidates[np.argmin(budgets[candidates])]

        plt.scatter(
            x[idx_best], y[idx_best], s=95, color=color,
            edgecolor="black", linewidth=1.2, zorder=6,
        )

        annotate_key_points(df, x, y, score, color)

    plt.xlabel("ASR (%)")
    plt.ylabel("CTA (%)")

    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True, linestyle="--", alpha=0.25)

    plt.legend(frameon=True, fontsize=11, loc="lower left")

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename or f"{dataset}.png")
        plt.savefig(path, dpi=300)
        print(f"[INFO] Saved plot: {path}")

    plt.close()


# =========================
# TABLE
# =========================
def format_cell(cta, cta_var, asr, asr_var):

    if np.isnan(cta):
        return "XXX"

    cta_std = np.sqrt(cta_var) * 100
    asr_std = np.sqrt(asr_var) * 100

    return f"{cta * 100:.1f}$\\pm${cta_std:.1f}/{asr * 100:.1f}$\\pm${asr_std:.1f}"


def build_table(block, budgets, series_labels, name):
    """`series_labels` is a list of column labels matching the keys used in `block`."""

    best, second = compute_best_second(block, budgets, series_labels)

    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\scriptsize")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append("")
    lines.append("\\begin{tabular}{c " + " ".join(["c"] * len(series_labels)) + "}")
    lines.append("\\toprule")
    lines.append("Budget & " + " & ".join([str(s).upper() for s in series_labels]) + " \\\\")
    lines.append("\\midrule")
    lines.append(f"\\multicolumn{{{1 + len(series_labels)}}}{{c}}{{\\textbf{{{name}}}}} \\\\")
    lines.append("\\midrule")

    for b in budgets:
        row = [str(b)]

        for series in series_labels:
            val = block.get((b, series), (np.nan,) * 4)
            cell = format_cell(*val)

            if best[series] == b:
                cell = f"\\textbf{{{cell}}}"
            elif second[series] == b:
                cell = f"\\underline{{{cell}}}"

            row.append(cell)

        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    # Sweep axes are read from gen_configs.py, not redeclared here.
    CSV_DIR = "./results_csv_trigger_joint"
    TABLE_DIR = "./tables_trigger_joint"
    PLOT_DIR = "./plots_trigger_joint"

    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # -------------------------------------------------------------------
    # Main campaign (3-poisoned/7-honest, agg_method-swept)
    # -------------------------------------------------------------------
    print("\n=== [main campaign] Trigger visuals ===")
    save_all_trigger_visuals(MODEL_FLAGS, DATASETS, AGG_METHODS, SEEDS)

    for model_flag in MODEL_FLAGS:
        for dataset in DATASETS:
            print(f"\n=== [main campaign] {model_flag} / {dataset} ===")

            all_data = {}
            block = {}

            for agg_method in AGG_METHODS:
                print(f"   -> {agg_method}")

                df = compute_cta_pta_mean_var(
                    model_flag=model_flag,
                    dataset=dataset,
                    agg_method=agg_method,
                    budgets=BUDGETS,
                    seeds=SEEDS,
                )

                df.to_csv(
                    f"{CSV_DIR}/{model_flag}_{dataset}_{agg_method}.csv",
                    index=False,
                )

                all_data[agg_method] = df

                for _, row in df.iterrows():
                    block[(row["budget"], agg_method)] = (
                        row["cta_mean"],
                        row["cta_var"],
                        row["pta_mean"],
                        row["pta_var"],
                    )

            latex_table = build_table(
                block, BUDGETS, AGG_METHODS, name=f"{model_flag}/{dataset}"
            )
            with open(f"{TABLE_DIR}/{model_flag}_{dataset}.tex", "w") as f:
                f.write(latex_table)
            print(f"[INFO] Saved LaTeX table for {model_flag}/{dataset}")

            plot_cta_vs_pta(
                all_data, dataset=dataset, save_dir=f"{PLOT_DIR}/{model_flag}/",
                colors=AGG_COLORS,
            )
