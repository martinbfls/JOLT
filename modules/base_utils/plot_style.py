"""Shared paper-style plotting helpers (same look as scripts/show_results/plot_paper_main_campaign_user.py)
for the training/generation curves written by ExperimentTracker and scripts/show_results/plot_training_curves.py."""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RC_PARAMS = {
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "lines.linewidth": 2.5,
}
FIGSIZE = (7.5, 6)
SMALL_FIGSIZE = (6.5, 4.5)
DPI = 300
FORMATS = ("png", "pdf")

# metric key -> (axis label, scale applied to the stored value). Fractions are stored in [0, 1]
# and shown in %, like the paper tables. Unknown keys fall back to the raw key and scale 1.
METRICS = {
    # train_user / train_expert (per epoch)
    "cta": ("CTA (%)", 100),
    "asr": ("ASR (%)", 100),
    "acc": ("Train accuracy (%)", 100),
    "loss": ("Train loss", 1),
    "lr": ("Learning rate", 1),
    "test_loss": ("Test loss", 1),
    "poison_test_loss": ("Poisoned test loss", 1),
    "grad_cosine_sim_poisoned_honest": ("cos(poisoned, honest gradient)", 1),
    "grad_l2_dist_poisoned_honest": (r"$\|g_{poisoned} - g_{honest}\|_2$", 1),
    # generate_labels / generate_labels_trigger_joint (per optimization step)
    "g_loss": ("Matching loss", 1),
    "grand_loss": ("Total loss", 1),
    "matching_term": ("Trajectory-matching loss", 1),
    "L_bd_mean": (r"$\mathcal{L}_{bd}$", 1),
    "L_joint_mean": (r"$\mathcal{L}_{joint}$", 1),
    "L_lpips_mean": (r"$\mathcal{L}_{LPIPS}$", 1),
    "expert_asr": ("Expert ASR (%)", 100),
    "expert_asr_frozen": ("Expert ASR, initial trigger (%)", 100),
    "retrain_expert_cta": ("Retrained expert CTA (%)", 100),
    "retrain_expert_asr": ("Retrained expert ASR (%)", 100),
    "delta_l2": (r"$\|\delta\|_2$", 1),
    "delta_linf": (r"$\|\delta\|_\infty$", 1),
    "cos_target": (r"cos($\delta$, target-class mean)", 1),
    "cos_source": (r"cos($\delta$, source-class mean)", 1),
    "cos_delta_to_init": (r"cos($\delta$, $\delta_{init}$)", 1),
    "delta_drift_l2": (r"$\|\delta - \delta_{init}\|_2$", 1),
    "delta_sign_flip_rate": (r"Sign-flip rate of $\delta$", 1),
    "mtt_delta_grad_norm": (r"$\|\nabla_\delta \mathcal{L}_{MTT}\|_2$", 1),
    "reg_term": ("Label regularization", 1),
}

# Line color per metric when several share an axis (tab: palette, as in the paper plots).
COLORS = {
    "cta": "tab:blue",
    "asr": "tab:red",
    "expert_asr": "tab:red",
    "expert_asr_frozen": "tab:gray",
    "retrain_expert_asr": "tab:purple",
    "retrain_expert_cta": "tab:blue",
}


def apply():
    plt.rcParams.update(RC_PARAMS)


def label(key):
    return METRICS.get(key, (key, 1))[0]


def legend_label(key):
    """Axis label without the unit suffix, for legends whose axis already carries it."""
    return label(key).replace(" (%)", "")


def scale(key):
    return METRICS.get(key, (key, 1))[1]


def style_axes(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.25)


def smooth(values, window):
    """Trailing rolling mean (same length as `values`); window <= 1 returns values unchanged."""
    values = np.asarray(values, dtype=float)
    if window <= 1 or len(values) < 2:
        return values
    csum = np.cumsum(np.insert(values, 0, 0.0))
    out = np.empty_like(values)
    for i in range(len(values)):
        lo = max(0, i + 1 - window)
        out[i] = (csum[i + 1] - csum[lo]) / (i + 1 - lo)
    return out


def auto_window(n_points):
    """Rolling-mean window for dense per-step curves (~1/50 of the run), 1 for sparse ones."""
    return max(1, n_points // 50) if n_points > 200 else 1


def save(fig, path_stem, formats=FORMATS):
    """Writes path_stem.{png,pdf}; returns the written paths."""
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    written = []
    for ext in formats:
        path = path_stem.with_suffix(f".{ext}")
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written
