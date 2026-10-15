#!/usr/bin/env python3
"""Side-by-side clean-vs-poisoned trigger grid (qualitative figure) for paper_main_campaign.

Usage:
  python scripts/show_results/plot_trigger_grid.py
  python scripts/show_results/plot_trigger_grid.py --seed 0 --models r32p convnext_micro \
      --datasets cifar svhn --tags baseline eps_16_255 --jolt-root out_iclr
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_paper_main_campaign_user import (  # noqa: E402
    EXP_BASE,
    SOURCE_LABEL,
    TARGET_LABEL,
    DATASET_DISPLAY,
    MODEL_DISPLAY,
    cell_name,
    trigger_path_in,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from modules.base_utils.datasets import load_dataset, pick_poisoner  # noqa: E402

# "baseline" tag's epsilon value, spelled out for readability.
TAG_COLUMN_LABELS = {
    "baseline": "Optimized Trigger ($\\epsilon$=1)",
    "eps_16_255": "Optimized Trigger ($\\epsilon$=16/255)",
    "eps_0p50_lpips_0p1": "Optimized Trigger ($\\epsilon$=0.5, LPIPS=0.1)",
}

plt.rcParams.update({"font.size": 13})

_dataset_sample_cache = {}


def _to_pil(img):
    if isinstance(img, torch.Tensor):
        return transforms.ToPILImage()(img)
    return img


def _get_clean_sample(dataset, sample_seed):
    """One fixed source-labeled training image per dataset, cached and reused across every model/tag."""
    key = (dataset, sample_seed)
    if key not in _dataset_sample_cache:
        dataset_obj = load_dataset(dataset, train=True)
        indices = [i for i, (_, y) in enumerate(dataset_obj) if y == SOURCE_LABEL]
        idx = np.random.RandomState(sample_seed).choice(indices)
        img, _ = dataset_obj[idx]
        _dataset_sample_cache[key] = img
    return _dataset_sample_cache[key]


def load_clean_and_poisoned(model_flag, dataset, tags, seed, exp_base, sample_seed=0):
    """Returns (clean_img, {tag: poisoned_img_or_None}); None for a tag whose trigger isn't generated yet."""
    img = _get_clean_sample(dataset, sample_seed)
    clean_img = _to_pil(img)

    poisoned = {}
    for tag in tags:
        module_dir = exp_base / cell_name(model_flag, dataset, tag, seed) / "gen_labels_trigger_joint"
        trig_path = trigger_path_in(module_dir, model_flag, dataset)
        if not trig_path.exists():
            poisoned[tag] = None
            continue
        poisoner = pick_poisoner("optimized", dataset, TARGET_LABEL, delta=str(trig_path))
        poisoned_img, _ = poisoner.poison((img, SOURCE_LABEL))
        poisoned[tag] = _to_pil(poisoned_img)

    return clean_img, poisoned


def plot_trigger_grid(model_flag, datasets, tags, seed, exp_base, save_path, sample_seed=0):
    """One figure: rows=datasets, columns=[Clean, one per tag]."""
    n_rows = len(datasets)
    n_cols = 1 + len(tags)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.6 * n_cols, 3.5 * n_rows), squeeze=False,
        gridspec_kw={"wspace": 0.12, "hspace": 0.08},
    )

    any_missing = []
    for row, dataset in enumerate(datasets):
        clean_img, poisoned = load_clean_and_poisoned(
            model_flag, dataset, tags, seed, exp_base, sample_seed=sample_seed,
        )

        ax = axes[row][0]
        ax.imshow(clean_img)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if row == 0:
            ax.set_title("Clean", fontsize=14, fontweight="bold")
        ax.set_ylabel(
            DATASET_DISPLAY.get(dataset, dataset), fontsize=14, fontweight="bold", labelpad=12,
        )

        for col, tag in enumerate(tags, start=1):
            ax = axes[row][col]
            img = poisoned[tag]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if img is None:
                any_missing.append((dataset, tag))
                ax.text(
                    0.5, 0.5, "not generated yet", ha="center", va="center", fontsize=10,
                    color="0.4", transform=ax.transAxes,
                )
                ax.set_facecolor("#f0f0f0")
            else:
                ax.imshow(img)
            if row == 0:
                ax.set_title(TAG_COLUMN_LABELS.get(tag, tag), fontsize=12, fontweight="bold")

    fig.suptitle(f"{MODEL_DISPLAY.get(model_flag, model_flag)} -- seed {seed}", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if any_missing:
        print(f"[INFO] {model_flag}: {len(any_missing)} trigger(s) not generated yet (shown as "
              f"placeholders): {any_missing}")
    print(f"[INFO] Saved trigger grid: {save_path}")
    return save_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["r32p", "convnext_micro"])
    parser.add_argument("--datasets", nargs="+", default=["cifar", "svhn"])
    parser.add_argument("--tags", nargs="+", default=["baseline", "eps_16_255"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sample-seed", type=int, default=0,
        help="RNG seed picking which source image is shown per dataset (independent of --seed).",
    )
    parser.add_argument("--out-dir", default="./plots_trigger_grid")
    parser.add_argument(
        "--jolt-root", default=None,
        help="override EXP_BASE, e.g. a mount mirroring the same subtree under a different root.",
    )
    args = parser.parse_args()

    exp_base = Path(args.jolt_root) if args.jolt_root else EXP_BASE
    out_dir = Path(args.out_dir)

    for model_flag in args.models:
        save_path = out_dir / f"trigger_grid_{model_flag}_seed{args.seed}.png"
        plot_trigger_grid(
            model_flag, args.datasets, args.tags, args.seed, exp_base, save_path,
            sample_seed=args.sample_seed,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
