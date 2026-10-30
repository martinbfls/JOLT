#!/usr/bin/env python3
"""Compares an additive-trigger run against a warp-trigger (WaNet-style) run produced by
compare_additive_warp_slurm.sh: final ASR/CTA from each train_user's metrics.json, plus a
figure with a clean sample, both poisoned versions, and (for warp) the raw displacement field.

Usage (from repo root, with the JOLT conda env active):
  python scripts/show_results/compare_additive_warp.py \\
      --additive-dir experiments/federated_experiments/threat_model_direct_trigger_joint_additive \\
      --warp-dir experiments/federated_experiments/threat_model_direct_trigger_joint_warp \\
      --model-flag r32p --dataset cifar --budget 1500 --seed 0 \\
      --out scripts/show_results/compare_additive_warp.png
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from modules.base_utils.datasets import load_dataset, pick_poisoner  # noqa: E402

SOURCE_LABEL = 9
TARGET_LABEL = 4


def cell_dir(exp_base, model_flag, dataset, seed):
    return Path(exp_base) / model_flag / dataset / "1vs0" / "mean" / f"seed{seed}"


def trigger_path(exp_base, model_flag, dataset, seed, init="stripe"):
    module_dir = cell_dir(exp_base, model_flag, dataset, seed) / "gen_labels_trigger_joint"
    return module_dir / "trigger" / f"opt_trig_direct_joint_{init}_{model_flag}_{dataset}_1vs0.pt"


def load_metrics(train_user_dir):
    """Reads logs/federated_train_user_metrics.json (ExperimentTracker's output); returns
    (final_cta, final_asr) or (None, None) if the file/keys are missing (run not finished)."""
    path = Path(train_user_dir) / "logs" / "federated_train_user_metrics.json"
    if not path.exists():
        return None, None
    with open(path) as f:
        data = json.load(f)
    cta = data.get("cta", [])
    asr = data.get("asr", [])
    final_cta = cta[-1]["value"] if cta else None
    final_asr = asr[-1]["value"] if asr else None
    return final_cta, final_asr


def to_pil(img):
    return transforms.ToPILImage()(img) if isinstance(img, torch.Tensor) else img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--additive-dir", required=True, help="EXP_BASE of the additive run")
    ap.add_argument("--warp-dir", required=True, help="EXP_BASE of the warp run")
    ap.add_argument("--model-flag", default="r32p")
    ap.add_argument("--dataset", default="cifar")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--out", default="scripts/show_results/compare_additive_warp.png")
    args = ap.parse_args()

    for label, exp_base, poisoner_flag in [
        ("additive", args.additive_dir, "optimized"),
        ("warp", args.warp_dir, "warp"),
    ]:
        train_user_dir = cell_dir(exp_base, args.model_flag, args.dataset, args.seed) / f"train_user_{args.budget}"
        cta, asr = load_metrics(train_user_dir)
        if cta is None:
            print(f"[{label}] metrics not found yet at {train_user_dir}/logs/federated_train_user_metrics.json")
        else:
            print(f"[{label}] final CTA={cta:.2f}%  ASR={asr:.2f}%  (budget={args.budget})")

    dataset_obj = load_dataset(args.dataset, train=True)
    indices = [i for i, (_, y) in enumerate(dataset_obj) if y == SOURCE_LABEL]
    idx = np.random.RandomState(0).choice(indices)
    clean_img, _ = dataset_obj[idx]

    panels = [("Clean", to_pil(clean_img))]
    warp_field = None
    for label, exp_base, poisoner_flag in [
        ("Additive trigger", args.additive_dir, "optimized"),
        ("Warp trigger (WaNet-style)", args.warp_dir, "warp"),
    ]:
        trig_path = trigger_path(exp_base, args.model_flag, args.dataset, args.seed)
        if not trig_path.exists():
            print(f"[{label}] trigger not found at {trig_path} -- skipping panel")
            continue
        poisoner = pick_poisoner(poisoner_flag, args.dataset, TARGET_LABEL, delta=str(trig_path))
        poisoned_img, _ = poisoner.poison((clean_img, SOURCE_LABEL))
        panels.append((label, to_pil(poisoned_img)))
        if poisoner_flag == "warp":
            warp_field = torch.load(trig_path)["control_grid"]  # (1, 2, k, k) pixel displacements

    n = len(panels) + (1 if warp_field is not None else 0)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    if warp_field is not None:
        ax = axes[-1]
        dx, dy = warp_field[0, 0].numpy(), warp_field[0, 1].numpy()
        ax.quiver(dx, -dy, angles="xy")
        ax.set_title("Warp control grid (pixel displacement)")
        ax.set_aspect("equal")
        ax.invert_yaxis()

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"[OK] figure saved to {args.out}")


if __name__ == "__main__":
    main()
