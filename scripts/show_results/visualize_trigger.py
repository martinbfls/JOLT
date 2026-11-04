#!/usr/bin/env python3
"""Visualizes ONE already-generated trigger: a clean sample, its poisoned version, and the raw
components it is made of -- the control grid's displacement field (trigger_type="warp" and
"hybrid"), the additive perturbation (trigger_type="hybrid", amplified so a small epsilon
stays legible), and, when the trigger carries them, the photometric map (color_trigger) and the
contrast mask (contrast_mask_floor). Meant to run automatically right after the GEN phase
(generate_labels_trigger_joint) in a Slurm pipeline.

Usage (from repo root, with the JOLT conda env active):
  python scripts/show_results/visualize_trigger.py \\
      --exp-base experiments/federated_experiments/threat_model_direct_trigger_joint_warp \\
      --model-flag r32p --dataset cifar --seed 0 --trigger-type warp \\
      --out experiments/federated_experiments/threat_model_direct_trigger_joint_warp/r32p/cifar/1vs0/mean/seed0/gen_labels_trigger_joint/trigger_preview.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from modules.base_utils.datasets import (  # noqa: E402
    POISONER_BY_TRIGGER_TYPE,
    load_dataset,
    pick_poisoner,
)

SOURCE_LABEL = 9
TARGET_LABEL = 4


def cell_dir(exp_base, model_flag, dataset, seed, gen_agg_method="mean"):
    return Path(exp_base) / model_flag / dataset / "1vs0" / gen_agg_method / f"seed{seed}"


def trigger_path(exp_base, model_flag, dataset, seed, gen_agg_method="mean", init="stripe"):
    module_dir = cell_dir(exp_base, model_flag, dataset, seed, gen_agg_method) / "gen_labels_trigger_joint"
    return module_dir / "trigger" / f"opt_trig_direct_joint_{init}_{model_flag}_{dataset}_1vs0.pt"


def to_pil(img):
    return transforms.ToPILImage()(img) if isinstance(img, torch.Tensor) else img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-base", required=True)
    ap.add_argument("--model-flag", default="r32p")
    ap.add_argument("--dataset", default="cifar")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trigger-type", choices=sorted(POISONER_BY_TRIGGER_TYPE), default="warp")
    ap.add_argument("--gen-agg-method", default="mean")
    ap.add_argument("--out", default=None, help="defaults to <module_dir>/trigger_preview.png")
    args = ap.parse_args()

    trig_path = trigger_path(args.exp_base, args.model_flag, args.dataset, args.seed, args.gen_agg_method)
    if not trig_path.exists():
        print(f"[ABORT] trigger not found at {trig_path}", file=sys.stderr)
        raise SystemExit(1)

    out = Path(args.out) if args.out else trig_path.parent / "trigger_preview.png"

    dataset_obj = load_dataset(args.dataset, train=True)
    indices = [i for i, (_, y) in enumerate(dataset_obj) if y == SOURCE_LABEL]
    idx = np.random.RandomState(0).choice(indices)
    clean_img, _ = dataset_obj[idx]

    poisoner = pick_poisoner(
        POISONER_BY_TRIGGER_TYPE[args.trigger_type], args.dataset, TARGET_LABEL, delta=str(trig_path),
    )
    poisoned_img, _ = poisoner.poison((clean_img, SOURCE_LABEL))

    panels = [("Clean", to_pil(clean_img)), (f"Poisoned ({args.trigger_type})", to_pil(poisoned_img))]
    # 'additive' saves a bare tensor, 'warp'/'hybrid' the dict rebuilding each component
    saved = torch.load(trig_path) if args.trigger_type != "additive" else None
    warp_field = saved["control_grid"] if saved is not None else None  # (1, 2, k, k) pixel displacements
    delta_field = saved.get("delta") if saved is not None else None  # (C, H, W), 'hybrid' only
    # Optional components, absent from a trigger optimized without them.
    color_params = saved.get("color_params") if saved is not None else None  # (3, 4) = [A | b]
    mask_floor = (saved.get("contrast_mask_floor") if saved is not None else None) or None

    # The perturbation actually deployed on this image, as a number: the headline epsilon is only
    # an upper bound once the contrast mask reweights it, and the warp/color components do not
    # show up in it at all.
    clean_t = transforms.ToTensor()(clean_img)
    pois_t = transforms.ToTensor()(to_pil(poisoned_img))
    measured = (
        f"measured on this image: ||x' - x||_inf={float((pois_t - clean_t).abs().max()):.4f}, "
        f"||x' - x||_2={float((pois_t - clean_t).norm()):.3f}"
    )

    n = len(panels) + sum(c is not None for c in (delta_field, warp_field, color_params, mask_floor))
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    extra_axes = list(axes[len(panels):])
    if delta_field is not None:
        ax = extra_axes.pop(0)
        # min-max normalized: a stealthy epsilon (e.g. 8/255) is invisible at true scale
        d = delta_field.permute(1, 2, 0).numpy()
        ax.imshow((d - d.min()) / max(d.max() - d.min(), 1e-12))
        ax.set_title(f"Additive delta, normalized (||delta||_inf={float(delta_field.abs().max()):.4f})")
        ax.axis("off")
    if warp_field is not None:
        ax = extra_axes.pop(0)
        dx, dy = warp_field[0, 0].numpy(), warp_field[0, 1].numpy()
        ax.quiver(dx, -dy, angles="xy")
        tv = float((np.abs(np.diff(warp_field[0].numpy(), axis=1)).sum()
                    + np.abs(np.diff(warp_field[0].numpy(), axis=2)).sum()))
        ax.set_title(
            f"Warp control grid {tuple(warp_field.shape[-2:])}\n"
            f"max |disp|={float(warp_field.abs().max()):.2f}px, TV={tv:.2f}px"
        )
        ax.set_aspect("equal")
        ax.invert_yaxis()
    if color_params is not None:
        from modules.optimizing_trigger.utils import apply_color  # noqa: E402

        ax = extra_axes.pop(0)
        a_max, b_max = saved.get("color_a_max", 0.05), saved.get("color_b_max", 0.02)
        # A neutral gray ramp through the map: the clearest way to see a 12-parameter color
        # shift, which on a natural image is close to invisible by construction.
        ramp = torch.linspace(0, 1, 64).view(1, 1, 64).expand(3, 16, 64)
        both = torch.cat([ramp, apply_color(ramp, color_params, a_max, b_max)], dim=1)
        ax.imshow(both.permute(1, 2, 0).numpy())
        ax.axhline(16, color="white", linewidth=1)
        ax.set_title(
            f"Photometric map, gray ramp\ntop clean / bottom mapped, "
            f"||A||_inf={float(color_params[:, :3].abs().max()):.4f}, "
            f"||b||_inf={float(color_params[:, 3].abs().max()):.4f}"
        )
        ax.axis("off")
    if mask_floor is not None:
        from modules.optimizing_trigger.utils import contrast_mask  # noqa: E402

        ax = extra_axes.pop(0)
        w = contrast_mask(clean_t, saved.get("contrast_mask_ksize", 5), mask_floor)
        im = ax.imshow(w[0].numpy(), cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"Contrast mask on this image\n(floor={mask_floor}, mean={float(w.mean()):.3f})")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(
        f"{args.model_flag}/{args.dataset} seed{args.seed} -- {args.trigger_type} trigger\n{measured}"
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"[OK] trigger visualization saved to {out}")


if __name__ == "__main__":
    main()
