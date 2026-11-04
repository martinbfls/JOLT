#!/usr/bin/env python3
"""Self-contained checks of the trigger composition (apply_trigger_raw and its components).

Two things it is here to protect:

1. **Campaign comparability.** The photometric map, the contrast mask and the EOT draw are all
   opt-in. With every new knob at its default, the composition must be BIT-IDENTICAL to what it
   was before they existed, under each of the three trigger_types -- otherwise a new run is no
   longer comparable with the results already on disk.
2. **Optimizer/deploy agreement.** The optimization loop composes the trigger on a batch of
   tensors; the deploy-time poisoner composes it on one PIL image at a time. Both go through
   apply_trigger_raw, and the two paths must agree, or the attack that is measured is not the one
   that was optimized.

It imports modules.optimizing_trigger.compose, which depends on torch alone, so it needs no
dataset, no checkpoint, no model and not even the pytorch_cifar submodule, and runs in a second:

  python scripts/check_trigger_composition.py

Exit status 0 = every check passed; 1 = at least one failed (the failing lines are printed).
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.optimizing_trigger.compose import (  # noqa: E402
    apply_color,
    apply_trigger_raw,
    apply_warp,
    contrast_mask,
    eot_augment,
    init_color_params,
    tv_loss,
)

FAILURES = []


def check(name, cond):
    print(f"  [{'OK  ' if cond else 'FAIL'}] {name}")
    if not cond:
        FAILURES.append(name)


def main():
    torch.manual_seed(0)
    x = torch.rand(4, 3, 32, 32)
    delta = (torch.rand(3, 32, 32) - 0.5) * 0.06
    grid = (torch.rand(1, 2, 8, 8) - 0.5) * 4
    warp_s = 4.0
    color_zero = init_color_params(device="cpu")

    print("1. backward compatibility -- the default path must be bit-identical")
    hybrid_ref = (apply_warp(x, grid, warp_s) + delta.unsqueeze(0)).clamp(0, 1)
    check("hybrid == clamp(apply_warp(x) + delta, 0, 1)", torch.equal(hybrid_ref, apply_trigger_raw(x, delta, grid, warp_s)))
    check("additive == clamp(x + delta, 0, 1)", torch.equal((x + delta.unsqueeze(0)).clamp(0, 1), apply_trigger_raw(x, delta)))
    check("warp == apply_warp(x)", torch.equal(apply_warp(x, grid, warp_s), apply_trigger_raw(x, None, grid, warp_s)))
    check("zero color_params is the exact identity map", torch.equal(apply_trigger_raw(x, delta, grid, warp_s, color_params=color_zero), hybrid_ref))
    check("eot_augment(pad=0, flip=False) is the identity", torch.equal(eot_augment(x, 0, False), x))

    print("2. photometric map -- budgets, range, formula")
    color = torch.randn(3, 4) * 10  # far outside the budgets, to exercise the internal clamp
    a_max, b_max = 0.05, 0.02
    out = apply_color(x, color, a_max, b_max)
    A, b = color[:, :3].clamp(-a_max, a_max), color[:, 3].clamp(-b_max, b_max)
    ref = (torch.einsum("ij,bjhw->bihw", torch.eye(3) + A, x) + b.view(1, 3, 1, 1)).clamp(0, 1)
    check("equals clamp((I + clamp(A)) x + clamp(b), 0, 1)", torch.allclose(out, ref, atol=1e-6))
    check("output stays in [0, 1]", bool(out.min() >= 0 and out.max() <= 1))
    check("budgets enforced inside apply_color", bool(A.abs().max() <= a_max + 1e-9 and b.abs().max() <= b_max + 1e-9))

    print("3. contrast mask -- epsilon must stay a valid bound on the deployed perturbation")
    floor = 0.25
    w = contrast_mask(x, 5, floor)
    check(f"weights in [{floor}, 1]", bool(w.min() >= floor - 1e-6 and w.max() <= 1 + 1e-6))
    check("mask is achromatic (one channel)", w.shape[1] == 1)
    masked = apply_trigger_raw(x, delta, grid, warp_s, contrast_mask_floor=floor)
    check(
        "||masked additive||_inf <= ||delta||_inf",
        bool((masked - apply_warp(x, grid, warp_s)).abs().max() <= delta.abs().max() + 1e-6),
    )
    flat = contrast_mask(torch.full((1, 3, 32, 32), 0.5), 5, floor)
    check("a flat image gets the floor everywhere", bool((flat - floor).abs().max() < 1e-5))

    print("4. optimizer path == deploy path")
    kwargs = dict(color_params=color, contrast_mask_floor=floor, contrast_mask_ksize=5)
    check(
        "(C, H, W) (the poisoner's) == row 0 of (B, C, H, W) (the optimizer's)",
        torch.allclose(
            apply_trigger_raw(x[0], delta, grid, warp_s, **kwargs),
            apply_trigger_raw(x, delta, grid, warp_s, **kwargs)[0],
            atol=1e-6,
        ),
    )

    print("5. EOT")
    check("shape preserved", eot_augment(x, 4, True).shape == x.shape)
    x_grad = x.clone().requires_grad_(True)
    eot_augment(x_grad, 4, True).sum().backward()
    check("differentiable w.r.t. the image", x_grad.grad is not None and bool(x_grad.grad.abs().sum() > 0))
    flipped = [
        float((eot_augment(x, 0, True)[i] - x[i]).abs().sum() > 0) for _ in range(40) for i in range(x.shape[0])
    ]
    rate = sum(flipped) / len(flipped)
    check(f"flip drawn per sample, rate={rate:.2f} (expected ~0.5)", 0.3 < rate < 0.7)

    print("6. gradients reach every optimized component")
    d2, g2, c2 = delta.clone().requires_grad_(True), grid.clone().requires_grad_(True), init_color_params(device="cpu")
    composed = apply_trigger_raw(x, d2, g2, warp_s, color_params=c2, contrast_mask_floor=floor)
    eot_augment(composed, 4, True).pow(2).sum().backward()
    for name, var in (("delta", d2), ("control_grid", g2), ("color_params", c2)):
        check(f"d/d {name} is nonzero", var.grad is not None and bool(var.grad.abs().sum() > 0))

    print("7. lambda_tv_warp's penalty accepts the control grid")
    tv = tv_loss(grid.squeeze(0))
    check("tv_loss((2, k, k)) is a finite scalar", bool(torch.isfinite(tv)) and tv.dim() == 0)

    print()
    if FAILURES:
        print(f"[FAIL] {len(FAILURES)} check(s) failed:")
        for name in FAILURES:
            print(f"  - {name}")
        return 1
    print("[OK] every trigger-composition check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
