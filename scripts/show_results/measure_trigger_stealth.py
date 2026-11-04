#!/usr/bin/env python3
"""Measures how visible each campaign's trigger actually is, and plots ASR against that -- the
only comparison that answers "which trigger is least visible at equal attack success".

`epsilon` cannot play that role any more. It bounds the additive component only, so it says
nothing about a warp or a photometric map, and once `contrast_mask_floor` reweights the budget it
is an upper bound rather than an equality. This script instead reconstructs each saved trigger
with its real deploy-time poisoner, applies it to real source-class test images, and measures the
perturbation that comes out:

  Linf, L2   the deployed pixel difference, after the poisoner's PIL round-trip -- so the 1/255
             quantization every deployed trigger goes through is included, not idealized away
  PSNR       dB, the conventional scale-free summary
  SSIM       structural similarity (11x11 Gaussian window, per channel then averaged)
  LPIPS      perceptual distance (lpips.LPIPS(net='alex')), the closest of these to "visible to a
             person"; skipped with a note when the `lpips` package is absent

It then joins those with the deployed ASR and draws the Pareto view: perceptual cost on x, ASR on
y, one point per variant. A variant that moves up and to the left is a real improvement; one that
moves up and to the right only bought ASR with visibility.

Usage (from repo root, with the JOLT conda env active):
  python scripts/show_results/measure_trigger_stealth.py \\
      --variant-tags hybrid_eps0p0627_s8p0 hybrid_tv_k8 --seeds 0 1 2 --budgets 150 500 1500

Outputs under --out-dir: stealth.{md,csv} (the table), pareto.{png,pdf}, per_image.csv.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from modules.base_utils import plot_style  # noqa: E402
from modules.base_utils.plot_style import plt  # noqa: E402
from modules.base_utils.datasets import load_dataset, pick_poisoner_for_trigger  # noqa: E402
from plot_warp_multibudget import read_final  # noqa: E402

SOURCE_LABEL = 9
TARGET_LABEL = 4
# Metric -> (column label, "lower is stealthier"). The Pareto plot uses the first available one.
STEALTH_METRICS = {
    "lpips": ("LPIPS", True),
    "ssim": ("SSIM", False),
    "psnr": ("PSNR (dB)", False),
    "linf": ("||x'-x||_inf", True),
    "l2": ("||x'-x||_2", True),
}


def gaussian_window(size=11, sigma=1.5):
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :]).view(1, 1, size, size)


def ssim(x, y, window=None, data_range=1.0):
    """Mean SSIM over a (B, C, H, W) pair, channels treated independently then averaged -- the
    standard 11x11 Gaussian-window formulation, implemented here to avoid a scikit-image
    dependency the rest of the repo does not have."""
    window = gaussian_window() if window is None else window
    C = x.shape[1]
    w = window.expand(C, 1, -1, -1).to(x.dtype)
    pad = window.shape[-1] // 2
    mu_x = F.conv2d(x, w, padding=pad, groups=C)
    mu_y = F.conv2d(y, w, padding=pad, groups=C)
    sxx = F.conv2d(x * x, w, padding=pad, groups=C) - mu_x ** 2
    syy = F.conv2d(y * y, w, padding=pad, groups=C) - mu_y ** 2
    sxy = F.conv2d(x * y, w, padding=pad, groups=C) - mu_x * mu_y
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    num = (2 * mu_x * mu_y + c1) * (2 * sxy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2)
    return (num / den).flatten(1).mean(dim=1)


def trigger_path(args, tag, seed, init="stripe"):
    return (
        Path(args.exp_root)
        / f"{args.exp_base_prefix}_{tag}"
        / args.model_flag
        / args.dataset
        / "1vs0"
        / args.gen_agg_method
        / f"seed{seed}"
        / "gen_labels_trigger_joint"
        / "trigger"
        / f"opt_trig_direct_joint_{init}_{args.model_flag}_{args.dataset}_1vs0.pt"
    )


def deploy_dir(args, tag, seed, budget, federated=True):
    base = trigger_path(args, tag, seed).parents[2]
    if not federated:
        return base / f"train_user_{budget}"
    deploy = f"{args.deploy_agg_method}_{args.deploy_num_poisoned}vs{args.deploy_num_honests}"
    return base / deploy / f"train_user_{budget}"


def load_clean_images(args):
    """`--n-images` source-class TEST images, as a (N, 3, H, W) tensor in [0, 1] plus the PIL
    originals the poisoner needs -- the ASR test set's own population, so the measurement is made
    on the images the attack is actually scored on."""
    dataset = load_dataset(args.dataset, train=False)
    to_tensor = transforms.ToTensor()
    pils, tensors = [], []
    for x, y in dataset:
        if y != SOURCE_LABEL:
            continue
        pils.append(x)
        tensors.append(to_tensor(x))
        if len(pils) >= args.n_images:
            break
    if not pils:
        raise SystemExit(f"[ABORT] no class-{SOURCE_LABEL} image in the {args.dataset} test set")
    return pils, torch.stack(tensors)


def measure_one(args, trig_file, pils, clean, lpips_model):
    """Per-image stealth metrics for one saved trigger, as a DataFrame of `len(pils)` rows."""
    poisoner = pick_poisoner_for_trigger(args.dataset, TARGET_LABEL, str(trig_file))
    to_tensor = transforms.ToTensor()
    poisoned = torch.stack([to_tensor(poisoner.poison((p, SOURCE_LABEL))[0]) for p in pils])

    diff = (poisoned - clean).flatten(1)
    mse = diff.pow(2).mean(dim=1)
    rows = {
        "linf": diff.abs().amax(dim=1).numpy(),
        "l2": diff.norm(dim=1).numpy(),
        # 1.0 is the max value of the [0, 1] data range; an exact match gets inf, hence the floor
        "psnr": (10 * torch.log10(1.0 / mse.clamp_min(1e-12))).numpy(),
        "ssim": ssim(poisoned, clean).numpy(),
    }
    if lpips_model is not None:
        with torch.no_grad():
            # lpips expects [-1, 1]
            rows["lpips"] = lpips_model(poisoned * 2 - 1, clean * 2 - 1).flatten().numpy()
    return pd.DataFrame(rows)


def collect(args, pils, clean, lpips_model):
    per_image, missing = [], []
    for tag in args.variant_tags:
        for seed in args.seeds:
            trig_file = trigger_path(args, tag, seed)
            if not trig_file.exists():
                missing.append((tag, seed, trig_file))
                continue
            df = measure_one(args, trig_file, pils, clean, lpips_model)
            df.insert(0, "seed", seed)
            df.insert(0, "tag", tag)
            per_image.append(df)
    if not per_image:
        return pd.DataFrame(), pd.DataFrame(), missing

    per_image = pd.concat(per_image, ignore_index=True)
    metrics = [m for m in STEALTH_METRICS if m in per_image.columns]
    stealth = per_image.groupby(["tag", "seed"])[metrics].mean().reset_index()

    rows = []
    for _, s in stealth.iterrows():
        for budget in args.budgets:
            _, fed_asr, fed_src = read_final(deploy_dir(args, s["tag"], int(s["seed"]), budget, True))
            cen_cta, cen_asr, _ = read_final(deploy_dir(args, s["tag"], int(s["seed"]), budget, False))
            rows.append({
                **{k: s[k] for k in ["tag", "seed", *metrics]},
                "budget": budget, "federated_asr": fed_asr,
                "centralized_asr": cen_asr, "centralized_cta": cen_cta,
                "has_deploy": fed_src is not None,
            })
    return per_image, pd.DataFrame(rows), missing


def write_table(joined, args, out_dir, metrics):
    agg_cols = [*metrics, "federated_asr", "centralized_asr"]
    means = joined.groupby(["tag", "budget"])[agg_cols].agg(["mean", "std"])
    means.columns = [f"{c}_{s}" for c, s in means.columns]
    means = means.reset_index()

    header = ["tag", "budget"] + [STEALTH_METRICS[m][0] for m in metrics] + ["fed ASR (%)", "cen ASR (%)"]
    lines = [
        "# Measured trigger stealth vs. ASR",
        "",
        f"`{args.model_flag}/{args.dataset}`, {len(args.seeds)} seed(s), {args.n_images} "
        f"source-class test images per trigger, deploy branch "
        f"`{args.deploy_agg_method}_{args.deploy_num_poisoned}vs{args.deploy_num_honests}`.",
        "",
        "Stealth is measured on the perturbation as DEPLOYED (the poisoner's PIL round-trip "
        "included, so 1/255 quantization counts). Lower LPIPS / higher SSIM / higher PSNR = "
        "stealthier. Compare variants along a ROW of equal budget: a variant is better only if "
        "it raises ASR without raising the perceptual cost.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for _, row in means.iterrows():
        cells = [str(row["tag"]), str(row["budget"])]
        for m in metrics:
            prec = 4 if m in ("lpips", "linf", "ssim") else 2
            cells.append(f"{row[f'{m}_mean']:.{prec}f}")
        for m in ("federated_asr", "centralized_asr"):
            mean, std = row[f"{m}_mean"], row[f"{m}_std"]
            cells.append("--" if pd.isna(mean) else f"{mean:.1f} +/- {0.0 if pd.isna(std) else std:.1f}")
        lines.append("| " + " | ".join(cells) + " |")

    text = "\n".join(lines) + "\n"
    (out_dir / "stealth.md").write_text(text)
    means.to_csv(out_dir / "stealth.csv", index=False)
    return text, means


def draw_pareto(means, args, out_dir, metric):
    """ASR against measured perceptual cost, one marker per (variant, budget). Up-and-left wins."""
    plot_style.apply()
    label, lower_is_better = STEALTH_METRICS[metric]
    fig, ax = plt.subplots(figsize=plot_style.SMALL_FIGSIZE)
    cmap = plt.get_cmap("tab10")
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    for i, tag in enumerate(args.variant_tags):
        cell = means[means["tag"] == tag].sort_values("budget")
        if cell.empty:
            continue
        xs, ys = cell[f"{metric}_mean"].to_numpy(float), cell["federated_asr_mean"].to_numpy(float)
        ax.plot(xs, ys, color=cmap(i % cmap.N), alpha=0.5, linewidth=1, zorder=1)
        for j, (_, row) in enumerate(cell.iterrows()):
            ax.scatter(row[f"{metric}_mean"], row["federated_asr_mean"],
                       color=cmap(i % cmap.N), marker=markers[j % len(markers)], s=70, zorder=3,
                       label=f"{tag}, budget {int(row['budget'])}")

    arrow = "lower = stealthier" if lower_is_better else "higher = stealthier"
    plot_style.style_axes(ax, f"{label} ({arrow})", "Federated ASR (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7, loc="best")
    ax.set_title(f"{args.model_flag}/{args.dataset} -- ASR at equal measured visibility")
    return plot_style.save(fig, out_dir / "pareto")


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
    ap.add_argument("--n-images", type=int, default=256, help="source-class test images per trigger")
    ap.add_argument("--pareto-metric", default=None, choices=sorted(STEALTH_METRICS),
                    help="x axis of the Pareto figure; default: LPIPS when available, else SSIM")
    ap.add_argument("--out-dir", default="experiments/federated_experiments/trigger_stealth")
    args = ap.parse_args()

    lpips_model = None
    try:
        import lpips as lpips_pkg

        lpips_model = lpips_pkg.LPIPS(net="alex")
        lpips_model.eval()
        for p in lpips_model.parameters():
            p.requires_grad_(False)
    except ImportError:
        print("[NOTE] the `lpips` package is not installed -- reporting SSIM/PSNR/Linf/L2 only "
              "(pip install lpips for the perceptual metric)")

    pils, clean = load_clean_images(args)
    print(f"[INFO] measuring on {len(pils)} class-{SOURCE_LABEL} {args.dataset} test images")
    per_image, joined, missing = collect(args, pils, clean, lpips_model)

    if joined.empty:
        print(f"[ABORT] no trigger found under {args.exp_root}/{args.exp_base_prefix}_<tag>/")
        for tag, seed, path in missing[:10]:
            print(f"  {tag} seed{seed}: {path}")
        raise SystemExit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_image.to_csv(out_dir / "per_image.csv", index=False)

    metrics = [m for m in STEALTH_METRICS if m in per_image.columns]
    text, means = write_table(joined, args, out_dir, metrics)
    metric = args.pareto_metric or ("lpips" if "lpips" in metrics else "ssim")
    written = draw_pareto(means, args, out_dir, metric)

    print(text)
    if not joined["has_deploy"].any():
        print("[NOTE] no deployed ASR found yet -- the stealth columns are final, the ASR ones "
              "will fill in once the train_user jobs finish.")
    if missing:
        print(f"[MISSING] {len(missing)} trigger(s) not on disk:")
        for tag, seed, path in missing[:10]:
            print(f"  {tag} seed{seed}: {path}")
    print("[OK] written:")
    for path in [out_dir / "stealth.md", out_dir / "stealth.csv", out_dir / "per_image.csv", *written]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
