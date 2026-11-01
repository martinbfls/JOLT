#!/usr/bin/env python3
"""Summarizes a sweep_warp_min_perturbation_slurm.sh campaign into one table: for each
(warp_s, lambda_lpips) cell, mean +/- std CTA/ASR across seeds, centralized vs federated.

Reads the same logs/federated_train_user_metrics.json files the sweep's cells produce, under
  experiments/federated_experiments/threat_model_direct_trigger_joint_warp_s<S>_lpips<L>/
      <model>/<dataset>/1vs0/mean/seed<seed>/train_user_<budget>/logs/...                       (centralized)
      <model>/<dataset>/1vs0/mean/seed<seed>/<agg>_<P>vs<H>/train_user_<budget>/logs/...         (federated)

Usage (from repo root, with the JOLT conda env active):
  python scripts/show_results/summarize_warp_sweep.py \\
      --model-flag r32p --dataset cifar --budget 1500 --seeds 0 1 2 3 4 \\
      --out experiments/federated_experiments/warp_sweep_summary
"""

import argparse
import re
import statistics
from pathlib import Path

import json


def load_final(train_user_dir):
    """Returns (cta, asr) final values from logs/federated_train_user_metrics.json, or
    (None, None) if the file/keys are missing (run not finished)."""
    path = Path(train_user_dir) / "logs" / "federated_train_user_metrics.json"
    if not path.exists():
        return None, None
    with open(path) as f:
        data = json.load(f)
    cta = data.get("cta", [])
    asr = data.get("asr", [])
    return (cta[-1]["value"] if cta else None), (asr[-1]["value"] if asr else None)


def unsanitize(token):
    """Inverse of sweep_warp_min_perturbation_slurm.sh's sanitize(): 'p' -> '.' (values in this
    sweep are never negative, so the '-' stripping side of sanitize() has no inverse to undo)."""
    return token.replace("p", ".")


def discover_cells(exp_root, exp_base_prefix):
    """Finds every threat_model_direct_trigger_joint_warp_s<S>_lpips<L> dir and returns
    [(warp_s_str, lpips_str, dir_path), ...] sorted by (warp_s, lpips) as floats."""
    pattern = re.compile(rf"^{re.escape(exp_base_prefix)}_warp_s(?P<s>[^_]+)_lpips(?P<l>.+)$")
    cells = []
    for d in Path(exp_root).glob(f"{exp_base_prefix}_warp_s*_lpips*"):
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if not m:
            continue
        cells.append((unsanitize(m.group("s")), unsanitize(m.group("l")), d))
    cells.sort(key=lambda c: (float(c[0]), float(c[1])))
    return cells


def mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None, 0
    if len(values) == 1:
        return values[0], 0.0, 1
    return statistics.mean(values), statistics.stdev(values), len(values)


def fmt(mean, std, n, n_expected):
    if mean is None:
        return "--"
    flag = "" if n == n_expected else f" (n={n})"
    return f"{mean:.2f} +/- {std:.2f}{flag}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-root", default="experiments/federated_experiments")
    ap.add_argument("--exp-base-prefix", default="threat_model_direct_trigger_joint")
    ap.add_argument("--model-flag", default="r32p")
    ap.add_argument("--dataset", default="cifar")
    ap.add_argument("--budget", type=int, default=1500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--deploy-agg-method", default="multikrum")
    ap.add_argument("--deploy-num-poisoned", type=int, default=3)
    ap.add_argument("--deploy-num-honests", type=int, default=7)
    ap.add_argument("--out", default=None, help="base path (no extension) for .md and .csv; omit to only print")
    args = ap.parse_args()

    deploy_tag = f"{args.deploy_agg_method}_{args.deploy_num_poisoned}vs{args.deploy_num_honests}"
    cells = discover_cells(args.exp_root, args.exp_base_prefix)
    if not cells:
        print(f"[WARN] no cells found under {args.exp_root}/{args.exp_base_prefix}_warp_s*_lpips*")
        return

    rows = []
    for warp_s, lpips, cell_dir in cells:
        base = cell_dir / args.model_flag / args.dataset / "1vs0" / "mean"
        cen_cta, cen_asr, fed_cta, fed_asr = [], [], [], []
        for seed in args.seeds:
            seed_dir = base / f"seed{seed}"
            c_cta, c_asr = load_final(seed_dir / f"train_user_{args.budget}")
            f_cta, f_asr = load_final(seed_dir / deploy_tag / f"train_user_{args.budget}")
            cen_cta.append(c_cta); cen_asr.append(c_asr)
            fed_cta.append(f_cta); fed_asr.append(f_asr)
        rows.append({
            "warp_s": warp_s, "lambda_lpips": lpips,
            "cen_cta": mean_std(cen_cta), "cen_asr": mean_std(cen_asr),
            "fed_cta": mean_std(fed_cta), "fed_asr": mean_std(fed_asr),
        })

    n_seeds = len(args.seeds)
    header = ["warp_s", "lambda_lpips", "cen_cta", "cen_asr", f"fed_cta ({deploy_tag})", f"fed_asr ({deploy_tag})"]
    lines_md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines_csv = [",".join(["warp_s", "lambda_lpips", "cen_cta_mean", "cen_cta_std", "cen_asr_mean", "cen_asr_std",
                           "fed_cta_mean", "fed_cta_std", "fed_asr_mean", "fed_asr_std", "n_seeds"])]
    for r in rows:
        cen_cta_m, cen_cta_s, cen_cta_n = r["cen_cta"]
        cen_asr_m, cen_asr_s, cen_asr_n = r["cen_asr"]
        fed_cta_m, fed_cta_s, fed_cta_n = r["fed_cta"]
        fed_asr_m, fed_asr_s, fed_asr_n = r["fed_asr"]
        lines_md.append("| " + " | ".join([
            r["warp_s"], r["lambda_lpips"],
            fmt(cen_cta_m, cen_cta_s, cen_cta_n, n_seeds), fmt(cen_asr_m, cen_asr_s, cen_asr_n, n_seeds),
            fmt(fed_cta_m, fed_cta_s, fed_cta_n, n_seeds), fmt(fed_asr_m, fed_asr_s, fed_asr_n, n_seeds),
        ]) + " |")
        lines_csv.append(",".join(str(x) for x in [
            r["warp_s"], r["lambda_lpips"],
            cen_cta_m if cen_cta_m is not None else "", cen_cta_s if cen_cta_s is not None else "",
            cen_asr_m if cen_asr_m is not None else "", cen_asr_s if cen_asr_s is not None else "",
            fed_cta_m if fed_cta_m is not None else "", fed_cta_s if fed_cta_s is not None else "",
            fed_asr_m if fed_asr_m is not None else "", fed_asr_s if fed_asr_s is not None else "",
            fed_asr_n,
        ]))

    table_md = "\n".join(lines_md)
    print(f"[SUMMARY] {len(cells)} cells, up to {n_seeds} seeds each (budget={args.budget}, {deploy_tag})")
    print(table_md)

    if args.out:
        out_md, out_csv = Path(f"{args.out}.md"), Path(f"{args.out}.csv")
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(table_md + "\n")
        out_csv.write_text("\n".join(lines_csv) + "\n")
        print(f"[OK] written to {out_md} and {out_csv}")


if __name__ == "__main__":
    main()
