#!/usr/bin/env python3
"""Summarizes CTA/ASR across seeds for the federated_3vs7 branch of a paper_main_campaign cell,
one row per budget, one column pair per agg_method (by default krum vs krum_full vs multikrum
vs multikrum_full) -- i.e. does the full-gradient aggregation rule change the attack's success.

Reads the same logs/federated_train_user_metrics.json files run_experiment.py's train_user
produces, under
  <exp-base>/<model_flag>/<dataset>/<tag>/seed<seed>/federated_3vs7/<agg_method>/train_user_<budget>/logs/...

Usage (from repo root, with the JOLT conda env active):
  python scripts/show_results/summarize_full_grad_vs_layerwise.py \\
      --exp-base experiments/out_iclr --model-flag r32p --dataset cifar --tag baseline \\
      --seeds 0 1 2 3 4 5 6 7 8 9 --budgets 0 150 300 500 1000 2000 2500 5000 \\
      --out experiments/out_iclr/r32p/cifar/baseline/full_grad_vs_layerwise_summary
"""

import argparse
import json
import statistics
from pathlib import Path

FEDERATED_TAG = "federated_3vs7"


def load_final(train_user_dir):
    path = Path(train_user_dir) / "logs" / "federated_train_user_metrics.json"
    if not path.exists():
        return None, None
    with open(path) as f:
        data = json.load(f)
    cta = data.get("cta", [])
    asr = data.get("asr", [])
    return (cta[-1]["value"] if cta else None), (asr[-1]["value"] if asr else None)


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
    ap.add_argument("--exp-base", default="experiments/out_iclr")
    ap.add_argument("--model-flag", default="r32p")
    ap.add_argument("--dataset", default="cifar")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--budgets", type=int, nargs="+", default=[0, 150, 300, 500, 1000, 2000, 2500, 5000])
    ap.add_argument("--agg-methods", nargs="+", default=["krum", "krum_full", "multikrum", "multikrum_full"])
    ap.add_argument("--out", default=None, help="base path (no extension) for .md and .csv; omit to only print")
    args = ap.parse_args()

    base = Path(args.exp_base) / args.model_flag / args.dataset / args.tag
    n_seeds = len(args.seeds)

    rows = []
    for budget in args.budgets:
        row = {"budget": budget}
        for agg in args.agg_methods:
            ctas, asrs = [], []
            for seed in args.seeds:
                train_user_dir = base / f"seed{seed}" / FEDERATED_TAG / agg / f"train_user_{budget}"
                cta, asr = load_final(train_user_dir)
                ctas.append(cta)
                asrs.append(asr)
            row[f"{agg}_cta"] = mean_std(ctas)
            row[f"{agg}_asr"] = mean_std(asrs)
        rows.append(row)

    header = ["budget"] + [f"{agg} cta" for agg in args.agg_methods] + [f"{agg} asr" for agg in args.agg_methods]
    lines_md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    csv_header = ["budget"]
    for agg in args.agg_methods:
        csv_header += [f"{agg}_cta_mean", f"{agg}_cta_std", f"{agg}_asr_mean", f"{agg}_asr_std", f"{agg}_n"]
    lines_csv = [",".join(csv_header)]

    for row in rows:
        cta_cells = [fmt(*row[f"{agg}_cta"], n_seeds) for agg in args.agg_methods]
        asr_cells = [fmt(*row[f"{agg}_asr"], n_seeds) for agg in args.agg_methods]
        lines_md.append("| " + " | ".join([str(row["budget"])] + cta_cells + asr_cells) + " |")

        csv_fields = [str(row["budget"])]
        for agg in args.agg_methods:
            cta_m, cta_s, cta_n = row[f"{agg}_cta"]
            asr_m, asr_s, _ = row[f"{agg}_asr"]
            csv_fields += [
                "" if cta_m is None else str(cta_m), "" if cta_s is None else str(cta_s),
                "" if asr_m is None else str(asr_m), "" if asr_s is None else str(asr_s),
                str(cta_n),
            ]
        lines_csv.append(",".join(csv_fields))

    table_md = "\n".join(lines_md)
    print(f"[SUMMARY] {args.model_flag}/{args.dataset}/{args.tag}, up to {n_seeds} seeds, "
          f"agg_methods={args.agg_methods}")
    print(table_md)

    if args.out:
        out_md, out_csv = Path(f"{args.out}.md"), Path(f"{args.out}.csv")
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(table_md + "\n")
        out_csv.write_text("\n".join(lines_csv) + "\n")
        print(f"[OK] written to {out_md} and {out_csv}")


if __name__ == "__main__":
    main()
