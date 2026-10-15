#!/usr/bin/env python3
"""Summarizes run_backdoorbench.sh's BackdoorBench records into a long CSV and an appendix
LaTeX table (same layout as plot_paper_main_campaign_user.py's tables).

Per record: the victim's final CTA/ASR (attack_df.csv) and, per detector, TPR/FPR (in %) from
its detection_info.csv. Data-level detectors score the flipped TRAINING samples (clean images,
JOLT labels); inference detectors score triggered TEST inputs against clean ones.

  python backdoorbench/collect_backdoorbench.py --bb-dir ~/BackdoorBench
  python backdoorbench/collect_backdoorbench.py --bb-dir ~/BackdoorBench --campaign flip_frozen --out tables_backdoorbench
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

RECORD = re.compile(r"^jolt_(?P<campaign>jolt|flip_frozen)_(?P<model>[a-z0-9_]+?)_cifar_(?P<tag>.+)_seed(?P<seed>\d+)_b(?P<budget>\d+)$")
REFERENCE = re.compile(r"^ref_(?P<attack>[a-z0-9]+)_cifar_t\d+$")
DETECTOR_DIR = re.compile(r"(?P<method>[a-z0-9]+)_(?P<family>pretrain|infer)$")

DETECTOR_DISPLAY = {
    "ac": "AC", "spectral": "SS", "spectre": "SPECTRE", "scan": "SCAn", "beatrix": "Beatrix", "agpd": "AGPD",
    "asset": "ASSET", "cd": "CD", "abl": "ABL", "strip": "STRIP", "sentinet": "SentiNet", "teco": "TeCo",
}
FAMILY_ORDER = {"pretrain": 0, "infer": 1}
TAG_DISPLAY = {"baseline": r"Optimized Trigger ($\varepsilon=1.0$)", "eps_16_255": r"Optimized Trigger ($\varepsilon=16/255$)",
               "eps_0p50_lpips_0p1": r"Optimized Trigger ($\varepsilon=0.5$, LPIPS$=0.1$)"}
CAMPAIGN_DISPLAY = {"jolt": r"\jolt{}", "flip_frozen": "FLIP"}


def last_row(path):
    """Last data row of a BackdoorBench detection_info.csv (its header is re-written on every append)."""
    rows = [r for r in csv.reader(open(path)) if r and r[0] != "record"]
    return rows[-1] if rows else None


def read_record(rec_dir):
    out = []
    df = rec_dir / "attack_df.csv"
    victim = {}
    if df.exists():
        last = pd.read_csv(df).iloc[-1]
        victim = {k: float(last[k]) * 100 for k in ("test_acc", "test_asr") if k in last}
    for path in rec_dir.glob("detection*/**/detection_info.csv"):
        rel = path.relative_to(rec_dir).parts
        if rel[0] == "detection_pretrain":  # agpd: detection_pretrain/agpd/
            method, family = rel[1], "pretrain"
        else:
            m = DETECTOR_DIR.match(rel[1])
            if not m:
                continue
            method, family = m["method"], m["family"]
        row = last_row(path)
        if row is None:
            continue
        # TPR/FPR as BackdoorBench reports them: when nothing is flagged it writes TPR=FPR=0 but
        # swapped TN/FP/FN/TP counts (FP=#poisoned, FN=0), so the counts must not be used to recompute them.
        tn, fp, fn, tp, tpr, fpr = (float(x) for x in row[1:7])
        out.append({"method": method, "family": family, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
                    "TPR": 100 * tpr, "FPR": 100 * fpr, "flagged_none": row[7] == "None"})
    return victim, out


def collect(bb_dir):
    rows = []
    for rec_dir in sorted((bb_dir / "record").iterdir()):
        if not rec_dir.is_dir() or ".incomplete." in rec_dir.name:
            continue
        m, r = RECORD.match(rec_dir.name), REFERENCE.match(rec_dir.name)
        if not (m or r):
            continue
        meta = ({**m.groupdict(), "attack": "jolt"} if m else
                {"campaign": "reference", "attack": r["attack"], "tag": None, "seed": None, "budget": None, "model": None})
        victim, detections = read_record(rec_dir)
        base = {"record": rec_dir.name, **meta, "victim_cta": victim.get("test_acc"), "victim_asr": victim.get("test_asr")}
        rows += [{**base, **d} for d in detections] or [base]
    df = pd.DataFrame(rows)
    for col in ("seed", "budget"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def cell(values_tpr, values_fpr):
    t, f = np.asarray(values_tpr, float), np.asarray(values_fpr, float)
    if len(t) == 0 or np.all(np.isnan(t)):
        return "--"
    return f"{np.nanmean(t):.1f}$\\pm${np.nanstd(t):.1f}/{np.nanmean(f):.1f}$\\pm${np.nanstd(f):.1f}"


def latex_table(df, campaign, references):
    jolt = df[(df["campaign"] == campaign) & df["method"].notna()]
    budgets = sorted(jolt["budget"].dropna().unique().astype(int))
    refs = [a for a in references if not df[(df["attack"] == a) & df["method"].notna()].empty]
    cols = [str(b) for b in budgets] + [a.capitalize() if a != "badnet" else "BadNets" for a in refs]
    methods = sorted(df["method"].dropna().unique(), key=lambda m: (FAMILY_ORDER[df[df["method"] == m]["family"].iloc[0]], m))
    lines = [
        "\\begin{table}[ht]", "\\centering", "% \\tiny", "% \\setlength{\\tabcolsep}{1pt}", "",
        f"\\caption{{BackdoorBench detection of {CAMPAIGN_DISPLAY[campaign]} on CIFAR-10 (victim: PreAct-ResNet18 trained "
        "by BackdoorBench on the poisoned training set, centralized deployment), against deployment budget"
        + (" and BackdoorBench's own attacks" if refs else "") +
        ". Results are reported as TPR/FPR (in \\%) with standard deviation over seeds. Data-level detectors "
        "(top) look for the poisoned training samples, inference-time detectors (bottom) for triggered test inputs.}",
        "\\vspace{2mm}", "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{@{}" + " ".join(["c"] * (1 + len(cols))) + "@{}}", "\\toprule",
        "Detector & " + " & ".join(cols) + " \\\\",
    ]
    for tag in [t for t in TAG_DISPLAY if t in set(jolt["tag"])] + sorted(set(jolt["tag"]) - set(TAG_DISPLAY)):
        sub = jolt[jolt["tag"] == tag]
        lines += ["\\midrule", f"\\multicolumn{{{1 + len(cols)}}}{{c}}{{\\textbf{{{TAG_DISPLAY.get(tag, tag)} -- "
                               f"{CAMPAIGN_DISPLAY[campaign]}}}}} \\\\", "\\midrule"]
        victim = sub.drop_duplicates("record")
        row = ["Victim CTA/ASR"] + [
            cell(victim[victim["budget"] == b]["victim_cta"], victim[victim["budget"] == b]["victim_asr"]).replace("/", " / ")
            for b in budgets]
        for a in refs:
            v = df[df["attack"] == a].drop_duplicates("record")
            row.append(cell(v["victim_cta"], v["victim_asr"]).replace("/", " / "))
        lines.append(" & ".join(row) + " \\\\")
        lines.append("\\midrule")
        for i, m in enumerate(methods):
            if i and df[df["method"] == m]["family"].iloc[0] != df[df["method"] == methods[i - 1]]["family"].iloc[0]:
                lines.append("\\midrule")
            row = [DETECTOR_DISPLAY.get(m, m)]
            for b in budgets:
                s = sub[(sub["method"] == m) & (sub["budget"] == b)]
                row.append(cell(s["TPR"], s["FPR"]))
            for a in refs:
                s = df[(df["attack"] == a) & (df["method"] == m)]
                row.append(cell(s["TPR"], s["FPR"]))
            lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "}", f"\\label{{tab:backdoorbench_{campaign}_cifar}}", "\\end{table}"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bb-dir", required=True, help="BackdoorBench checkout used by run_backdoorbench.sh")
    parser.add_argument("--campaign", choices=list(CAMPAIGN_DISPLAY), default="jolt")
    parser.add_argument("--references", nargs="*", default=["badnet", "blended"])
    parser.add_argument("--out", default="./tables_backdoorbench")
    args = parser.parse_args()

    df = collect(Path(args.bb_dir).expanduser())
    if df.empty or "method" not in df or df[df["campaign"] == args.campaign]["method"].dropna().empty:
        raise SystemExit(f"No {args.campaign} detection results under {args.bb_dir}/record/ yet.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "backdoorbench_results.csv", index=False)
    table = latex_table(df, args.campaign, args.references)
    (out / f"backdoorbench_{args.campaign}_cifar.tex").write_text(table)
    print(table)
    print(f"\n[INFO] Saved {out / 'backdoorbench_results.csv'} and {out / f'backdoorbench_{args.campaign}_cifar.tex'}")


if __name__ == "__main__":
    main()
