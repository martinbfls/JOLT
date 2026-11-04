#!/usr/bin/env python3
"""Builds a symlink view that makes a warp/hybrid campaign readable by the paper_main_campaign
results scripts, so run_all_out_iclr.sh's four visualizations work on it unchanged.

The two trees only differ in where the tag and the deployment sit:

  campaign (gen_configs.py --exp-base-tag)        paper (gen_configs_paper_main_campaign.py)
  <root>/<prefix>_<tag>/<model>/<dataset>/        <root>/<model>/<dataset>/<tag>/seed<seed>/
      1vs0/<gen_agg>/seed<seed>/
          train_user_<budget>/           <->          mean/train_user_<budget>/
          <agg>_<P>vs<H>/...             <->          federated_<P>vs<H>/<agg>/...
          select_flips/                  <->          select_flips/
          gen_labels_trigger_joint/      <->          gen_labels_trigger_joint/

so a handful of per-cell symlinks is enough -- no copying, and nothing in the real tree moves.
The view is rebuilt from scratch on every run (only its own symlinks are removed), so it stays
correct as a campaign fills in.

Pass the result as --jolt-root / --root to plot_paper_main_campaign_user.py, plot_asr_per_seed.py,
plot_flip_matrix.py and plot_trigger_grid.py. Note that those scripts WRITE their per-cell trigger
previews into gen_labels_trigger_joint/, which the view points at the real directory: that is the
same behavior they have on a paper campaign.

Usage (from repo root):
  python scripts/show_results/make_paper_layout_view.py \\
      --variant-tags hybrid_eps0p25_s8p0 hybrid_eps0p0627_s8p0 --seeds 0 1 2 \\
      --out experiments/federated_experiments/paper_layout_view
"""

import argparse
import shutil
from pathlib import Path


def campaign_cell(args, tag, seed):
    return (
        Path(args.exp_root).resolve()
        / f"{args.exp_base_prefix}_{tag}"
        / args.model_flag
        / args.dataset
        / "1vs0"
        / args.gen_agg_method
        / f"seed{seed}"
    )


def link(src: Path, dst: Path):
    """Points dst at src, replacing whatever the previous run of this script left there."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    dst.symlink_to(src, target_is_directory=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-root", default="experiments/federated_experiments")
    ap.add_argument("--exp-base-prefix", default="threat_model_direct_trigger_joint")
    ap.add_argument("--variant-tags", nargs="+", required=True)
    ap.add_argument("--model-flag", default="r32p")
    ap.add_argument("--dataset", default="cifar")
    ap.add_argument("--gen-agg-method", default="mean")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--deploy-agg-method", default="multikrum")
    ap.add_argument("--deploy-num-poisoned", type=int, default=3)
    ap.add_argument("--deploy-num-honests", type=int, default=7)
    ap.add_argument(
        "--federated-tag",
        default=None,
        help="directory name the paper layout expects for the federated branch; defaults to "
        "federated_<P>vs<H>, which must match FEDERATED_TAG in gen_configs_paper_main_campaign.py "
        "(it is hardcoded to 'federated_3vs7' there, so a deployment that is not 3vs7 will not be "
        "picked up by the paper scripts)",
    )
    ap.add_argument("--out", required=True, help="directory to build the view in")
    args = ap.parse_args()

    deploy = f"{args.deploy_agg_method}_{args.deploy_num_poisoned}vs{args.deploy_num_honests}"
    federated_tag = args.federated_tag or f"federated_{args.deploy_num_poisoned}vs{args.deploy_num_honests}"
    if federated_tag != "federated_3vs7":
        print(
            f"[WARN] federated_tag={federated_tag!r} but gen_configs_paper_main_campaign.py "
            "hardcodes FEDERATED_TAG='federated_3vs7' -- the paper scripts will not find this "
            "branch. The centralized one still works."
        )

    out = Path(args.out).resolve()
    linked, missing = 0, []
    for tag in args.variant_tags:
        for seed in args.seeds:
            cell = campaign_cell(args, tag, seed)
            if not cell.is_dir():
                missing.append(cell)
                continue
            view_cell = out / args.model_flag / args.dataset / tag / f"seed{seed}"
            # the centralized train_user_<budget>/ sit directly in the cell, where the paper
            # layout expects them one level down under the single_user aggregator's name
            link(cell, view_cell / "mean")
            for name in ("select_flips", "gen_labels_trigger_joint"):
                if (cell / name).is_dir():
                    link(cell / name, view_cell / name)
            if (cell / deploy).is_dir():
                link(cell / deploy, view_cell / federated_tag / args.deploy_agg_method)
            linked += 1

    print(f"[OK] view for {linked} cell(s) at {out}")
    if missing:
        print(f"[INFO] {len(missing)} cell(s) not generated yet (skipped):")
        for cell in missing[:10]:
            print(f"  {cell}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
    if linked == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
