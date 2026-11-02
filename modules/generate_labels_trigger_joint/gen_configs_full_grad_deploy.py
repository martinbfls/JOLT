"""Adds full-gradient krum_full/multikrum_full train_user configs to an EXISTING paper_main_campaign
cell, reusing its already-generated trigger and federated select_flips (no GEN/FLIPS re-run).

Targets exactly the on-disk layout gen_configs_paper_main_campaign.py produces:
  experiments/<exp-base>/<model_flag>/<dataset>/<tag>/seed<seed>/federated_3vs7/{select_flips,<agg_method>/train_user_<budget>}

Usage (from repo root, with the JOLT conda env active):
  python -m modules.generate_labels_trigger_joint.gen_configs_full_grad_deploy \\
      --exp-base experiments/out_iclr --model-flag r32p --dataset cifar --tag baseline \\
      --seeds 0 1 2 3 4 5 6 7 8 9 --agg-methods krum_full multikrum_full \\
      --budgets 0 150 300 500 1000 2000 2500 5000
"""

import argparse
from pathlib import Path

from modules.generate_labels_trigger_joint.gen_configs_paper_main_campaign import (
    LEARNING_RATE,
    WEIGHT_DECAY,
    MILESTONE,
    DEPLOY_FEDERATED_NUM_POISONED,
    DEPLOY_FEDERATED_NUM_HONESTS,
    DEPLOY_BUDGETS,
    FEDERATED_TAG,
    cell_name,
    joint_trigger_path,
    _train_user_configs,
)
from modules.base_utils.config_validation import (
    write_config,
    validate_config_file as validate_config,
)


def generate(exp_base, model_flag, dataset, tag, seeds, agg_methods, budgets, dry_run=False):
    lr = LEARNING_RATE.get(model_flag, 0.1)
    wd = WEIGHT_DECAY.get(model_flag, 2e-4)
    milestones = MILESTONE.get(model_flag, [75, 125])

    all_paths = []
    missing = []
    for seed in seeds:
        fed_dir = exp_base / cell_name(model_flag, dataset, tag, seed) / FEDERATED_TAG
        flips_dir = fed_dir / "select_flips"
        trigger_path = joint_trigger_path(model_flag, dataset, tag, seed, exp_base=exp_base)

        if not (flips_dir / "config.toml").exists():
            missing.append(str(flips_dir / "config.toml"))
        if not Path(trigger_path).exists():
            missing.append(str(trigger_path))

        configs = _train_user_configs(
            base_dir=fed_dir, budgets=budgets,
            num_honests=DEPLOY_FEDERATED_NUM_HONESTS, num_poisoned=DEPLOY_FEDERATED_NUM_POISONED,
            agg_methods=agg_methods, flips_dir=flips_dir,
            model_flag=model_flag, dataset=dataset, trigger_path=trigger_path,
            lr=lr, wd=wd, milestones=milestones,
        )
        for path, content in configs.items():
            assert "out/checkpoints" not in str(path), f"Refusing to write under out/checkpoints/: {path}"
            if not dry_run:
                write_config(path, content)
                validate_config(path)
            all_paths.append(path)

    return all_paths, missing


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp-base", default="experiments/out_iclr")
    parser.add_argument("--model-flag", default="r32p")
    parser.add_argument("--dataset", default="cifar")
    parser.add_argument("--tag", default="baseline")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--agg-methods", nargs="+", default=["krum_full", "multikrum_full"])
    parser.add_argument("--budgets", type=int, nargs="+", default=DEPLOY_BUDGETS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    exp_base = Path(args.exp_base)
    paths, missing = generate(
        exp_base, args.model_flag, args.dataset, args.tag,
        args.seeds, args.agg_methods, args.budgets, dry_run=args.dry_run,
    )

    verb = "would be written" if args.dry_run else "written and schema-validated"
    print(f"\ngen_configs_full_grad_deploy: {len(paths)} config files {verb} "
          f"({args.agg_methods} x seeds={args.seeds} x budgets={args.budgets}).")
    if missing:
        print(f"\n[WARN] {len(missing)} expected pre-existing path(s) not found "
              f"(select_flips config or trigger .pt) -- train_user will fail at runtime for those:")
        for p in missing:
            print(f"  {p}")
