"""FLIP with a frozen JOLT trigger: for each paper_main_campaign cell, takes the trigger the joint
labels+trigger optimization produced and runs the FLIP pipeline with it held fixed.

Per (model_flag, dataset, tag, seed), reading the trigger of the SAME paper_main_campaign cell:
  train_expert   experts trained on data poisoned with that trigger (poisoner = "optimized")
  generate_labels  FLIP trajectory-matching label optimization, trigger fixed
  select_flips / train_user  same single_user + federated_3vs7 deployment as paper_main_campaign

Same directory layout as paper_main_campaign ({model}/{dataset}/{tag}/seed{seed}/...), so the
orchestrators and plot_paper_main_campaign_user.py read it through EXP_BASE alone. The joint
campaign's GEN phase must have run first (its trigger .pt files are only read at run time).
"""

import argparse
from pathlib import Path

from modules.base_utils.config_validation import (
    write_config,
    validate_config_file as validate_config,
)
from modules.generate_labels_trigger_joint.gen_configs import (
    CLUSTER_ROOT,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MILESTONE,
    draw_rng_seed,
    wandb_block,
    WANDB_ENABLED,
    WANDB_PROJECT,
    WANDB_MODE,
    WANDB_ENTITY,
)
from modules.generate_labels_trigger_joint import gen_configs_paper_main_campaign as main_campaign
from modules.generate_labels_trigger_joint.gen_configs_paper_main_campaign import (
    SOURCE_LABEL,
    TARGET_LABEL,
    SELECT_FLIPS_TEMPLATE,
    BOOTSTRAP_BUDGET,
    GEN_EXPERT_RETRAIN_EPOCHS,
    GEN_EXPERT_RETRAIN_CHECKPOINT_ITERS,
    DEPLOY_BUDGETS,
    DEPLOY_SINGLE_USER_NUM_POISONED,
    DEPLOY_SINGLE_USER_NUM_HONESTS,
    DEPLOY_SINGLE_USER_AGG_METHOD,
    DEPLOY_FEDERATED_NUM_POISONED,
    DEPLOY_FEDERATED_NUM_HONESTS,
    DEPLOY_AGG_METHODS_FEDERATED,
    FEDERATED_TAG,
    STEALTH_BASE,
    cell_name,
    joint_trigger_path,
    _train_user_configs,
)

MODULE_NAME = "generate_labels"
TAGS = main_campaign.list_tags()
MODEL_FLAGS = main_campaign.MODEL_FLAGS
DATASETS = main_campaign.DATASETS
SEEDS = main_campaign.SEEDS

EXP_BASE = Path("experiments/federated_experiments/flip_frozen_trigger_paper_main_campaign")

# Same expert schedule as the joint campaign's bootstrap/retraining experts, and the same
# label-optimization budget (iterations, one-hot temperature, checkpoint window).
TRAIN_EXPERT_TEMPLATE = """[train_expert]
output_dir = "{expert_dir}/0/"
model = "{model_flag}"
dataset = "{dataset}"
trainer = "sgd"
source_label = {source_label}
target_label = {target_label}
poisoner = "optimized"
delta = "{trigger_path}"
budget = {budget}
checkpoint_iters = {checkpoint_iters}
epochs = {epochs}
optim_kwargs = {{lr = {lr}, momentum = 0.9, nesterov = true, weight_decay = {wd}}}
scheduler_kwargs = {{milestones = {milestones}, gamma = 0.1}}
seed = {rng_seed}
{wandb_block_train_expert}"""

GEN_LABELS_TEMPLATE = """[generate_labels]
input_pths = "{expert_dir}/{{}}/model_{{}}_{{}}.pth"
opt_pths = "{expert_dir}/{{}}/model_{{}}_{{}}_opt.pth"
output_dir = "{module_dir}/labels/"
expert_model = "{model_flag}"
dataset = "{dataset}"
trainer = "sgd"
source_label = {source_label}
target_label = {target_label}
poisoner = "optimized"
delta = "{trigger_path}"
train_pct = 1.0

[generate_labels.expert_config]
experts = 1
min = 0
max = 20
trajectories = [50, 100, 150, 200]

[generate_labels.attack_config]
iterations = {iterations}
one_hot_temp = 5
"""


def expert_output_dir(model_flag, dataset, tag, seed):
    return f"{CLUSTER_ROOT}/out/checkpoints/flip_frozen_trigger/{model_flag}_{dataset}_{tag}/seed{seed}"


def expert_cell_name(model_flag, dataset, tag, seed):
    return f"train_expert/{model_flag}_{dataset}_{tag}/seed{seed}"


def generate_cell(model_flag, dataset, tag, seed, dry_run=False):
    lr = LEARNING_RATE.get(model_flag, 0.1)
    wd = WEIGHT_DECAY.get(model_flag, 2e-4)
    milestones = MILESTONE.get(model_flag, [75, 125])
    trigger_path = joint_trigger_path(model_flag, dataset, tag, seed)
    expert_dir = expert_output_dir(model_flag, dataset, tag, seed)
    cell_dir = EXP_BASE / cell_name(model_flag, dataset, tag, seed)
    module_dir = cell_dir / "gen_labels"

    configs = {
        EXP_BASE / expert_cell_name(model_flag, dataset, tag, seed) / "config.toml": TRAIN_EXPERT_TEMPLATE.format(
            expert_dir=expert_dir,
            model_flag=model_flag,
            dataset=dataset,
            source_label=SOURCE_LABEL,
            target_label=TARGET_LABEL,
            trigger_path=trigger_path,
            budget=BOOTSTRAP_BUDGET[dataset],
            checkpoint_iters=GEN_EXPERT_RETRAIN_CHECKPOINT_ITERS,
            epochs=GEN_EXPERT_RETRAIN_EPOCHS,
            lr=lr,
            wd=wd,
            milestones=milestones,
            rng_seed=draw_rng_seed(model_flag, ("flip_frozen_trigger", dataset, tag, seed)),
            wandb_block_train_expert=wandb_block(
                "train_expert", f"flip_frozen_trigger/train_expert/{model_flag}/{dataset}/{tag}/seed{seed}",
                enabled=WANDB_ENABLED, project=WANDB_PROJECT,
                mode=WANDB_MODE, entity=WANDB_ENTITY, group="flip_frozen_trigger",
            ),
        ),
        module_dir / "config.toml": GEN_LABELS_TEMPLATE.format(
            expert_dir=expert_dir,
            module_dir=module_dir,
            model_flag=model_flag,
            dataset=dataset,
            source_label=SOURCE_LABEL,
            target_label=TARGET_LABEL,
            trigger_path=trigger_path,
            iterations=STEALTH_BASE["iterations"],
        ),
    }

    for branch_dir, num_honests, num_poisoned, agg_methods in (
        (cell_dir, DEPLOY_SINGLE_USER_NUM_HONESTS, DEPLOY_SINGLE_USER_NUM_POISONED,
         [DEPLOY_SINGLE_USER_AGG_METHOD]),
        (cell_dir / FEDERATED_TAG, DEPLOY_FEDERATED_NUM_HONESTS, DEPLOY_FEDERATED_NUM_POISONED,
         DEPLOY_AGG_METHODS_FEDERATED),
    ):
        flips_dir = branch_dir / "select_flips"
        configs[flips_dir / "config.toml"] = SELECT_FLIPS_TEMPLATE.format(
            budgets=DEPLOY_BUDGETS,
            module_dir=module_dir,
            flips_dir=flips_dir,
            deploy_num_honests=num_honests,
            deploy_num_poisoned=num_poisoned,
        )
        configs.update(_train_user_configs(
            base_dir=branch_dir, budgets=DEPLOY_BUDGETS,
            num_honests=num_honests, num_poisoned=num_poisoned,
            agg_methods=agg_methods, flips_dir=flips_dir,
            model_flag=model_flag, dataset=dataset, trigger_path=trigger_path,
            lr=lr, wd=wd, milestones=milestones,
        ))

    for path, content in configs.items():
        assert "out/checkpoints" not in str(path), f"Refusing to write under out/checkpoints/: {path}"
        if not dry_run:
            write_config(path, content)
            validate_config(path)
    return list(configs), trigger_path


def generate_all(model_flags, datasets, tags, seeds, dry_run=False):
    paths, missing_triggers = [], []
    for model_flag in model_flags:
        for dataset in datasets:
            for tag in tags:
                for seed in seeds:
                    cell_paths, trigger_path = generate_cell(model_flag, dataset, tag, seed, dry_run=dry_run)
                    paths += cell_paths
                    if not trigger_path.exists():
                        missing_triggers.append(trigger_path)
    return paths, missing_triggers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--models", nargs="+", default=MODEL_FLAGS, choices=MODEL_FLAGS)
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--tags", nargs="+", default=TAGS, choices=TAGS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    paths, missing = generate_all(args.models, args.datasets, args.tags, args.seeds, dry_run=args.dry_run)
    verb = "would be written" if args.dry_run else "written and schema-validated"
    print(f"\nflip_frozen_trigger: {len(paths)} config files {verb} under {EXP_BASE}.")
    if missing:
        print(
            f"[NOTE] {len(missing)} joint trigger(s) not on disk yet (e.g. {missing[0]}) -- run the "
            "paper_main_campaign GEN phase for those cells before launching them."
        )
