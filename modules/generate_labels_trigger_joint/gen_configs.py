"""Config generator: train_expert -> generate_labels_trigger_joint ->
federated_select_flips -> federated_train_user. Refuses to generate a cell whose
delta_min_frac makes the magnitude floor infeasible under the given epsilon."""

import argparse
import math
import os
import random
from pathlib import Path

import toml
import torch

from modules.base_utils.gen_configs import wandb_block
from modules.base_utils.config_validation import (
    write_config,
    validate_config_file as validate_config,
)

# Real RNG seed written into train_expert's/this module's `seed` field, distinct from the
# `seed` loop variable used for cell/path naming; cached per (model_flag, path_seed) so every
# cell sharing a train_expert checkpoint writes the same value (required for
# expert_retrain_interval's seed-matching).
_RNG_SEED_CACHE = {}


def draw_rng_seed(model_flag, path_seed):
    key = (model_flag, path_seed)
    if key not in _RNG_SEED_CACHE:
        _RNG_SEED_CACHE[key] = random.randint(0, 2**31 - 1)
    return _RNG_SEED_CACHE[key]


# Weights & Biases mirroring (see modules/base_utils/experiment_tracker.py); off by default.
WANDB_ENABLED = False
WANDB_PROJECT = "flip"
WANDB_ENTITY = None
WANDB_MODE = "online"

# Sweep axes -- edit these for a real campaign.
NUM_POISONED = 1
NUM_HONESTS = 0
SEEDS = [0]
BUDGETS = [150, 300, 500, 1000, 2000, 2500, 5000]
AGG_METHODS = ["mean"]
DATASETS = ["cifar"]
MODEL_FLAGS = ["r32p"]
SOURCE_LABEL = 9
TARGET_LABEL = 4

# This module's own default; a cross-module comparison must set both generators the same way.
CHECKPOINT_SAMPLING = "biased"
INDIRECT_MODULE_CHECKPOINT_SAMPLING = "uniform"

ALPHA_CKPT = 0.01
TRAIN_PCT = 1.0
EPOCHS_EXPERT = 20
CHECKPOINT_ITERS = 50
N_ITERATIONS = 25

# Live expert retraining: every EXPERT_RETRAIN_INTERVAL outer iterations, retrain a fresh
# expert against the current trigger for EXPERT_RETRAIN_EPOCHS epochs. 0 disables entirely.
EXPERT_RETRAIN_INTERVAL = 1
EXPERT_RETRAIN_EPOCHS = EPOCHS_EXPERT
EXPERT_RETRAIN_CHECKPOINT_ITERS = CHECKPOINT_ITERS

# Every batch-step averages the matching/gradient-mismatch terms over this many distinct
# checkpoints instead of one; requires pool_size >= N_CHECKPOINTS_PER_STEP.
N_CHECKPOINTS_PER_STEP = 5

# Trigger regularization knobs, tuned by a prior one-at-a-time sweep (see run_module.py for
# term semantics); `single_cell` mode fixes all of these and sweeps only SEEDS/BUDGETS/AGG_METHODS.
EPSILON = 1.0
LR_DELTA = 1e-2
LAMBDA_BD = 2.0
GAMMA_STEALTH = 1.0
LAMBDA_DELTA = 1.0

TRIGGER_CONSTRAINT = "penalty"
ALIGN_KAPPA = 0.3
LAMBDA_ALIGN = 0.0
LAMBDA_MAG = 0.3
DELTA_MIN_FRAC = 0.00

# init_delta (modules/optimizing_trigger/utils.py): "stripe" (deterministic) or
# "random" (unseeded global torch RNG).
INIT = "stripe"

LAMBDA_GRADMATCH = 1.0
GRADMATCH_EPS = 1e-8
GRADMATCH_METRIC = "relerr"  # or "cosine"

DETACH_PARAM_DIST = False  # if true, param_dist is detached in mtt_term_k's denominator

LEARNING_RATE = {"r32p": 0.1, "r18": 0.1, "vgg": 0.01}
WEIGHT_DECAY = {"r32p": 2e-4, "r18": 2e-4, "vgg": 2e-4}
MILESTONE = {"r32p": [75, 125], "r18": [75, 125], "vgg": [125]}

CLUSTER_ROOT = "."

EXP_BASE = Path("experiments/federated_experiments/threat_model_direct_trigger_joint").resolve()

MODULE_NAME = "generate_labels_trigger_joint"

# Trigger image shape used for the delta_min feasibility check -- only the small (32x32)
# datasets are supported here (needs_big_ims models/datasets would need a different shape;
# not in the default sweep, see MODEL_FLAGS/DATASETS above).
_TRIGGER_SHAPE = {"cifar": (3, 32, 32), "cifar_100": (3, 32, 32), "svhn": (3, 32, 32)}


def check_delta_min_feasible(dataset, epsilon, delta_min_frac, init="stripe"):
    """Returns (feasible, delta_min, max_reachable); delta_min_frac<=1.0 is always feasible. `init` is unused."""
    if dataset not in _TRIGGER_SHAPE:
        raise ValueError(
            f"check_delta_min_feasible: no known trigger shape for dataset={dataset!r} -- "
            f"add it to _TRIGGER_SHAPE (only {list(_TRIGGER_SHAPE)} are supported)."
        )
    shape = _TRIGGER_SHAPE[dataset]
    numel = shape[0] * shape[1] * shape[2]
    max_reachable = epsilon * math.sqrt(numel)
    delta_min = delta_min_frac * max_reachable
    return delta_min <= max_reachable, delta_min, max_reachable


TRAIN_EXPERT_TEMPLATE = """[train_expert]
output_dir = "{cluster_root}/out/checkpoints/{model_flag}_1xs/seed{seed}/0/"
model = "{model_flag}"
dataset = "{dataset}"
trainer = "sgd"
source_label = {source_label}
target_label = {target_label}
poisoner = "1xs"
checkpoint_iters = {checkpoint_iters}
epochs = {epochs}
optim_kwargs = {{lr = {lr}, momentum = 0.9, nesterov = true, weight_decay = {wd}}}
scheduler_kwargs = {{milestones = {milestones}, gamma = 0.1}}
seed = {rng_seed}
{wandb_block_train_expert}"""

JOINT_TRIGGER_TEMPLATE = """[generate_labels_trigger_joint]
input_pths = "{cluster_root}/out/checkpoints/{model_flag}_1xs/seed{seed}/{{}}/model_{{}}_{{}}.pth"
opt_pths = "{cluster_root}/out/checkpoints/{model_flag}_1xs/seed{seed}/{{}}/model_{{}}_{{}}_opt.pth"
output_dir = "{cell_dir}/labels/"
output_dir_trigger = "{cell_dir}/trigger"
expert_model = "{model_flag}"
dataset = "{dataset}"
source_label = {source_label}
target_label = {target_label}

epsilon = {epsilon}
lr_delta = {lr_delta}
lambda_bd = {lambda_bd}
lambda_delta = {lambda_delta}
lambda_gradmatch = {lambda_gradmatch}
gradmatch_eps = {gradmatch_eps}
gradmatch_metric = "{gradmatch_metric}"
detach_param_dist = {detach_param_dist}

train_pct = {train_pct}
num_honests = {num_honests}
num_poisoned = {num_poisoned}
agg_method = "{agg_method}"
attack = "backdoor"
gamma_stealth = {gamma_stealth}
checkpoint_sampling = "{checkpoint_sampling}"
alpha_ckpt = {alpha_ckpt}

init = "{init}"

trigger_constraint = "{trigger_constraint}"
align_kappa = {align_kappa}
lambda_align = {lambda_align}
lambda_mag = {lambda_mag}
delta_min_frac = {delta_min_frac}

expert_retrain_interval = {expert_retrain_interval}
expert_retrain_epochs = {expert_retrain_epochs}
expert_retrain_checkpoint_iters = {expert_retrain_checkpoint_iters}
expert_retrain_optim_kwargs = {{lr = {lr}, momentum = 0.9, nesterov = true, weight_decay = {wd}}}
expert_retrain_scheduler_kwargs = {{milestones = {milestones}, gamma = 0.1}}
seed = {rng_seed}

n_checkpoints_per_step = {n_checkpoints_per_step}
{wandb_block_module}
[generate_labels_trigger_joint.expert_config]
experts = 1
min = 0
max = 20
trajectories = [50, 100, 150, 200]

[generate_labels_trigger_joint.attack_config]
iterations = {n_iterations}
one_hot_temp = 5
"""

SELECT_FLIPS_TEMPLATE = """[federated_select_flips]
budgets = {budgets}
input_label_glob = "{module_dir}/labels/labels.npy"
true_labels = "{module_dir}/labels/true.npy"
output_dir = "{flips_dir}"
num_honests = {num_honests}
num_poisoned = {num_poisoned}
"""

TRAIN_USER_TEMPLATE = """[federated_train_user]
input_labels = "{flips_dir}/"
output_dir = "{train_user_dir}"
user_model = "{model_flag}"
trainer = "sgd"
dataset = "{dataset}"
source_label = {source_label}
target_label = {target_label}
poisoner = "optimized"
delta = "{trigger_path}"
budget = {budget}
num_honests = {num_honests}
num_poisoned = {num_poisoned}
agg_method = "{agg_method}"
optim_kwargs = {{lr = {lr}, momentum = 0.9, nesterov = true, weight_decay = {wd}}}
schedule_kwargs = {{milestones = {milestones}, gamma = 0.1}}
{wandb_block_train_user}"""


def trigger_output_path(module_dir, model_flag, dataset, num_poisoned, num_honests, init):
    """Path to the .pt trigger run_module.py saves; train_user reads it back to poison with poisoner="optimized"."""
    return (
        module_dir
        / "trigger"
        / f"opt_trig_direct_joint_{init}_{model_flag}_{dataset}_{num_poisoned}vs{num_honests}.pt"
    )


def cell_name(model_flag, dataset, agg_method, seed):
    return f"{model_flag}/{dataset}/{NUM_POISONED}vs{NUM_HONESTS}/{agg_method}/seed{seed}"


def generate_cell(
    model_flag,
    dataset,
    agg_method,
    seed,
    budgets,
    dry_run=False,
    delta_min_frac=None,
    init=None,
    detach_param_dist=None,
):
    delta_min_frac = DELTA_MIN_FRAC if delta_min_frac is None else delta_min_frac
    init = INIT if init is None else init
    detach_param_dist = DETACH_PARAM_DIST if detach_param_dist is None else detach_param_dist

    if CHECKPOINT_SAMPLING != INDIRECT_MODULE_CHECKPOINT_SAMPLING:
        print(
            f"WARNING: checkpoint_sampling={CHECKPOINT_SAMPLING!r} differs from the reference "
            f"value {INDIRECT_MODULE_CHECKPOINT_SAMPLING!r} used for cross-run comparability."
        )

    feasible, delta_min, max_reachable = check_delta_min_feasible(
        dataset,
        EPSILON,
        delta_min_frac,
        init=init,
    )
    if not feasible:
        reason = (
            f"delta_min_frac={delta_min_frac} -> delta_min={delta_min:.4f} > "
            f"epsilon*sqrt(numel)={max_reachable:.4f} at epsilon={EPSILON} -- structurally "
            "unreachable post-clamp; refusing to generate this cell, it would measure nothing."
        )
        print(f"REFUSED [{model_flag}/{dataset}/{agg_method}/seed{seed}]: {reason}")
        return [], reason

    lr = LEARNING_RATE.get(model_flag, 0.1)
    wd = WEIGHT_DECAY.get(model_flag, 2e-4)
    milestones = MILESTONE.get(model_flag, [75, 125])
    rng_seed = draw_rng_seed(model_flag, seed)

    cell_dir = EXP_BASE / cell_name(model_flag, dataset, agg_method, seed)
    train_expert_dir = EXP_BASE / f"train_expert/{model_flag}_1xs/seed{seed}"
    module_dir = cell_dir / "gen_labels_trigger_joint"
    flips_dir = cell_dir / "select_flips"

    configs = {
        train_expert_dir / "config.toml": TRAIN_EXPERT_TEMPLATE.format(
            cluster_root=CLUSTER_ROOT,
            model_flag=model_flag,
            dataset=dataset,
            seed=seed,
            rng_seed=rng_seed,
            source_label=SOURCE_LABEL,
            target_label=TARGET_LABEL,
            checkpoint_iters=CHECKPOINT_ITERS,
            epochs=EPOCHS_EXPERT,
            lr=lr,
            wd=wd,
            milestones=milestones,
            wandb_block_train_expert=wandb_block(
                "train_expert",
                f"train_expert/{model_flag}/{dataset}/seed{seed}",
                enabled=WANDB_ENABLED,
                project=WANDB_PROJECT,
                mode=WANDB_MODE,
                entity=WANDB_ENTITY,
                group=model_flag,
            ),
        ),
        module_dir / "config.toml": JOINT_TRIGGER_TEMPLATE.format(
            cluster_root=CLUSTER_ROOT,
            model_flag=model_flag,
            dataset=dataset,
            seed=seed,
            rng_seed=rng_seed,
            cell_dir=module_dir,
            source_label=SOURCE_LABEL,
            target_label=TARGET_LABEL,
            epsilon=EPSILON,
            lr_delta=LR_DELTA,
            lambda_bd=LAMBDA_BD,
            lambda_delta=LAMBDA_DELTA,
            lambda_gradmatch=LAMBDA_GRADMATCH,
            gradmatch_eps=GRADMATCH_EPS,
            gradmatch_metric=GRADMATCH_METRIC,
            detach_param_dist=str(detach_param_dist).lower(),
            train_pct=TRAIN_PCT,
            num_honests=NUM_HONESTS,
            num_poisoned=NUM_POISONED,
            agg_method=agg_method,
            gamma_stealth=GAMMA_STEALTH,
            checkpoint_sampling=CHECKPOINT_SAMPLING,
            alpha_ckpt=ALPHA_CKPT,
            n_iterations=N_ITERATIONS,
            trigger_constraint=TRIGGER_CONSTRAINT,
            align_kappa=ALIGN_KAPPA,
            lambda_align=LAMBDA_ALIGN,
            lambda_mag=LAMBDA_MAG,
            delta_min_frac=delta_min_frac,
            init=init,
            expert_retrain_interval=EXPERT_RETRAIN_INTERVAL,
            expert_retrain_epochs=EXPERT_RETRAIN_EPOCHS,
            expert_retrain_checkpoint_iters=EXPERT_RETRAIN_CHECKPOINT_ITERS,
            n_checkpoints_per_step=N_CHECKPOINTS_PER_STEP,
            lr=lr,
            wd=wd,
            milestones=milestones,
            wandb_block_module=wandb_block(
                MODULE_NAME,
                f"{MODULE_NAME}/{model_flag}/{dataset}/{agg_method}/seed{seed}",
                enabled=WANDB_ENABLED,
                project=WANDB_PROJECT,
                mode=WANDB_MODE,
                entity=WANDB_ENTITY,
                group=model_flag,
            ),
        ),
        flips_dir / "config.toml": SELECT_FLIPS_TEMPLATE.format(
            budgets=budgets,
            module_dir=module_dir,
            flips_dir=flips_dir,
            num_honests=NUM_HONESTS,
            num_poisoned=NUM_POISONED,
        ),
    }
    trigger_path = trigger_output_path(
        module_dir, model_flag, dataset, NUM_POISONED, NUM_HONESTS, init,
    )
    for budget in budgets:
        train_user_dir = cell_dir / f"train_user_{budget}"
        configs[train_user_dir / "config.toml"] = TRAIN_USER_TEMPLATE.format(
            flips_dir=flips_dir,
            train_user_dir=train_user_dir,
            model_flag=model_flag,
            dataset=dataset,
            source_label=SOURCE_LABEL,
            target_label=TARGET_LABEL,
            trigger_path=trigger_path,
            budget=budget,
            num_honests=NUM_HONESTS,
            num_poisoned=NUM_POISONED,
            agg_method=agg_method,
            lr=lr,
            wd=wd,
            milestones=milestones,
            wandb_block_train_user=wandb_block(
                "federated_train_user",
                f"train_user/{model_flag}/{dataset}/{agg_method}/{budget}/seed{seed}",
                enabled=WANDB_ENABLED,
                project=WANDB_PROJECT,
                mode=WANDB_MODE,
                entity=WANDB_ENTITY,
                group=model_flag,
            ),
        )

    paths = []
    for path, content in configs.items():
        assert "out/checkpoints" not in str(path), f"Refusing to write under out/checkpoints/: {path}"
        if dry_run:
            paths.append(path)
            continue
        write_config(path, content)
        validate_config(path)
        paths.append(path)

    return paths, None


def generate_single_cell(dry_run=True):
    """Exactly one cell (first model/dataset/agg_method/seed, first budget) at default regularization."""
    paths, reason = generate_cell(
        MODEL_FLAGS[0],
        DATASETS[0],
        AGG_METHODS[0],
        SEEDS[0],
        BUDGETS[:1],
        dry_run=dry_run,
    )
    refused = [(MODEL_FLAGS[0], DATASETS[0], AGG_METHODS[0], SEEDS[0], reason)] if reason else []
    return paths, refused


def generate_minimal_campaign(dry_run=True):
    all_paths, refused = [], []
    for model_flag in MODEL_FLAGS[:1]:
        for dataset in DATASETS[:1]:
            for agg_method in AGG_METHODS[:2]:
                for seed in SEEDS[:3]:
                    paths, reason = generate_cell(
                        model_flag,
                        dataset,
                        agg_method,
                        seed,
                        BUDGETS[:3],
                        dry_run=dry_run,
                    )
                    all_paths += paths
                    if reason:
                        refused.append((model_flag, dataset, agg_method, seed, reason))
    return all_paths, refused


def generate_all_configs(dry_run=False):
    all_paths, refused = [], []
    for model_flag in MODEL_FLAGS:
        for dataset in DATASETS:
            for agg_method in AGG_METHODS:
                for seed in SEEDS:
                    paths, reason = generate_cell(
                        model_flag,
                        dataset,
                        agg_method,
                        seed,
                        BUDGETS,
                        dry_run=dry_run,
                    )
                    all_paths += paths
                    if reason:
                        refused.append((model_flag, dataset, agg_method, seed, reason))
    return all_paths, refused


def list_grid():
    """Enumerates this campaign's sweep cells (model/dataset/agg_method/seed) as plain dicts."""
    return [
        {"model_flag": model_flag, "dataset": dataset, "agg_method": agg_method, "seed": seed}
        for model_flag in MODEL_FLAGS
        for dataset in DATASETS
        for agg_method in AGG_METHODS
        for seed in SEEDS
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--minimal", action="store_true", help="minimal campaign only")
    parser.add_argument(
        "--single-cell",
        action="store_true",
        help="exactly one cell, REGULARIZATION_GRID fixed at defaults -- the minimal preliminary campaign",
    )
    parser.add_argument(
        "--print-grid",
        action="store_true",
        help="print this campaign's sweep cells as JSON and exit, without writing configs",
    )
    args = parser.parse_args()

    if args.print_grid:
        import json

        print(json.dumps(list_grid(), indent=2))
        raise SystemExit(0)

    assert not (args.minimal and args.single_cell), "pass at most one of --minimal/--single-cell"

    if args.single_cell:
        gen_fn = generate_single_cell
    elif args.minimal:
        gen_fn = generate_minimal_campaign
    else:
        gen_fn = generate_all_configs
    paths, refused = gen_fn(dry_run=args.dry_run)

    if args.dry_run:
        print(f"\n[DRY RUN] {MODULE_NAME}: {len(paths)} config files would be written.")
        for p in paths:
            print(f"  {p}")
    else:
        print(f"\n{MODULE_NAME}: {len(paths)} config files written and schema-validated.")

    if refused:
        print(f"\n{len(refused)} cell(s) REFUSED (delta_min infeasible):")
        for model_flag, dataset, agg_method, seed, reason in refused:
            print(f"  [{model_flag}/{dataset}/{agg_method}/seed{seed}] {reason}")
