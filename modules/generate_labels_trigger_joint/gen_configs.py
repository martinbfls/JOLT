"""Config generator: train_expert -> generate_labels_trigger_joint ->
federated_select_flips -> federated_train_user. Refuses to generate a cell whose
delta_min_frac makes the magnitude floor infeasible under the given epsilon."""

import argparse
import math
import os
import random
import sys
from pathlib import Path

import toml

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
# Weight for L_joint (clean-task loss over the WHOLE local batch, additive to delta.grad the
# same way as L_bd); 0.0 (default) is a full no-op. See schemas/generate_labels_trigger_joint.toml.
LAMBDA_JOINT = 0.0
# Weight for a perceptual LPIPS penalty between clean and triggered images (works under both
# trigger_type='additive' and 'warp'); 0.0 (default) is a full no-op. Requires the `lpips`
# package when > 0 (lazily imported by run_module.py, not a hard dependency otherwise).
LAMBDA_LPIPS = 0.0

TRIGGER_CONSTRAINT = "penalty"
ALIGN_KAPPA = 0.3
LAMBDA_ALIGN = 0.0
LAMBDA_MAG = 0.3
DELTA_MIN_FRAC = 0.00

# init_delta (modules/optimizing_trigger/utils.py): "stripe" (deterministic) or
# "random" (unseeded global torch RNG).
INIT = "stripe"

# trigger_type (run_module.py): "additive" (default, optimizes the pixel delta) or "warp"
# (WaNet-style learned control-grid warp, see schemas/generate_labels_trigger_joint.toml).
# The warp_*/lr_warp knobs below are only used when TRIGGER_TYPE == "warp".
TRIGGER_TYPE = "additive"
WARP_K = 4
WARP_S = 1.0
LR_WARP = 0.1

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
output_dir = "{cluster_root}/out/checkpoints/{ckpt_subdir}/seed{seed}/0/"
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
input_pths = "{cluster_root}/out/checkpoints/{ckpt_subdir}/seed{seed}/{{}}/model_{{}}_{{}}.pth"
opt_pths = "{cluster_root}/out/checkpoints/{ckpt_subdir}/seed{seed}/{{}}/model_{{}}_{{}}_opt.pth"
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
lambda_joint = {lambda_joint}
lambda_lpips = {lambda_lpips}
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

trigger_type = "{trigger_type}"
warp_k = {warp_k}
warp_s = {warp_s}
lr_warp = {lr_warp}

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
poisoner = "{poisoner}"
delta = "{trigger_path}"
budget = {budget}
num_honests = {num_honests}
num_poisoned = {num_poisoned}
agg_method = "{agg_method}"
optim_kwargs = {{lr = {lr}, momentum = 0.9, nesterov = true, weight_decay = {wd}}}
scheduler_kwargs = {{milestones = {milestones}, gamma = 0.1}}
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


def generate_federated_deploy(
    model_flag,
    dataset,
    seed,
    budgets,
    trigger_type,
    deploy_num_poisoned,
    deploy_num_honests,
    deploy_agg_method,
    gen_agg_method="mean",
    init=None,
    dry_run=False,
):
    """Re-deploys an ALREADY-GENERATED trigger (from a prior --single-cell/--minimal/--all run,
    GEN side fixed at NUM_POISONED/NUM_HONESTS, agg_method=gen_agg_method) under federated
    aggregation: writes a NEW federated_select_flips + federated_train_user (NOT a re-run of
    GEN/train_expert -- select_flips genuinely must be redone here, since its flip assignment
    depends on num_honests/num_poisoned, which differ between the single-user GEN deployment and
    this federated one). Returns (paths, trigger_path, trigger_exists)."""
    init = INIT if init is None else init
    if trigger_type not in ("additive", "warp"):
        raise ValueError(f"trigger_type must be 'additive' or 'warp', got {trigger_type!r}")
    poisoner = "warp" if trigger_type == "warp" else "optimized"

    lr = LEARNING_RATE.get(model_flag, 0.1)
    wd = WEIGHT_DECAY.get(model_flag, 2e-4)
    milestones = MILESTONE.get(model_flag, [75, 125])

    cell_dir = EXP_BASE / cell_name(model_flag, dataset, gen_agg_method, seed)
    module_dir = cell_dir / "gen_labels_trigger_joint"
    trig_path = trigger_output_path(module_dir, model_flag, dataset, NUM_POISONED, NUM_HONESTS, init)
    trigger_exists = trig_path.exists()

    deploy_dir = cell_dir / f"{deploy_agg_method}_{deploy_num_poisoned}vs{deploy_num_honests}"
    flips_dir = deploy_dir / "select_flips"

    configs = {
        flips_dir / "config.toml": SELECT_FLIPS_TEMPLATE.format(
            budgets=budgets,
            module_dir=module_dir,
            flips_dir=flips_dir,
            num_honests=deploy_num_honests,
            num_poisoned=deploy_num_poisoned,
        ),
    }
    for budget in budgets:
        train_user_dir = deploy_dir / f"train_user_{budget}"
        configs[train_user_dir / "config.toml"] = TRAIN_USER_TEMPLATE.format(
            flips_dir=flips_dir,
            train_user_dir=train_user_dir,
            model_flag=model_flag,
            dataset=dataset,
            source_label=SOURCE_LABEL,
            target_label=TARGET_LABEL,
            trigger_path=trig_path,
            poisoner=poisoner,
            budget=budget,
            num_honests=deploy_num_honests,
            num_poisoned=deploy_num_poisoned,
            agg_method=deploy_agg_method,
            lr=lr,
            wd=wd,
            milestones=milestones,
            wandb_block_train_user=wandb_block(
                "federated_train_user",
                f"train_user/{model_flag}/{dataset}/{deploy_agg_method}/{budget}/seed{seed}",
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

    return paths, trig_path, trigger_exists


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
    trigger_type=None,
    warp_k=None,
    warp_s=None,
    lr_warp=None,
    checkpoint_tag=None,
    lambda_bd=None,
    lambda_joint=None,
    lambda_lpips=None,
):
    delta_min_frac = DELTA_MIN_FRAC if delta_min_frac is None else delta_min_frac
    init = INIT if init is None else init
    detach_param_dist = DETACH_PARAM_DIST if detach_param_dist is None else detach_param_dist
    trigger_type = TRIGGER_TYPE if trigger_type is None else trigger_type
    lambda_bd = LAMBDA_BD if lambda_bd is None else lambda_bd
    lambda_joint = LAMBDA_JOINT if lambda_joint is None else lambda_joint
    lambda_lpips = LAMBDA_LPIPS if lambda_lpips is None else lambda_lpips
    warp_k = WARP_K if warp_k is None else warp_k
    warp_s = WARP_S if warp_s is None else warp_s
    lr_warp = LR_WARP if lr_warp is None else lr_warp
    if trigger_type not in ("additive", "warp"):
        raise ValueError(f"trigger_type must be 'additive' or 'warp', got {trigger_type!r}")
    poisoner = "warp" if trigger_type == "warp" else "optimized"
    ckpt_subdir = f"{model_flag}_1xs" if checkpoint_tag is None else f"{model_flag}_1xs_{checkpoint_tag}"

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
    train_expert_dir = EXP_BASE / f"train_expert/{ckpt_subdir}/seed{seed}"
    module_dir = cell_dir / "gen_labels_trigger_joint"
    flips_dir = cell_dir / "select_flips"

    configs = {
        train_expert_dir / "config.toml": TRAIN_EXPERT_TEMPLATE.format(
            cluster_root=CLUSTER_ROOT,
            ckpt_subdir=ckpt_subdir,
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
            ckpt_subdir=ckpt_subdir,
            model_flag=model_flag,
            dataset=dataset,
            seed=seed,
            rng_seed=rng_seed,
            cell_dir=module_dir,
            source_label=SOURCE_LABEL,
            target_label=TARGET_LABEL,
            epsilon=EPSILON,
            lr_delta=LR_DELTA,
            lambda_bd=lambda_bd,
            lambda_delta=LAMBDA_DELTA,
            lambda_joint=lambda_joint,
            lambda_lpips=lambda_lpips,
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
            trigger_type=trigger_type,
            warp_k=warp_k,
            warp_s=warp_s,
            lr_warp=lr_warp,
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
            poisoner=poisoner,
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


def generate_single_cell(
    dry_run=True, budgets=None, model_flag=None, dataset=None, agg_method=None, seed=None,
    checkpoint_tag=None,
):
    """Exactly one cell (default: first model/dataset/agg_method/seed) at default regularization.
    budgets=None (default): just BUDGETS[:1] (one train_user config). Pass an explicit list
    (e.g. via --budgets) to get a train_user config per budget in that same single cell."""
    model_flag = MODEL_FLAGS[0] if model_flag is None else model_flag
    dataset = DATASETS[0] if dataset is None else dataset
    agg_method = AGG_METHODS[0] if agg_method is None else agg_method
    seed = SEEDS[0] if seed is None else seed
    paths, reason = generate_cell(
        model_flag,
        dataset,
        agg_method,
        seed,
        BUDGETS[:1] if budgets is None else budgets,
        dry_run=dry_run,
        checkpoint_tag=checkpoint_tag,
    )
    refused = [(model_flag, dataset, agg_method, seed, reason)] if reason else []
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
    parser.add_argument(
        "--trigger-type",
        choices=["additive", "warp"],
        default=None,
        help="override TRIGGER_TYPE: 'warp' uses a learned WaNet-style warp trigger instead of the additive delta",
    )
    parser.add_argument("--warp-k", type=int, default=None, help="override WARP_K (warp control-grid side)")
    parser.add_argument("--warp-s", type=float, default=None, help="override WARP_S (max warp displacement, px)")
    parser.add_argument("--lr-warp", type=float, default=None, help="override LR_WARP")
    parser.add_argument("--lambda-bd", type=float, default=None, help="override LAMBDA_BD (default 2.0)")
    parser.add_argument(
        "--lambda-joint",
        type=float,
        default=None,
        help="override LAMBDA_JOINT (default 0.0/off): weight for L_joint, the clean-task-loss "
        "regularization term (whole local batch, not just poisoned rows) -- see "
        "schemas/generate_labels_trigger_joint.toml's lambda_joint entry.",
    )
    parser.add_argument(
        "--lambda-lpips",
        type=float,
        default=None,
        help="override LAMBDA_LPIPS (default 0.0/off): weight for a perceptual LPIPS penalty "
        "between clean and triggered images. Works under both --trigger-type additive and warp. "
        "Requires the `lpips` package installed in the conda env when > 0.",
    )
    parser.add_argument(
        "--exp-base-tag",
        default=None,
        help="suffix appended to EXP_BASE's dir name (e.g. 'warp'), so a side-by-side run "
        "(e.g. additive vs. warp) writes to a sibling experiments/ tree instead of colliding "
        "on the same cell paths",
    )
    parser.add_argument(
        "--budgets",
        default=None,
        help="comma/space-separated list of flip budgets, overriding BUDGETS (e.g. '1500' or "
        "'150,500,1500'). With --single-cell this produces one train_user config per budget "
        "in that one cell (default without this flag: just BUDGETS[:1]).",
    )
    parser.add_argument("--model-flag", default=None, help="override MODEL_FLAGS[0] for --single-cell")
    parser.add_argument("--dataset", default=None, help="override DATASETS[0] for --single-cell")
    parser.add_argument("--agg-method", default=None, help="override AGG_METHODS[0] for --single-cell")
    parser.add_argument("--seed", type=int, default=None, help="override SEEDS[0] for --single-cell")
    parser.add_argument(
        "--checkpoint-tag",
        default=None,
        help="suffix for the train_expert checkpoint subdir (out/checkpoints/{model_flag}_1xs_<tag>/...), "
        "so a side-by-side run doesn't race on/clobber the same checkpoint files. Defaults to "
        "--exp-base-tag when that is set (so e.g. --exp-base-tag warp also isolates checkpoints "
        "unless you pass --checkpoint-tag explicitly); pass --checkpoint-tag '' to force sharing.",
    )
    parser.add_argument(
        "--deploy-federated",
        action="store_true",
        help="re-deploy an ALREADY-GENERATED single-cell trigger (--model-flag/--dataset/--seed) "
        "under federated aggregation: writes only a NEW federated_select_flips + "
        "federated_train_user (the GEN/train_expert phases are NOT re-run -- the trigger must "
        "already exist on disk). Combine with --deploy-num-poisoned/--deploy-num-honests/"
        "--deploy-agg-method and --budgets/--trigger-type/--exp-base-tag.",
    )
    parser.add_argument("--deploy-num-poisoned", type=int, default=3, help="with --deploy-federated")
    parser.add_argument("--deploy-num-honests", type=int, default=7, help="with --deploy-federated")
    parser.add_argument(
        "--deploy-agg-method",
        default="multikrum",
        choices=["mean", "median", "trmean", "krum", "multikrum", "krum_full", "multikrum_full", "geomed"],
        help="with --deploy-federated",
    )
    parser.add_argument(
        "--gen-agg-method",
        default="mean",
        help="with --deploy-federated: agg_method the GEN cell was originally generated under "
        "(gen_configs.py's own single-cell default is 'mean') -- used only to locate the "
        "already-generated trigger/cell directory, not written into the new federated configs.",
    )
    args = parser.parse_args()

    if args.exp_base_tag is not None:
        EXP_BASE = EXP_BASE.parent / f"{EXP_BASE.name}_{args.exp_base_tag}"
    checkpoint_tag = args.checkpoint_tag
    if checkpoint_tag is None:
        checkpoint_tag = args.exp_base_tag
    elif checkpoint_tag == "":
        checkpoint_tag = None
    budgets_override = None
    if args.budgets is not None:
        budgets_override = [int(b) for b in args.budgets.replace(",", " ").split()]
        BUDGETS = budgets_override
    if args.trigger_type is not None:
        TRIGGER_TYPE = args.trigger_type
    if args.warp_k is not None:
        WARP_K = args.warp_k
    if args.warp_s is not None:
        WARP_S = args.warp_s
    if args.lr_warp is not None:
        LR_WARP = args.lr_warp
    if args.lambda_bd is not None:
        LAMBDA_BD = args.lambda_bd
    if args.lambda_joint is not None:
        LAMBDA_JOINT = args.lambda_joint
    if args.lambda_lpips is not None:
        LAMBDA_LPIPS = args.lambda_lpips

    if args.print_grid:
        import json

        print(json.dumps(list_grid(), indent=2))
        raise SystemExit(0)

    if args.deploy_federated:
        assert args.budgets is not None, "--deploy-federated requires --budgets"
        paths, trig_path, trigger_exists = generate_federated_deploy(
            model_flag=args.model_flag or MODEL_FLAGS[0],
            dataset=args.dataset or DATASETS[0],
            seed=args.seed if args.seed is not None else SEEDS[0],
            budgets=budgets_override,
            trigger_type=TRIGGER_TYPE,
            deploy_num_poisoned=args.deploy_num_poisoned,
            deploy_num_honests=args.deploy_num_honests,
            deploy_agg_method=args.deploy_agg_method,
            gen_agg_method=args.gen_agg_method,
            dry_run=args.dry_run,
        )
        if not trigger_exists:
            print(
                f"[ABORT] trigger not found at {trig_path} -- run the GEN phase for this cell "
                "first (e.g. --single-cell with the same --model-flag/--dataset/--seed/--trigger-type/"
                "--exp-base-tag) before --deploy-federated.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        label = "[DRY RUN]" if args.dry_run else ""
        print(f"\n{label} generate_labels_trigger_joint (federated deploy): {len(paths)} config files "
              f"{'would be ' if args.dry_run else ''}written, trigger={trig_path}")
        for p in paths:
            print(f"  {p}")
        raise SystemExit(0)

    assert not (args.minimal and args.single_cell), "pass at most one of --minimal/--single-cell"

    if args.single_cell:
        paths, refused = generate_single_cell(
            dry_run=args.dry_run,
            budgets=budgets_override,
            model_flag=args.model_flag,
            dataset=args.dataset,
            agg_method=args.agg_method,
            seed=args.seed,
            checkpoint_tag=checkpoint_tag,
        )
    elif args.minimal:
        paths, refused = generate_minimal_campaign(dry_run=args.dry_run)
    else:
        paths, refused = generate_all_configs(dry_run=args.dry_run)

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
