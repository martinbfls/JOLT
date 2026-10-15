"""Generates the paper's main-results campaign: 3 finalist stealth configs x 2 models x 2 datasets x 10 seeds, each deployed single_user + federated_3vs7."""

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
    TRAIN_EXPERT_TEMPLATE,
    check_delta_min_feasible,
    draw_rng_seed,
    wandb_block,
    WANDB_ENABLED,
    WANDB_PROJECT,
    WANDB_MODE,
    WANDB_ENTITY,
)

# Generation-time config for the finalist trigger (fixed, not swept): 1-poisoned/0-honest/mean.
SOURCE_LABEL = 9
TARGET_LABEL = 4
GEN_INIT = "stripe"
GEN_TRIGGER_CONSTRAINT = "penalty"
GEN_N_CHECKPOINTS_PER_STEP = 5
GEN_EXPERT_RETRAIN_INTERVAL = 1
GEN_EXPERT_RETRAIN_EPOCHS = 20
GEN_EXPERT_RETRAIN_CHECKPOINT_ITERS = 50
GEN_NUM_HONESTS = 0
GEN_NUM_POISONED = 1
GEN_AGG_METHOD = "mean"

SELECT_FLIPS_TEMPLATE = """[federated_select_flips]
budgets = {budgets}
input_label_glob = "{module_dir}/labels/labels.npy"
true_labels = "{module_dir}/labels/true.npy"
output_dir = "{flips_dir}"
num_honests = {deploy_num_honests}
num_poisoned = {deploy_num_poisoned}
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
num_honests = {deploy_num_honests}
num_poisoned = {deploy_num_poisoned}
agg_method = "{agg_method}"
optim_kwargs = {{lr = {lr}, momentum = 0.9, nesterov = true, weight_decay = {wd}}}
scheduler_kwargs = {{milestones = {milestones}, gamma = 0.1}}
"""

JOINT_TRIGGER_TEMPLATE = """[generate_labels_trigger_joint]
input_pths = "{bootstrap_dir}/{{}}/model_{{}}_{{}}.pth"
opt_pths = "{bootstrap_dir}/{{}}/model_{{}}_{{}}_opt.pth"
output_dir = "{cell_dir}/labels/"
output_dir_trigger = "{cell_dir}/trigger"
expert_model = "{model_flag}"
dataset = "{dataset}"
source_label = {source_label}
target_label = {target_label}

init = "{init}"
epsilon = {epsilon}
lr_delta = {lr_delta}
lambda_bd = {lambda_bd}
lambda_penalty = {lambda_penalty}
lambda_delta = {lambda_delta}

num_honests = {gen_num_honests}
num_poisoned = {gen_num_poisoned}
agg_method = "{gen_agg_method}"

lambda_align = 0.0
lambda_mag = 0.0
lambda_tv = 0.0
lambda_lpips = 0.0
lambda_gradmatch = 0.0
lambda_match = 0.0
trigger_constraint = "{trigger_constraint}"
n_checkpoints_per_step = {n_checkpoints_per_step}

expert_retrain_interval = {expert_retrain_interval}
expert_retrain_epochs = {expert_retrain_epochs}
expert_retrain_checkpoint_iters = {expert_retrain_checkpoint_iters}
expert_retrain_optim_kwargs = {{lr = {lr}, momentum = 0.9, nesterov = true, weight_decay = {wd}}}
expert_retrain_scheduler_kwargs = {{milestones = {milestones}, gamma = 0.1}}

checkpoint_sampling = "biased"
alpha_ckpt = 0.01
train_pct = 1.0
gamma_stealth = 1.0
attack = "backdoor"

metrics_log_path = "{cell_dir}/metrics.json"

[generate_labels_trigger_joint.expert_config]
experts = 1
min = 0
max = 20
trajectories = [50, 100, 150, 200]

[generate_labels_trigger_joint.attack_config]
iterations = {iterations}
one_hot_temp = 5
"""

# Adds lambda_joint and makes lambda_tv/lambda_lpips format slots instead of hardcoded 0.0.
_JOINT_TRIGGER_TEMPLATE_STEALTH = JOINT_TRIGGER_TEMPLATE.replace(
    "lambda_delta = {lambda_delta}\n",
    "lambda_delta = {lambda_delta}\nlambda_joint = {lambda_joint}\n",
).replace(
    "lambda_tv = 0.0\nlambda_lpips = 0.0\n",
    "lambda_tv = {lambda_tv}\nlambda_lpips = {lambda_lpips}\n",
)
assert "{lambda_joint}" in _JOINT_TRIGGER_TEMPLATE_STEALTH, (
    "JOINT_TRIGGER_TEMPLATE's lambda_delta line changed shape -- update the splice."
)
assert "{lambda_tv}" in _JOINT_TRIGGER_TEMPLATE_STEALTH and (
    "{lambda_lpips}" in _JOINT_TRIGGER_TEMPLATE_STEALTH
), "JOINT_TRIGGER_TEMPLATE's lambda_tv/lambda_lpips lines changed shape -- update the splice."

# The on-disk winning reference config, including lambda_joint/iterations.
STEALTH_BASE = {
    "epsilon": 1.0,
    "lr_delta": 1e-2,
    "lambda_bd": 1.0,
    "lambda_penalty": 1.0,
    "lambda_delta": 0.0,
    "lambda_joint": 1.0,
    "lambda_tv": 0.0,
    "lambda_lpips": 0.0,
    "iterations": 25,
}

# The three finalist tags (one-at-a-time stealth-regularization variants around STEALTH_BASE).
CONFIG_TAGS = [
    ("baseline", {}),
    ("eps_0p50_lpips_0p1", {"epsilon": 0.5, "lambda_lpips": 0.1}),
    ("eps_16_255", {"epsilon": 16 / 255}),
]

MODEL_FLAGS = ["r32p", "convnext_micro"]
DATASETS = ["cifar", "svhn"]
SEEDS = list(range(10))

GEN_NUM_POISONED = 1
GEN_NUM_HONESTS = 0
GEN_AGG_METHOD = "mean"

DEPLOY_BUDGETS = [0, 150, 300, 500, 1000, 2000, 2500, 5000]

DEPLOY_SINGLE_USER_NUM_POISONED = 1
DEPLOY_SINGLE_USER_NUM_HONESTS = 0
DEPLOY_SINGLE_USER_AGG_METHOD = "mean"

DEPLOY_FEDERATED_NUM_POISONED = 3
DEPLOY_FEDERATED_NUM_HONESTS = 7
DEPLOY_AGG_METHODS_FEDERATED = ["mean", "krum", "multikrum", "median", "trmean"]
FEDERATED_TAG = "federated_3vs7"
# Only multikrum/krum are instrumented for poison-selection tracking (see util.py's mini_train_multi).
_TRACKED_AGG_METHODS = {"multikrum", "krum"}

MODULE_NAME = "generate_labels_trigger_joint"

# Relative path: GEN/FLIPS and USER stages may run on machines with different repo roots.
EXP_BASE = Path(
    "experiments/federated_experiments/threat_model_direct_trigger_joint_paper_main_campaign"
)

# Makes the bootstrap checkpoint path dataset-aware.
_TRAIN_EXPERT_TEMPLATE_DATASET = TRAIN_EXPERT_TEMPLATE.replace(
    'output_dir = "{cluster_root}/out/checkpoints/{model_flag}_1xs/seed{seed}/0/"\n',
    'output_dir = "{cluster_root}/out/checkpoints/{model_flag}_{dataset}_1xs/seed{seed}/0/"\n',
).replace(
    'poisoner = "1xs"\n',
    'poisoner = "1xs"\n'
    "budget = {budget}\n",
)
assert _TRAIN_EXPERT_TEMPLATE_DATASET != TRAIN_EXPERT_TEMPLATE, (
    "TRAIN_EXPERT_TEMPLATE's output_dir/poisoner lines changed shape -- update the splice."
)

# Explicit per-dataset budget: SVHN's imbalanced classes need a lower default.
BOOTSTRAP_BUDGET = {"cifar": 5000, "svhn": 4500}

_JOINT_TRIGGER_TEMPLATE_MAIN = _JOINT_TRIGGER_TEMPLATE_STEALTH.replace(
    "expert_retrain_scheduler_kwargs = {{milestones = {milestones}, gamma = 0.1}}\n",
    "expert_retrain_scheduler_kwargs = {{milestones = {milestones}, gamma = 0.1}}\n"
    "seed = {rng_seed}\n",
)
assert _JOINT_TRIGGER_TEMPLATE_MAIN != _JOINT_TRIGGER_TEMPLATE_STEALTH, (
    "_JOINT_TRIGGER_TEMPLATE_STEALTH's expert_retrain_scheduler_kwargs line changed shape -- "
    "update the splice."
)

_TRAIN_USER_TEMPLATE_TRACKABLE = TRAIN_USER_TEMPLATE.replace(
    'agg_method = "{agg_method}"\n',
    'agg_method = "{agg_method}"\n'
    "track_poison_selection = {track_poison_selection}\n",
)
assert _TRAIN_USER_TEMPLATE_TRACKABLE != TRAIN_USER_TEMPLATE, (
    "TRAIN_USER_TEMPLATE's agg_method line changed shape -- update the splice."
)

DELTA_MIN_FRAC = 0.0  # inert (lambda_mag=0.0 in the template); kept for check_delta_min_feasible's guard.


def bootstrap_output_dir(model_flag, dataset, seed):
    return f"{CLUSTER_ROOT}/out/checkpoints/{model_flag}_{dataset}_1xs/seed{seed}"


def cell_name(model_flag, dataset, tag, seed):
    return f"{model_flag}/{dataset}/{tag}/seed{seed}"


def bootstrap_cell_name(model_flag, dataset, seed):
    return f"train_expert/{model_flag}_{dataset}_1xs/seed{seed}"


def _train_user_configs(
    *, base_dir, budgets, num_honests, num_poisoned, agg_methods, flips_dir, model_flag,
    dataset, trigger_path, lr, wd, milestones,
):
    configs = {}
    for agg_method in agg_methods:
        for budget in budgets:
            train_user_dir = base_dir / agg_method / f"train_user_{budget}"
            configs[train_user_dir / "config.toml"] = _TRAIN_USER_TEMPLATE_TRACKABLE.format(
                flips_dir=flips_dir,
                train_user_dir=train_user_dir,
                model_flag=model_flag,
                dataset=dataset,
                source_label=SOURCE_LABEL,
                target_label=TARGET_LABEL,
                trigger_path=trigger_path,
                budget=budget,
                deploy_num_honests=num_honests,
                deploy_num_poisoned=num_poisoned,
                agg_method=agg_method,
                track_poison_selection=(
                    "true" if agg_method in _TRACKED_AGG_METHODS else "false"
                ),
                lr=lr,
                wd=wd,
                milestones=milestones,
            )
    return configs


def generate_bootstrap_cell(model_flag, dataset, seed, dry_run=False):
    lr = LEARNING_RATE.get(model_flag, 0.1)
    wd = WEIGHT_DECAY.get(model_flag, 2e-4)
    milestones = MILESTONE.get(model_flag, [75, 125])
    rng_seed = draw_rng_seed(model_flag, (dataset, seed))

    bootstrap_cfg_dir = EXP_BASE / bootstrap_cell_name(model_flag, dataset, seed)

    content = _TRAIN_EXPERT_TEMPLATE_DATASET.format(
        cluster_root=CLUSTER_ROOT,
        model_flag=model_flag,
        dataset=dataset,
        seed=seed,
        rng_seed=rng_seed,
        source_label=SOURCE_LABEL,
        target_label=TARGET_LABEL,
        budget=BOOTSTRAP_BUDGET[dataset],
        checkpoint_iters=GEN_EXPERT_RETRAIN_CHECKPOINT_ITERS,
        epochs=GEN_EXPERT_RETRAIN_EPOCHS,
        lr=lr,
        wd=wd,
        milestones=milestones,
        wandb_block_train_expert=wandb_block(
            "train_expert", f"train_expert/{model_flag}/{dataset}/seed{seed}",
            enabled=WANDB_ENABLED, project=WANDB_PROJECT,
            mode=WANDB_MODE, entity=WANDB_ENTITY, group=MODULE_NAME,
        ),
    )
    path = bootstrap_cfg_dir / "config.toml"
    assert "out/checkpoints" not in str(path), f"Refusing to write under out/checkpoints/: {path}"
    if not dry_run:
        write_config(path, content)
        validate_config(path)
    return [path]


def generate_cell(model_flag, dataset, tag, overrides, seed, dry_run=False):
    cfg = {**STEALTH_BASE, **overrides}

    feasible, delta_min, max_reachable = check_delta_min_feasible(
        dataset, cfg["epsilon"], DELTA_MIN_FRAC,
    )
    if not feasible:
        reason = (
            f"delta_min_frac={DELTA_MIN_FRAC} -> delta_min={delta_min:.4f} > "
            f"epsilon*sqrt(numel)={max_reachable:.4f} at epsilon={cfg['epsilon']} -- "
            "structurally unreachable post-clamp; refusing to generate this cell."
        )
        print(f"REFUSED [{model_flag}/{dataset}/{tag}/seed{seed}]: {reason}")
        return [], reason

    lr = LEARNING_RATE.get(model_flag, 0.1)
    wd = WEIGHT_DECAY.get(model_flag, 2e-4)
    milestones = MILESTONE.get(model_flag, [75, 125])
    rng_seed = draw_rng_seed(model_flag, (dataset, seed))

    bootstrap_dir = bootstrap_output_dir(model_flag, dataset, seed)
    cell_dir = EXP_BASE / cell_name(model_flag, dataset, tag, seed)
    module_dir = cell_dir / "gen_labels_trigger_joint"

    configs = {
        module_dir / "config.toml": _JOINT_TRIGGER_TEMPLATE_MAIN.format(
            bootstrap_dir=bootstrap_dir,
            cell_dir=module_dir,
            model_flag=model_flag,
            dataset=dataset,
            source_label=SOURCE_LABEL,
            target_label=TARGET_LABEL,
            init=GEN_INIT,
            epsilon=cfg["epsilon"],
            lr_delta=cfg["lr_delta"],
            lambda_bd=cfg["lambda_bd"],
            lambda_penalty=cfg["lambda_penalty"],
            lambda_delta=cfg["lambda_delta"],
            lambda_joint=cfg["lambda_joint"],
            lambda_tv=cfg["lambda_tv"],
            lambda_lpips=cfg["lambda_lpips"],
            gen_num_honests=GEN_NUM_HONESTS,
            gen_num_poisoned=GEN_NUM_POISONED,
            gen_agg_method=GEN_AGG_METHOD,
            trigger_constraint=GEN_TRIGGER_CONSTRAINT,
            n_checkpoints_per_step=GEN_N_CHECKPOINTS_PER_STEP,
            expert_retrain_interval=GEN_EXPERT_RETRAIN_INTERVAL,
            expert_retrain_epochs=GEN_EXPERT_RETRAIN_EPOCHS,
            expert_retrain_checkpoint_iters=GEN_EXPERT_RETRAIN_CHECKPOINT_ITERS,
            rng_seed=rng_seed,
            lr=lr,
            wd=wd,
            milestones=milestones,
            iterations=cfg["iterations"],
        ),
    }

    trigger_path = (
        module_dir / "trigger"
        / f"opt_trig_direct_joint_{GEN_INIT}_{model_flag}_{dataset}_{GEN_NUM_POISONED}vs{GEN_NUM_HONESTS}.pt"
    )

    # -- single_user branch: 1-poisoned/0-honest/mean, same split as generation time. --
    single_user_flips_dir = cell_dir / "select_flips"
    configs[single_user_flips_dir / "config.toml"] = SELECT_FLIPS_TEMPLATE.format(
        budgets=DEPLOY_BUDGETS,
        module_dir=module_dir,
        flips_dir=single_user_flips_dir,
        deploy_num_honests=DEPLOY_SINGLE_USER_NUM_HONESTS,
        deploy_num_poisoned=DEPLOY_SINGLE_USER_NUM_POISONED,
    )
    configs.update(_train_user_configs(
        base_dir=cell_dir, budgets=DEPLOY_BUDGETS,
        num_honests=DEPLOY_SINGLE_USER_NUM_HONESTS, num_poisoned=DEPLOY_SINGLE_USER_NUM_POISONED,
        agg_methods=[DEPLOY_SINGLE_USER_AGG_METHOD], flips_dir=single_user_flips_dir,
        model_flag=model_flag, dataset=dataset, trigger_path=trigger_path,
        lr=lr, wd=wd, milestones=milestones,
    ))

    # -- federated branch: 3-poisoned/7-honest, swept over DEPLOY_AGG_METHODS_FEDERATED. --
    fed_dir = cell_dir / FEDERATED_TAG
    fed_flips_dir = fed_dir / "select_flips"
    configs[fed_flips_dir / "config.toml"] = SELECT_FLIPS_TEMPLATE.format(
        budgets=DEPLOY_BUDGETS,
        module_dir=module_dir,
        flips_dir=fed_flips_dir,
        deploy_num_honests=DEPLOY_FEDERATED_NUM_HONESTS,
        deploy_num_poisoned=DEPLOY_FEDERATED_NUM_POISONED,
    )
    configs.update(_train_user_configs(
        base_dir=fed_dir, budgets=DEPLOY_BUDGETS,
        num_honests=DEPLOY_FEDERATED_NUM_HONESTS, num_poisoned=DEPLOY_FEDERATED_NUM_POISONED,
        agg_methods=DEPLOY_AGG_METHODS_FEDERATED, flips_dir=fed_flips_dir,
        model_flag=model_flag, dataset=dataset, trigger_path=trigger_path,
        lr=lr, wd=wd, milestones=milestones,
    ))

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


def generate_all(dry_run=False):
    all_paths, refused = [], []
    for model_flag in MODEL_FLAGS:
        for dataset in DATASETS:
            for seed in SEEDS:
                all_paths += generate_bootstrap_cell(model_flag, dataset, seed, dry_run=dry_run)
            for tag, overrides in CONFIG_TAGS:
                for seed in SEEDS:
                    paths, reason = generate_cell(
                        model_flag, dataset, tag, overrides, seed, dry_run=dry_run,
                    )
                    all_paths += paths
                    if reason:
                        refused.append((model_flag, dataset, tag, seed, reason))
    return all_paths, refused


def list_grid():
    """Enumerates this campaign's (model_flag, dataset, tag, seed, branch, agg_method, budget) cells as plain dicts."""
    cells = []
    for model_flag in MODEL_FLAGS:
        for dataset in DATASETS:
            for tag, _ in CONFIG_TAGS:
                for seed in SEEDS:
                    for budget in DEPLOY_BUDGETS:
                        cells.append({
                            "model_flag": model_flag, "dataset": dataset, "tag": tag,
                            "seed": seed, "branch": "single_user",
                            "agg_method": DEPLOY_SINGLE_USER_AGG_METHOD, "budget": budget,
                        })
                    for agg_method in DEPLOY_AGG_METHODS_FEDERATED:
                        for budget in DEPLOY_BUDGETS:
                            cells.append({
                                "model_flag": model_flag, "dataset": dataset, "tag": tag,
                                "seed": seed, "branch": FEDERATED_TAG,
                                "agg_method": agg_method, "budget": budget,
                            })
    return cells


def list_tags():
    return [tag for tag, _ in CONFIG_TAGS]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--print-grid", action="store_true",
        help="print the full (model, dataset, tag, seed, branch, agg_method, budget) cell list "
             "as JSON and exit, without writing configs",
    )
    parser.add_argument(
        "--print-tags", action="store_true",
        help="print just the three finalist tags (one per line) and exit, without writing configs",
    )
    args = parser.parse_args()

    if args.print_grid:
        import json

        print(json.dumps(list_grid(), indent=2))
        raise SystemExit(0)

    if args.print_tags:
        for tag in list_tags():
            print(tag)
        raise SystemExit(0)

    paths, refused = generate_all(dry_run=args.dry_run)

    if args.dry_run:
        print(f"\n[DRY RUN] {MODULE_NAME} paper_main_campaign: {len(paths)} config files would be written.")
    else:
        print(f"\n{MODULE_NAME} paper_main_campaign: {len(paths)} config files written and schema-validated.")

    if refused:
        print(f"\n{len(refused)} cell(s) REFUSED (delta_min infeasible):")
        for model_flag, dataset, tag, seed, reason in refused:
            print(f"  [{model_flag}/{dataset}/{tag}/seed{seed}] {reason}")
