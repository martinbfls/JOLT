#!/bin/bash
###
# compare_additive_warp_slurm.sh
#
# One-command A/B test: generates and submits the full
# train_expert -> generate_labels_trigger_joint -> federated_select_flips ->
# federated_train_user pipeline ONCE PER VARIANT -- by default trigger_type="additive"
# and trigger_type="warp" (WaNet-style learned warp) -- for the same
# model/dataset/seed/budget(s), so the runs' metrics.json / ASR/CTA are
# directly comparable.
#
# Each variant writes to its own sibling experiments/ tree (via
# gen_configs.py's --exp-base-tag), so the runs never collide -- each
# also retrains its own train_expert checkpoint (not shared between variants).
#
# Usage (from repo root, login-node shell):
#   SLURM_ACCOUNT=<account> ./orchestrate_slurm/compare_additive_warp_slurm.sh
#   SLURM_ACCOUNT=<account> DRY_RUN=1 ./orchestrate_slurm/compare_additive_warp_slurm.sh
#   MODEL_FLAGS=r32p DATASETS=cifar SEEDS=0 BUDGETS="150 500" \
#       SLURM_ACCOUNT=<account> ./orchestrate_slurm/compare_additive_warp_slurm.sh
#
# VARIANTS (default "additive warp") is the list of trigger_types to run; add "hybrid" (warp
# composed with the additive delta, optimized jointly) for the three-way comparison. EPSILON,
# if set, is applied to EVERY variant, which is what makes the comparison fair: the additive
# run and the hybrid run then share the same additive Linf budget, the hybrid one simply also
# having the warp (WARP_S) on top.
#   VARIANTS="additive warp hybrid" EPSILON=0.0314 WARP_S=4.0 BUDGETS=1500 \
#       SLURM_ACCOUNT=<account> ./orchestrate_slurm/compare_additive_warp_slurm.sh
###

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
export BASE_DIR
cd "$BASE_DIR"

# shellcheck source=slurm_lib.sh
source "$SCRIPT_DIR/slurm_lib.sh"

# The config generation (gen_configs.py) just writes/validates .toml files --
# no GPU needed -- but needs the conda env's python, which is normally only
# on compute nodes, not the login node. Run it via `srun` on a tiny CPU
# allocation instead of requiring an active conda env / python in PATH here.
GEN_TIME="${GEN_CONFIGS_TIME:-00:10:00}"
GEN_CPUS="${GEN_CONFIGS_CPUS:-2}"
GEN_MEM="${GEN_CONFIGS_MEM:-4G}"

run_gen_configs_on_compute_node() {
    srun -A "$SLURM_ACCOUNT" -p "$SLURM_PARTITION" \
        --cpus-per-task="$GEN_CPUS" --mem="$GEN_MEM" --time="$GEN_TIME" \
        bash -c '
            set -euo pipefail
            cd "$0"
            unset PYTHONPATH PYTHONHOME || true
            export PATH="$1/bin:$PATH"
            if [ "$(command -v python)" != "$1/bin/python" ]; then
                echo "[gen_configs] FATAL: python is $(command -v python), expected $1/bin/python" >&2
                exit 1
            fi
            shift 1
            python -m modules.generate_labels_trigger_joint.gen_configs "$@"
        ' "$BASE_DIR" "$CONDA_ENV" "$@"
}

MODEL_FLAGS_SINGLE="${MODEL_FLAGS:-r32p}"
DATASETS_SINGLE="${DATASETS:-cifar}"
SEEDS_SINGLE="${SEEDS:-0}"
BUDGETS_SINGLE="${BUDGETS:-150}"
DRY_RUN="${DRY_RUN:-0}"
read -ra _VARIANTS <<< "${VARIANTS:-additive warp}"
# Optional, applied to every variant (see the header): the additive component's budget / the
# warp's max displacement. Unset leaves gen_configs.py's own defaults.
EPSILON="${EPSILON:-}"
WARP_S="${WARP_S:-}"

read -ra _MODEL_FLAGS <<< "$MODEL_FLAGS_SINGLE"
read -ra _DATASETS    <<< "$DATASETS_SINGLE"
read -ra _SEEDS       <<< "$SEEDS_SINGLE"
MODEL_FLAG="${_MODEL_FLAGS[0]}"
DATASET="${_DATASETS[0]}"
SEED="${_SEEDS[0]}"

if [ "${#_MODEL_FLAGS[@]}" -gt 1 ] || [ "${#_DATASETS[@]}" -gt 1 ] || [ "${#_SEEDS[@]}" -gt 1 ]; then
    echo "[NOTE] this A/B test runs exactly ONE model/dataset/seed cell per variant;" >&2
    echo "[NOTE] only the first of each (MODEL_FLAGS, DATASETS, SEEDS) is used: $MODEL_FLAG/$DATASET/seed$SEED" >&2
fi

echo "[PLAN] comparing trigger_type in [${_VARIANTS[*]}] on $MODEL_FLAG/$DATASET/seed$SEED, budgets=$BUDGETS_SINGLE"
echo "[PLAN] epsilon=${EPSILON:-<gen_configs default>} warp_s=${WARP_S:-<gen_configs default>}"

for VARIANT in "${_VARIANTS[@]}"; do
    echo "[GEN] generating $VARIANT configs (exp-base-tag=$VARIANT)..."
    GEN_ARGS=(--single-cell --trigger-type "$VARIANT" --exp-base-tag "$VARIANT" --budgets "$BUDGETS_SINGLE"
              --model-flag "$MODEL_FLAG" --dataset "$DATASET" --seed "$SEED")
    [ -n "$EPSILON" ] && GEN_ARGS+=(--epsilon "$EPSILON")
    [ -n "$WARP_S" ] && GEN_ARGS+=(--warp-s "$WARP_S")
    [ "$DRY_RUN" = "1" ] && GEN_ARGS+=(--dry-run)
    run_gen_configs_on_compute_node "${GEN_ARGS[@]}"
done

for VARIANT in "${_VARIANTS[@]}"; do
    echo "[SUBMIT] launching $VARIANT pipeline on Slurm..."
    RUN_EXPERT=1 \
    MODEL_FLAGS="$MODEL_FLAG" DATASETS="$DATASET" SEEDS="$SEED" BUDGETS="$BUDGETS_SINGLE" \
    AGG_METHODS=mean NUM_POISONED=1 NUM_HONESTS=0 \
    EXP_BASE_REL="federated_experiments/threat_model_direct_trigger_joint_${VARIANT}" \
    CHECKPOINT_TAG="$VARIANT" \
    DRY_RUN="$DRY_RUN" \
        "$SCRIPT_DIR/orchestrate_runs_trigger_joint_slurm.sh"
done

echo "[DONE] ${#_VARIANTS[@]} variant(s) submitted (or dry-run printed above)."
echo "[COMPARE] once finished, compare:"
for VARIANT in "${_VARIANTS[@]}"; do
    echo "  experiments/federated_experiments/threat_model_direct_trigger_joint_${VARIANT}/$MODEL_FLAG/$DATASET/1vs0/mean/seed$SEED/train_user_*/metrics.json"
done
echo "[COMPARE] side-by-side ASR/CTA + trigger figure:"
COMPARE_CMD="python scripts/show_results/compare_additive_warp.py"
for VARIANT in "${_VARIANTS[@]}"; do
    COMPARE_CMD+=" --${VARIANT}-dir experiments/federated_experiments/threat_model_direct_trigger_joint_${VARIANT}"
done
echo "  $COMPARE_CMD --model-flag $MODEL_FLAG --dataset $DATASET --seed $SEED --budget <budget>"
