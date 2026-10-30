#!/bin/bash
###
# compare_additive_warp_slurm.sh
#
# One-command A/B test: generates and submits the full
# train_expert -> generate_labels_trigger_joint -> federated_select_flips ->
# federated_train_user pipeline TWICE, once with trigger_type="additive"
# and once with trigger_type="warp" (WaNet-style learned warp), for the same
# model/dataset/seed/budget(s), so the two runs' metrics.json / ASR/CTA are
# directly comparable.
#
# Each variant writes to its own sibling experiments/ tree (via
# gen_configs.py's --exp-base-tag), so the two runs never collide -- each
# also retrains its own train_expert checkpoint (not shared between variants).
#
# Usage (from repo root, login-node shell):
#   SLURM_ACCOUNT=<account> ./orchestrate_slurm/compare_additive_warp_slurm.sh
#   SLURM_ACCOUNT=<account> DRY_RUN=1 ./orchestrate_slurm/compare_additive_warp_slurm.sh
#   MODEL_FLAGS=r32p DATASETS=cifar SEEDS=0 BUDGETS="150 500" \
#       SLURM_ACCOUNT=<account> ./orchestrate_slurm/compare_additive_warp_slurm.sh
###

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
export BASE_DIR
cd "$BASE_DIR"

MODEL_FLAGS_SINGLE="${MODEL_FLAGS:-r32p}"
DATASETS_SINGLE="${DATASETS:-cifar}"
SEEDS_SINGLE="${SEEDS:-0}"
BUDGETS_SINGLE="${BUDGETS:-150}"
DRY_RUN="${DRY_RUN:-0}"

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

echo "[PLAN] comparing trigger_type=additive vs trigger_type=warp on $MODEL_FLAG/$DATASET/seed$SEED, budgets=$BUDGETS_SINGLE"

for VARIANT in additive warp; do
    echo "[GEN] generating $VARIANT configs (exp-base-tag=$VARIANT)..."
    GEN_ARGS=(--single-cell --trigger-type "$VARIANT" --exp-base-tag "$VARIANT")
    [ "$DRY_RUN" = "1" ] && GEN_ARGS+=(--dry-run)
    MODEL_FLAGS="$MODEL_FLAG" DATASETS="$DATASET" SEEDS="$SEED" BUDGETS="$BUDGETS_SINGLE" \
        python -m modules.generate_labels_trigger_joint.gen_configs "${GEN_ARGS[@]}"
done

for VARIANT in additive warp; do
    echo "[SUBMIT] launching $VARIANT pipeline on Slurm..."
    RUN_EXPERT=1 \
    MODEL_FLAGS="$MODEL_FLAG" DATASETS="$DATASET" SEEDS="$SEED" BUDGETS="$BUDGETS_SINGLE" \
    AGG_METHODS=mean NUM_POISONED=1 NUM_HONESTS=0 \
    EXP_BASE_REL="federated_experiments/threat_model_direct_trigger_joint_${VARIANT}" \
    DRY_RUN="$DRY_RUN" \
        "$SCRIPT_DIR/orchestrate_runs_trigger_joint_slurm.sh"
done

echo "[DONE] both variants submitted (or dry-run printed above)."
echo "[COMPARE] once finished, compare:"
echo "  experiments/federated_experiments/threat_model_direct_trigger_joint_additive/$MODEL_FLAG/$DATASET/1vs0/mean/seed$SEED/train_user_*/metrics.json"
echo "  experiments/federated_experiments/threat_model_direct_trigger_joint_warp/$MODEL_FLAG/$DATASET/1vs0/mean/seed$SEED/train_user_*/metrics.json"
