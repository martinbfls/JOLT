#!/bin/bash
###
# compare_federated_multikrum_slurm.sh
#
# One-command federated A/B test: re-deploys the ALREADY-GENERATED additive
# and warp triggers (from compare_additive_warp_slurm.sh) under per-layer
# multikrum aggregation with real honest workers (default 3 poisoned / 7
# honest, matching the paper's federated_3vs7), for the same budget(s).
#
# Unlike the single-user comparison, this federated deployment's
# federated_select_flips MUST be re-run (not reused): its flip assignment
# depends on num_honests/num_poisoned, which differ from the single-user
# GEN deployment (1 poisoned / 0 honest). GEN/train_expert are NOT re-run --
# the trigger itself is unchanged, only how it's deployed federated-ly.
#
# Prerequisite: compare_additive_warp_slurm.sh (or an equivalent --single-cell
# run) must have already produced the additive/warp triggers for this
# model/dataset/seed -- this script aborts with a clear message if not.
#
# Usage (from repo root, login-node shell):
#   SLURM_ACCOUNT=<account> ./orchestrate_slurm/compare_federated_multikrum_slurm.sh
#   SLURM_ACCOUNT=<account> DRY_RUN=1 ./orchestrate_slurm/compare_federated_multikrum_slurm.sh
#   BUDGETS=1500 DEPLOY_NUM_POISONED=3 DEPLOY_NUM_HONESTS=7 DEPLOY_AGG_METHOD=multikrum \
#       SLURM_ACCOUNT=<account> ./orchestrate_slurm/compare_federated_multikrum_slurm.sh
###

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
export BASE_DIR
cd "$BASE_DIR"

# shellcheck source=slurm_lib.sh
source "$SCRIPT_DIR/slurm_lib.sh"

MODEL_FLAG="${MODEL_FLAG:-r32p}"
DATASET="${DATASET:-cifar}"
SEED="${SEED:-0}"
BUDGETS_SINGLE="${BUDGETS:-1500}"
DEPLOY_NUM_POISONED="${DEPLOY_NUM_POISONED:-3}"
DEPLOY_NUM_HONESTS="${DEPLOY_NUM_HONESTS:-7}"
DEPLOY_AGG_METHOD="${DEPLOY_AGG_METHOD:-multikrum}"
DRY_RUN="${DRY_RUN:-0}"

FLIPS_TIME="${FLIPS_TIME:-00:10:00}"
USER_TIME="${USER_TIME:-0-01:00:00}"

GEN_TIME="${GEN_CONFIGS_TIME:-00:10:00}"
GEN_CPUS="${GEN_CONFIGS_CPUS:-2}"
GEN_MEM="${GEN_CONFIGS_MEM:-4G}"

echo "[PLAN] federated deploy: agg=$DEPLOY_AGG_METHOD (per-layer) poisoned=$DEPLOY_NUM_POISONED honests=$DEPLOY_NUM_HONESTS"
echo "[PLAN] model=$MODEL_FLAG dataset=$DATASET seed=$SEED budgets=$BUDGETS_SINGLE, variants: additive, warp"

run_gen_configs_on_compute_node() {
    srun -A "$SLURM_ACCOUNT" -p "$SLURM_PARTITION" \
        --cpus-per-task="$GEN_CPUS" --mem="$GEN_MEM" --time="$GEN_TIME" \
        bash -c "
            set -euo pipefail
            cd '$BASE_DIR'
            unset PYTHONPATH PYTHONHOME || true
            export PATH='$CONDA_ENV/bin:\$PATH'
            if [ \"\$(command -v python)\" != '$CONDA_ENV/bin/python' ]; then
                echo '[gen_configs] FATAL: python is '\"\$(command -v python)\"', expected $CONDA_ENV/bin/python' >&2
                exit 1
            fi
            python -m modules.generate_labels_trigger_joint.gen_configs $*
        "
}

FLIPS_JOBS=()
USER_JOBS=()

for VARIANT in additive warp; do
    echo "[GEN] re-deploying $VARIANT trigger under $DEPLOY_AGG_METHOD ${DEPLOY_NUM_POISONED}vs${DEPLOY_NUM_HONESTS}..."
    GEN_ARGS=(--deploy-federated --trigger-type "$VARIANT" --exp-base-tag "$VARIANT"
              --model-flag "$MODEL_FLAG" --dataset "$DATASET" --seed "$SEED" --budgets "$BUDGETS_SINGLE"
              --deploy-num-poisoned "$DEPLOY_NUM_POISONED" --deploy-num-honests "$DEPLOY_NUM_HONESTS"
              --deploy-agg-method "$DEPLOY_AGG_METHOD")
    [ "$DRY_RUN" = "1" ] && GEN_ARGS+=(--dry-run)
    run_gen_configs_on_compute_node "${GEN_ARGS[@]}"

    deploy_dir="federated_experiments/threat_model_direct_trigger_joint_${VARIANT}/${MODEL_FLAG}/${DATASET}/1vs0/mean/seed${SEED}/${DEPLOY_AGG_METHOD}_${DEPLOY_NUM_POISONED}vs${DEPLOY_NUM_HONESTS}"
    tag="${MODEL_FLAG}_${DATASET}_seed${SEED}_${VARIANT}"

    FLIPS_JOBS+=("python run_experiment.py $deploy_dir/select_flips|fed_flips_${tag}")
    for budget in $BUDGETS_SINGLE; do
        USER_JOBS+=("python run_experiment.py $deploy_dir/train_user_${budget}|fed_user_${tag}_${budget}")
    done
done

if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY-RUN] nothing submitted."
    printf '[DRY-RUN] %s\n' "${FLIPS_JOBS[@]}"
    printf '[DRY-RUN] %s\n' "${USER_JOBS[@]}"
    exit 0
fi

preflight_slurm || exit 1

TIME_PER_TASK="$FLIPS_TIME"
GPUS_PER_TASK=0 submit_job_pool_slurm FLIPS_JOBS "fed_flips" "" || exit 1
flips_barrier=$(submit_barrier_slurm "fed_barrier_flips" "afterok:$(join_job_ids "${SUBMITTED_JOB_IDS[@]}")") || exit 1
echo "[PHASE] FLIPS submitted (${#FLIPS_JOBS[@]} jobs) -> barrier $flips_barrier"

TIME_PER_TASK="$USER_TIME"
submit_job_pool_slurm USER_JOBS "fed_user" "afterok:$flips_barrier" || exit 1
echo "[PHASE] USER submitted (${#USER_JOBS[@]} jobs)."

echo "[DONE] federated $DEPLOY_AGG_METHOD deploy submitted for both additive and warp."
echo "[COMPARE] once finished, compare:"
for VARIANT in additive warp; do
    for budget in $BUDGETS_SINGLE; do
        echo "  experiments/federated_experiments/threat_model_direct_trigger_joint_${VARIANT}/${MODEL_FLAG}/${DATASET}/1vs0/mean/seed${SEED}/${DEPLOY_AGG_METHOD}_${DEPLOY_NUM_POISONED}vs${DEPLOY_NUM_HONESTS}/train_user_${budget}/logs/federated_train_user_metrics.json"
    done
done
