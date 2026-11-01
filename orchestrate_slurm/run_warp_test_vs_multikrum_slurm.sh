#!/bin/bash
###
# run_warp_test_vs_multikrum_slurm.sh
#
# One-command warp-trigger test: regenerates the warp campaign FROM SCRATCH
# with a deliberately loosened/rebalanced optimization (warp_s=8.0, 8x the
# default max displacement; lambda_bd=1.0; lambda_joint=1.0, the clean-task-
# loss regularization term), then submits BOTH deployments of that one
# trigger on Slurm so the two training regimes are directly comparable:
#
#   EXPERT -> GEN (trigger_type=warp, least-constrained) -+-> CENTRALIZED FLIPS -> CENTRALIZED USER (1 poisoned / 0 honest)
#                                                          +-> FED_GEN (--deploy-federated) -> FEDERATED FLIPS -> FEDERATED USER (3 poisoned / 7 honest, multikrum)
#
# FED_GEN is its own tiny Slurm job (not run at submission time): the
# federated select_flips/train_user configs can only be generated once the
# trigger .pt file actually exists on disk, i.e. after GEN has finished, so
# it depends on GEN's barrier exactly like the centralized branch does.
#
# Usage (from repo root, login-node shell):
#   SLURM_ACCOUNT=<account> ./orchestrate_slurm/run_warp_test_vs_multikrum_slurm.sh
#   SLURM_ACCOUNT=<account> DRY_RUN=1 ./orchestrate_slurm/run_warp_test_vs_multikrum_slurm.sh
#   WARP_S=4.0 BUDGETS=1500 SLURM_ACCOUNT=<account> ./orchestrate_slurm/run_warp_test_vs_multikrum_slurm.sh
#
# VARIANT_TAG (default "warp") names this cell's experiments/out/checkpoints subtree AND its
# Slurm job names -- override it (e.g. per sweep_warp_min_perturbation_slurm.sh cell) to run
# several WARP_S/LAMBDA_LPIPS combos side by side without colliding on paths or job names.
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
DRY_RUN="${DRY_RUN:-0}"
VARIANT_TAG="${VARIANT_TAG:-warp}"

# Least-constrained warp + rebalanced losses, per this test's spec.
WARP_S="${WARP_S:-8.0}"
WARP_K="${WARP_K:-4}"
LR_WARP="${LR_WARP:-0.1}"
LAMBDA_BD="${LAMBDA_BD:-1.0}"
LAMBDA_JOINT="${LAMBDA_JOINT:-1.0}"
LAMBDA_LPIPS="${LAMBDA_LPIPS:-0.0}"

DEPLOY_NUM_POISONED="${DEPLOY_NUM_POISONED:-3}"
DEPLOY_NUM_HONESTS="${DEPLOY_NUM_HONESTS:-7}"
DEPLOY_AGG_METHOD="${DEPLOY_AGG_METHOD:-multikrum}"

EXPERT_TIME="${EXPERT_TIME:-0-02:00:00}"
GEN_TIME="${GEN_TIME:-1-00:00:00}"
FLIPS_TIME="${FLIPS_TIME:-00:10:00}"
USER_TIME="${USER_TIME:-0-01:00:00}"
GEN_CONFIGS_TIME="${GEN_CONFIGS_TIME:-00:10:00}"

EXP_BASE_REL="federated_experiments/threat_model_direct_trigger_joint_${VARIANT_TAG}"
CELL_REL="$EXP_BASE_REL/$MODEL_FLAG/$DATASET/1vs0/mean/seed$SEED"
FED_DEPLOY_REL="$CELL_REL/${DEPLOY_AGG_METHOD}_${DEPLOY_NUM_POISONED}vs${DEPLOY_NUM_HONESTS}"

echo "[PLAN] variant=$VARIANT_TAG warp_s=$WARP_S warp_k=$WARP_K lambda_bd=$LAMBDA_BD lambda_joint=$LAMBDA_JOINT lambda_lpips=$LAMBDA_LPIPS"
echo "[PLAN] model=$MODEL_FLAG dataset=$DATASET seed=$SEED budgets=$BUDGETS_SINGLE"
echo "[PLAN] centralized (1vs0, mean) vs federated ($DEPLOY_AGG_METHOD, ${DEPLOY_NUM_POISONED}vs${DEPLOY_NUM_HONESTS})"

run_gen_configs_on_compute_node() {
    srun -A "$SLURM_ACCOUNT" -p "$SLURM_PARTITION" \
        --cpus-per-task=2 --mem=4G --time="$GEN_CONFIGS_TIME" \
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

echo "[GEN] regenerating the whole $VARIANT_TAG cell from scratch (train_expert, gen, centralized flips/user)..."
GEN_ARGS=(--single-cell --trigger-type warp --exp-base-tag "$VARIANT_TAG"
          --model-flag "$MODEL_FLAG" --dataset "$DATASET" --seed "$SEED" --budgets "$BUDGETS_SINGLE"
          --warp-s "$WARP_S" --warp-k "$WARP_K" --lr-warp "$LR_WARP"
          --lambda-bd "$LAMBDA_BD" --lambda-joint "$LAMBDA_JOINT" --lambda-lpips "$LAMBDA_LPIPS")
[ "$DRY_RUN" = "1" ] && GEN_ARGS+=(--dry-run)
run_gen_configs_on_compute_node "${GEN_ARGS[@]}"

MISSING=0
for p in \
    "$EXP_BASE_REL/train_expert/${MODEL_FLAG}_1xs_${VARIANT_TAG}/seed${SEED}" \
    "$CELL_REL/gen_labels_trigger_joint" \
    "$CELL_REL/select_flips"; do
    [ -f "experiments/$p/config.toml" ] || { echo "[CONFIG] missing: experiments/$p/config.toml" >&2; MISSING=$((MISSING + 1)); }
done
for budget in $BUDGETS_SINGLE; do
    p="$CELL_REL/train_user_${budget}"
    [ -f "experiments/$p/config.toml" ] || { echo "[CONFIG] missing: experiments/$p/config.toml" >&2; MISSING=$((MISSING + 1)); }
done
if [ "$MISSING" -gt 0 ] && [ "$DRY_RUN" != "1" ]; then
    echo "[ABORT] $MISSING centralized config(s) missing after GEN step." >&2
    exit 1
fi

if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY-RUN] nothing submitted (configs were only dry-run-previewed above)."
    exit 0
fi

preflight_slurm || exit 1

# ---------------------------------------------------------------------------
# Shared trunk: EXPERT -> GEN
# ---------------------------------------------------------------------------
EXPERT_JOBS=("python run_experiment.py $EXP_BASE_REL/train_expert/${MODEL_FLAG}_1xs_${VARIANT_TAG}/seed${SEED}|${VARIANT_TAG}_expert_${MODEL_FLAG}_seed${SEED}")
TIME_PER_TASK="$EXPERT_TIME"
submit_job_pool_slurm EXPERT_JOBS "${VARIANT_TAG}_expert" "" || exit 1
expert_barrier=$(submit_barrier_slurm "${VARIANT_TAG}_barrier_expert" "afterok:$(join_job_ids "${SUBMITTED_JOB_IDS[@]}")") || exit 1
echo "[PHASE] EXPERT submitted -> barrier $expert_barrier"

GEN_JOBS=("python run_experiment.py $CELL_REL/gen_labels_trigger_joint|${VARIANT_TAG}_gen_${MODEL_FLAG}_${DATASET}_seed${SEED}")
TIME_PER_TASK="$GEN_TIME"
submit_job_pool_slurm GEN_JOBS "${VARIANT_TAG}_gen" "afterok:$expert_barrier" || exit 1
gen_barrier=$(submit_barrier_slurm "${VARIANT_TAG}_barrier_gen" "afterok:$(join_job_ids "${SUBMITTED_JOB_IDS[@]}")") || exit 1
echo "[PHASE] GEN submitted -> barrier $gen_barrier"

# ---------------------------------------------------------------------------
# Side job: automatic trigger visualization, right after GEN -- doesn't
# block/gate either branch below, just runs off gen_barrier in parallel.
# ---------------------------------------------------------------------------
VIZ_OUT="$CELL_REL/gen_labels_trigger_joint/trigger_preview.png"
viz_cmd="python scripts/show_results/visualize_trigger.py --exp-base experiments/$EXP_BASE_REL"
viz_cmd+=" --model-flag $MODEL_FLAG --dataset $DATASET --seed $SEED --trigger-type warp"
viz_cmd+=" --out experiments/$VIZ_OUT"
VIZ_JOBS=("$viz_cmd|${VARIANT_TAG}_viz_${MODEL_FLAG}_${DATASET}_seed${SEED}")
TIME_PER_TASK="00:05:00"
GPUS_PER_TASK=0 CPUS_PER_TASK=2 MEM_PER_TASK=4G submit_job_pool_slurm VIZ_JOBS "${VARIANT_TAG}_viz" "afterok:$gen_barrier" || exit 1
echo "[PHASE] trigger visualization submitted -> will land at experiments/$VIZ_OUT"

# ---------------------------------------------------------------------------
# Branch A: centralized (1 poisoned / 0 honest, mean -- a no-op aggregator)
# ---------------------------------------------------------------------------
CEN_FLIPS_JOBS=("python run_experiment.py $CELL_REL/select_flips|${VARIANT_TAG}_cen_flips_${MODEL_FLAG}_${DATASET}_seed${SEED}")
TIME_PER_TASK="$FLIPS_TIME"
GPUS_PER_TASK=0 submit_job_pool_slurm CEN_FLIPS_JOBS "${VARIANT_TAG}_cen_flips" "afterok:$gen_barrier" || exit 1
cen_flips_barrier=$(submit_barrier_slurm "${VARIANT_TAG}_barrier_cen_flips" "afterok:$(join_job_ids "${SUBMITTED_JOB_IDS[@]}")") || exit 1
echo "[PHASE] CENTRALIZED FLIPS submitted -> barrier $cen_flips_barrier"

CEN_USER_JOBS=()
for budget in $BUDGETS_SINGLE; do
    CEN_USER_JOBS+=("python run_experiment.py $CELL_REL/train_user_${budget}|${VARIANT_TAG}_cen_user_${MODEL_FLAG}_${DATASET}_seed${SEED}_${budget}")
done
TIME_PER_TASK="$USER_TIME"
submit_job_pool_slurm CEN_USER_JOBS "${VARIANT_TAG}_cen_user_seed${SEED}" "afterok:$cen_flips_barrier" || exit 1
echo "[PHASE] CENTRALIZED USER submitted (${#CEN_USER_JOBS[@]} jobs)."

# ---------------------------------------------------------------------------
# Branch B: federated deploy (multikrum, 3vs7 by default) of the SAME trigger.
# FED_GEN only writes federated_select_flips/federated_train_user once GEN's
# trigger .pt exists -- hence its own job, depending on gen_barrier, not run
# at submission time (the trigger doesn't exist yet on the login node).
# ---------------------------------------------------------------------------
fed_gen_cmd="python -m modules.generate_labels_trigger_joint.gen_configs --deploy-federated"
fed_gen_cmd+=" --trigger-type warp --exp-base-tag $VARIANT_TAG --model-flag $MODEL_FLAG --dataset $DATASET --seed $SEED"
fed_gen_cmd+=" --budgets '$BUDGETS_SINGLE' --deploy-num-poisoned $DEPLOY_NUM_POISONED"
fed_gen_cmd+=" --deploy-num-honests $DEPLOY_NUM_HONESTS --deploy-agg-method $DEPLOY_AGG_METHOD"
FED_GEN_JOBS=("$fed_gen_cmd|${VARIANT_TAG}_fed_gen_${MODEL_FLAG}_${DATASET}_seed${SEED}")
TIME_PER_TASK="$GEN_CONFIGS_TIME"
GPUS_PER_TASK=0 CPUS_PER_TASK=2 MEM_PER_TASK=4G submit_job_pool_slurm FED_GEN_JOBS "${VARIANT_TAG}_fed_gen" "afterok:$gen_barrier" || exit 1
fed_gen_barrier=$(submit_barrier_slurm "${VARIANT_TAG}_barrier_fed_gen" "afterok:$(join_job_ids "${SUBMITTED_JOB_IDS[@]}")") || exit 1
echo "[PHASE] FED_GEN submitted -> barrier $fed_gen_barrier"

FED_FLIPS_JOBS=("python run_experiment.py $FED_DEPLOY_REL/select_flips|${VARIANT_TAG}_fed_flips_${MODEL_FLAG}_${DATASET}_seed${SEED}")
TIME_PER_TASK="$FLIPS_TIME"
GPUS_PER_TASK=0 submit_job_pool_slurm FED_FLIPS_JOBS "${VARIANT_TAG}_fed_flips" "afterok:$fed_gen_barrier" || exit 1
fed_flips_barrier=$(submit_barrier_slurm "${VARIANT_TAG}_barrier_fed_flips" "afterok:$(join_job_ids "${SUBMITTED_JOB_IDS[@]}")") || exit 1
echo "[PHASE] FEDERATED FLIPS submitted -> barrier $fed_flips_barrier"

FED_USER_JOBS=()
for budget in $BUDGETS_SINGLE; do
    FED_USER_JOBS+=("python run_experiment.py $FED_DEPLOY_REL/train_user_${budget}|${VARIANT_TAG}_fed_user_${MODEL_FLAG}_${DATASET}_seed${SEED}_${budget}")
done
TIME_PER_TASK="$USER_TIME"
submit_job_pool_slurm FED_USER_JOBS "${VARIANT_TAG}_fed_user_seed${SEED}" "afterok:$fed_flips_barrier" || exit 1
echo "[PHASE] FEDERATED USER submitted (${#FED_USER_JOBS[@]} jobs)."

echo "[DONE] warp least-constrained campaign submitted: centralized AND federated ($DEPLOY_AGG_METHOD) branches both queued off the same trigger."
echo "[COMPARE] once finished, compare:"
echo "  trigger visual: experiments/$VIZ_OUT (ready right after GEN, before either branch finishes)"
for budget in $BUDGETS_SINGLE; do
    echo "  centralized: experiments/$CELL_REL/train_user_${budget}/logs/federated_train_user_metrics.json"
    echo "  federated:   experiments/$FED_DEPLOY_REL/train_user_${budget}/logs/federated_train_user_metrics.json"
done
