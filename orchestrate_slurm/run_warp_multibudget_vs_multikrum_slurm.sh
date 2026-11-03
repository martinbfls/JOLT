#!/bin/bash
###
# run_warp_multibudget_vs_multikrum_slurm.sh
#
# Re-deploy already-trained warp triggers (one or more VARIANT_TAGs, e.g.
# "warp_s8_lpips0p0 warp_s16_lpips0p0" from sweep_warp_min_perturbation_slurm.sh)
# across the FULL budget list, federated multikrum ONLY (3vs7 by default,
# the centralized 1vs0/mean branch is intentionally NOT re-deployed here)
# -- no EXPERT/GEN/select_flips re-run for budgets that already exist, no
# re-training of the trigger at all: this script only (re)writes
# federated_select_flips + federated_train_user configs for the requested
# budgets and submits those jobs, trusting that EXPERT -> GEN already
# produced each (VARIANT_TAG, SEED) cell's trigger on disk (see
# run_warp_test_vs_multikrum_slurm.sh, which is what originally generates a
# cell).
#
# Layout assumed (gen_configs.py --deploy-federated, --exp-base-tag=VARIANT_TAG):
#   experiments/federated_experiments/threat_model_direct_trigger_joint_<VARIANT_TAG>/
#       <MODEL_FLAG>/<DATASET>/1vs0/mean/seed<SEED>/
#           gen_labels_trigger_joint/trigger/...                              (reused, must already exist)
#           <DEPLOY_AGG_METHOD>_<P>vs<H>/select_flips/, train_user_<budget>/  (federated, (re)written here)
#
# Usage (from repo root, login-node shell):
#   SLURM_ACCOUNT=<account> ./orchestrate_slurm/run_warp_multibudget_vs_multikrum_slurm.sh
#   SLURM_ACCOUNT=<account> DRY_RUN=1 ./orchestrate_slurm/run_warp_multibudget_vs_multikrum_slurm.sh
#   VARIANT_TAGS="warp_s8_lpips0p0 warp_s16_lpips0p0" SEEDS="0 1 2 3 4" \
#       BUDGETS="0 150 300 500 1000 2000 2500 5000" \
#       SLURM_ACCOUNT=<account> ./orchestrate_slurm/run_warp_multibudget_vs_multikrum_slurm.sh
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
GEN_AGG_METHOD="${GEN_AGG_METHOD:-mean}"
VARIANT_TAGS="${VARIANT_TAGS:-warp_s8_lpips0p0 warp_s16_lpips0p0}"
SEEDS="${SEEDS:-0 1 2 3 4}"
BUDGETS="${BUDGETS:-0 150 300 500 1000 2000 2500 5000}"
DEPLOY_NUM_POISONED="${DEPLOY_NUM_POISONED:-3}"
DEPLOY_NUM_HONESTS="${DEPLOY_NUM_HONESTS:-7}"
DEPLOY_AGG_METHOD="${DEPLOY_AGG_METHOD:-multikrum}"
DRY_RUN="${DRY_RUN:-0}"

GEN_CONFIGS_TIME="${GEN_CONFIGS_TIME:-00:10:00}"
FLIPS_TIME="${FLIPS_TIME:-00:10:00}"
USER_TIME="${USER_TIME:-0-01:00:00}"

echo "[PLAN] model=$MODEL_FLAG dataset=$DATASET variant_tags=($VARIANT_TAGS) seeds=($SEEDS) budgets=($BUDGETS)"
echo "[PLAN] federated-only ($DEPLOY_AGG_METHOD, ${DEPLOY_NUM_POISONED}vs${DEPLOY_NUM_HONESTS}) -- centralized branch skipped"
echo "[PLAN] reusing existing triggers per (variant_tag, seed) -- no EXPERT/GEN re-run."

run_gen_configs_on_compute_node() {
    srun -A "$SLURM_ACCOUNT" -p "$SLURM_PARTITION" \
        --cpus-per-task=2 --mem=4G --time="$GEN_CONFIGS_TIME" \
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

CELL_RELS=()
FED_DEPLOY_RELS=()
MISSING=0
for tag in $VARIANT_TAGS; do
    EXP_BASE_REL="federated_experiments/threat_model_direct_trigger_joint_${tag}"
    for seed in $SEEDS; do
        CELL_REL="$EXP_BASE_REL/$MODEL_FLAG/$DATASET/1vs0/$GEN_AGG_METHOD/seed$seed"
        FED_DEPLOY_REL="$CELL_REL/${DEPLOY_AGG_METHOD}_${DEPLOY_NUM_POISONED}vs${DEPLOY_NUM_HONESTS}"
        CELL_RELS+=("$CELL_REL")
        FED_DEPLOY_RELS+=("$FED_DEPLOY_REL")

        if [ ! -f "experiments/$CELL_REL/gen_labels_trigger_joint/config.toml" ]; then
            echo "[ABORT] no existing cell at experiments/$CELL_REL/gen_labels_trigger_joint -- " \
                 "run run_warp_test_vs_multikrum_slurm.sh with VARIANT_TAG=$tag SEED=$seed first." >&2
            exit 1
        fi

        echo "[GEN] federated ($DEPLOY_AGG_METHOD) select_flips/train_user for tag=$tag seed=$seed, budgets=($BUDGETS)..."
        FED_GEN_ARGS=(--deploy-federated --trigger-type warp --exp-base-tag "$tag"
                      --model-flag "$MODEL_FLAG" --dataset "$DATASET" --seed "$seed"
                      --gen-agg-method "$GEN_AGG_METHOD" --budgets "$BUDGETS"
                      --deploy-num-poisoned "$DEPLOY_NUM_POISONED" --deploy-num-honests "$DEPLOY_NUM_HONESTS"
                      --deploy-agg-method "$DEPLOY_AGG_METHOD")
        [ "$DRY_RUN" = "1" ] && FED_GEN_ARGS+=(--dry-run)
        run_gen_configs_on_compute_node "${FED_GEN_ARGS[@]}"

        for budget in $BUDGETS; do
            p="$FED_DEPLOY_REL/train_user_${budget}"
            [ -f "experiments/$p/config.toml" ] || { echo "[CONFIG] missing: experiments/$p/config.toml" >&2; MISSING=$((MISSING + 1)); }
        done
    done
done
if [ "$MISSING" -gt 0 ] && [ "$DRY_RUN" != "1" ]; then
    echo "[ABORT] $MISSING train_user config(s) missing after GEN step." >&2
    exit 1
fi

if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY-RUN] nothing submitted (configs were only dry-run-previewed above)."
    exit 0
fi

preflight_slurm || exit 1

# ---------------------------------------------------------------------------
# federated select_flips, then federated train_user for every (tag, seed,
# budget) -- no EXPERT/GEN dependency since the trigger already exists on
# disk, and no centralized branch at all.
# ---------------------------------------------------------------------------
FLIPS_JOBS=()
idx=0
for tag in $VARIANT_TAGS; do
    for seed in $SEEDS; do
        FED_DEPLOY_REL="${FED_DEPLOY_RELS[$idx]}"
        FLIPS_JOBS+=("python run_experiment.py $FED_DEPLOY_REL/select_flips|${tag}_fed_flips_${MODEL_FLAG}_${DATASET}_seed${seed}")
        idx=$((idx + 1))
    done
done
TIME_PER_TASK="$FLIPS_TIME"
GPUS_PER_TASK=0 submit_job_pool_slurm FLIPS_JOBS "warp_mb_flips" "" || exit 1
flips_barrier=$(submit_barrier_slurm "warp_mb_barrier_flips" "afterok:$(join_job_ids "${SUBMITTED_JOB_IDS[@]}")") || exit 1
echo "[PHASE] FLIPS submitted (${#FLIPS_JOBS[@]} jobs) -> barrier $flips_barrier"

USER_JOBS=()
idx=0
for tag in $VARIANT_TAGS; do
    for seed in $SEEDS; do
        FED_DEPLOY_REL="${FED_DEPLOY_RELS[$idx]}"
        for budget in $BUDGETS; do
            USER_JOBS+=("python run_experiment.py $FED_DEPLOY_REL/train_user_${budget}|${tag}_fed_user_${MODEL_FLAG}_${DATASET}_seed${seed}_${budget}")
        done
        idx=$((idx + 1))
    done
done
TIME_PER_TASK="$USER_TIME"
submit_job_pool_slurm USER_JOBS "warp_mb_user" "afterok:$flips_barrier" || exit 1
echo "[PHASE] USER submitted (${#USER_JOBS[@]} jobs)."

echo "[DONE] multikrum-only multi-budget campaign submitted for variant_tags=($VARIANT_TAGS), seeds=($SEEDS), budgets=($BUDGETS)."
echo "[COMPARE] once finished, per (tag, seed, budget):"
echo "  experiments/<tag_cell>/${DEPLOY_AGG_METHOD}_${DEPLOY_NUM_POISONED}vs${DEPLOY_NUM_HONESTS}/train_user_<budget>/logs/federated_train_user_metrics.json"
