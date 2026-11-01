#!/bin/bash
###
# sweep_warp_min_perturbation_slurm.sh
#
# One-command sweep to find the MINIMAL warp perturbation (warp_s) that still
# backdoors multikrum, optionally cross-checking LPIPS regularization at each
# point, repeated over several seeds so the per-cell variance is visible (a
# single seed wasn't enough to tell signal from noise). For each (warp_s,
# lambda_lpips, seed) cell, calls run_warp_test_vs_multikrum_slurm.sh as its
# own child process (own VARIANT_TAG, own SEED, own experiments/out/checkpoints
# subtree, own Slurm job names) -- the two scripts stay independent, this one
# is just a driver.
#
# Once every cell has finished, a final Slurm job (gated on all of them via
# --dependency) runs scripts/show_results/summarize_warp_sweep.py and writes a
# mean +/- std (across seeds) table comparing centralized vs federated ASR/CTA
# for every (warp_s, lambda_lpips) cell -- see SUMMARY_OUT below.
#
# Usage (from repo root, login-node shell):
#   SLURM_ACCOUNT=<account> ./orchestrate_slurm/sweep_warp_min_perturbation_slurm.sh
#   SLURM_ACCOUNT=<account> DRY_RUN=1 ./orchestrate_slurm/sweep_warp_min_perturbation_slurm.sh
#   WARP_S_VALUES="2 4 8 16" LAMBDA_LPIPS_VALUES="0.0 0.1" SEEDS="0 1 2 3 4" \
#       SLURM_ACCOUNT=<account> ./orchestrate_slurm/sweep_warp_min_perturbation_slurm.sh
#
# Each cell's results, once finished, land at (sanitized warp_s/lambda_lpips, e.g.
# warp_s4p0_lpips0p1):
#   experiments/federated_experiments/threat_model_direct_trigger_joint_warp_s<S>_lpips<L>/
#       r32p/cifar/1vs0/mean/seed<SEED>/train_user_<budget>/logs/federated_train_user_metrics.json          (centralized)
#       r32p/cifar/1vs0/mean/seed<SEED>/multikrum_3vs7/train_user_<budget>/logs/federated_train_user_metrics.json  (federated)
###

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Bisecting-ish grid around the known-working WARP_S=16 (ASR confirmed to pass multikrum) down to
# the original WARP_S=1.0 default, to locate the minimal perturbation that still backdoors it.
read -ra WARP_S_ARR <<< "${WARP_S_VALUES:-16 12 8 6 4 2 1}"
read -ra LPIPS_ARR <<< "${LAMBDA_LPIPS_VALUES:-0.0}"
# Several seeds per cell by default -- a single seed's ASR swing was too close to the
# centralized/federated gap to tell apart from noise.
read -ra SEEDS_ARR <<< "${SEEDS:-0 1 2 3 4}"

# Exported so each run_warp_test_vs_multikrum_slurm.sh child inherits them (its own defaults
# are 1.0/1.0 if left unset, matching this sweep's "least-constrained + lambda_bd=1.0" baseline).
export LAMBDA_BD="${LAMBDA_BD:-1.0}"
export LAMBDA_JOINT="${LAMBDA_JOINT:-1.0}"

MODEL_FLAG="${MODEL_FLAG:-r32p}"
DATASET="${DATASET:-cifar}"
BUDGETS_SINGLE="${BUDGETS:-1500}"
DEPLOY_AGG_METHOD="${DEPLOY_AGG_METHOD:-multikrum}"
DEPLOY_NUM_POISONED="${DEPLOY_NUM_POISONED:-3}"
DEPLOY_NUM_HONESTS="${DEPLOY_NUM_HONESTS:-7}"
DRY_RUN="${DRY_RUN:-0}"
SUMMARY_OUT="${SUMMARY_OUT:-experiments/federated_experiments/warp_sweep_summary}"

echo "[SWEEP] warp_s values: ${WARP_S_ARR[*]}"
echo "[SWEEP] lambda_lpips values: ${LPIPS_ARR[*]}"
echo "[SWEEP] seeds: ${SEEDS_ARR[*]}"
echo "[SWEEP] ${#WARP_S_ARR[@]} x ${#LPIPS_ARR[@]} x ${#SEEDS_ARR[@]} = $((${#WARP_S_ARR[@]} * ${#LPIPS_ARR[@]} * ${#SEEDS_ARR[@]})) cells, each its own EXPERT->GEN->(centralized + multikrum) DAG"

sanitize() { echo "$1" | tr '.' 'p' | tr -d '-'; }

# shellcheck source=slurm_lib.sh
source "$SCRIPT_DIR/slurm_lib.sh"

ALL_FINAL_JOB_IDS=()
TAGS=()

for warp_s in "${WARP_S_ARR[@]}"; do
    for lpips in "${LPIPS_ARR[@]}"; do
        tag="warp_s$(sanitize "$warp_s")_lpips$(sanitize "$lpips")"
        TAGS+=("$tag")
        for seed in "${SEEDS_ARR[@]}"; do
            echo "[SWEEP] submitting cell: warp_s=$warp_s lambda_lpips=$lpips seed=$seed -> VARIANT_TAG=$tag"
            VARIANT_TAG="$tag" WARP_S="$warp_s" LAMBDA_LPIPS="$lpips" SEED="$seed" \
                "$SCRIPT_DIR/run_warp_test_vs_multikrum_slurm.sh"

            # run_warp_test_vs_multikrum_slurm.sh's own submit_job_pool_slurm calls leave each
            # phase's job ids in $LOG_DIR/jobids_<phase>.txt; the final phases of each branch
            # are named with this cell's tag+seed (see run_warp_test_vs_multikrum_slurm.sh), so
            # they can be collected here without the child process returning anything.
            if [ "$DRY_RUN" != "1" ]; then
                for id_file in "${LOG_DIR:-$PWD/logs_slurm}/jobids_${tag}_cen_user_seed${seed}.txt" \
                               "${LOG_DIR:-$PWD/logs_slurm}/jobids_${tag}_fed_user_seed${seed}.txt"; do
                    [ -f "$id_file" ] && while read -r jid _; do ALL_FINAL_JOB_IDS+=("$jid"); done < "$id_file"
                done
            fi
        done
    done
done

echo "[SWEEP] all cells submitted (or dry-run-previewed)."

if [ "$DRY_RUN" = "1" ]; then
    echo "[SWEEP] DRY_RUN=1: skipping the final summary job."
    exit 0
fi

if [ ${#ALL_FINAL_JOB_IDS[@]} -eq 0 ]; then
    echo "[SWEEP] no final job ids collected -- skipping the automatic summary job." >&2
    echo "[SWEEP] run it by hand once every cell has finished:"
    echo "  python scripts/show_results/summarize_warp_sweep.py --model-flag $MODEL_FLAG --dataset $DATASET \\"
    echo "      --budget $BUDGETS_SINGLE --seeds ${SEEDS_ARR[*]} --deploy-agg-method $DEPLOY_AGG_METHOD \\"
    echo "      --deploy-num-poisoned $DEPLOY_NUM_POISONED --deploy-num-honests $DEPLOY_NUM_HONESTS --out $SUMMARY_OUT"
    exit 0
fi

preflight_slurm || exit 1
summary_dep="afterok:$(join_job_ids "${ALL_FINAL_JOB_IDS[@]}")"
summary_cmd="python scripts/show_results/summarize_warp_sweep.py"
summary_cmd+=" --model-flag $MODEL_FLAG --dataset $DATASET --budget $BUDGETS_SINGLE"
summary_cmd+=" --seeds ${SEEDS_ARR[*]} --deploy-agg-method $DEPLOY_AGG_METHOD"
summary_cmd+=" --deploy-num-poisoned $DEPLOY_NUM_POISONED --deploy-num-honests $DEPLOY_NUM_HONESTS"
summary_cmd+=" --out $SUMMARY_OUT"
GPUS_PER_TASK=0 CPUS_PER_TASK=2 MEM_PER_TASK=4G TIME_PER_TASK=00:10:00 \
    summary_job=$(submit_job_slurm "$summary_cmd" "warp_sweep_summary" "$summary_dep") || exit 1
echo "[SWEEP] summary job $summary_job submitted, depends on ${#ALL_FINAL_JOB_IDS[@]} final jobs (afterok)"
echo "[SWEEP] once it completes: ${SUMMARY_OUT}.md and ${SUMMARY_OUT}.csv"
