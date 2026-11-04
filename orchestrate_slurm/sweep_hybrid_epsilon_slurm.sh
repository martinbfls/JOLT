#!/bin/bash
###
# sweep_hybrid_epsilon_slurm.sh
#
# One-command hybrid-trigger campaign: the whole epsilon x seed grid, every budget, both
# deployments, and the comparison figure, from a single invocation.
#
# For each (epsilon, warp_s, seed) cell it calls run_warp_test_vs_multikrum_slurm.sh as its own
# child process (own VARIANT_TAG, own SEED, own experiments/out/checkpoints subtree, own Slurm
# job names), with TRIGGER_TYPE=hybrid -- so each cell is a full
#
#   EXPERT -> GEN (hybrid) -+-> CENTRALIZED FLIPS -> CENTRALIZED USER (1 poisoned / 0 honest)
#                           +-> FED_GEN -> FEDERATED FLIPS -> FEDERATED USER (multikrum 3vs7)
#
# DAG over EVERY budget in BUDGETS at once. There is no separate multi-budget re-deploy step to
# run afterwards: that script (run_warp_multibudget_vs_multikrum_slurm.sh) exists to EXTEND a
# cell that was already generated at one budget, which is not the case here.
#
# Once every cell has finished, a final Slurm job (gated on all of them via --dependency) runs
# scripts/show_results/run_all_campaign.sh over the hybrid cells PLUS REFERENCE_TAGS (the
# existing warp runs, by default), producing both the ASR/CTA-vs-budget figures and the
# per-epoch dynamics analysis, on the same seeds.
#
# The final figure job goes to the campaign's own partition (SLURM_PARTITION); override it with
# SUMMARY_PARTITION if you have a CPU-only one that accepts sbatch from this login node. If it
# cannot be submitted at all, the campaign is unaffected and the plotting command is printed for
# you to run by hand -- it needs no Slurm.
#
# Usage (from repo root, login-node shell -- rdhpc-n1, not maple-1: the conda env is x86_64):
#   SLURM_ACCOUNT=<account> ./orchestrate_slurm/sweep_hybrid_epsilon_slurm.sh
#   SLURM_ACCOUNT=<account> DRY_RUN=1 ./orchestrate_slurm/sweep_hybrid_epsilon_slurm.sh
#   EPSILON_VALUES="0.5 0.25 0.0627" SEEDS="0 1 2" WARP_S_VALUES="8.0" \
#       SLURM_ACCOUNT=<account> ./orchestrate_slurm/sweep_hybrid_epsilon_slurm.sh
#
# Each cell lands at (sanitized epsilon/warp_s, e.g. hybrid_eps0p0627_s8p0):
#   experiments/federated_experiments/threat_model_direct_trigger_joint_hybrid_eps<E>_s<S>/
#       r32p/cifar/1vs0/mean/seed<SEED>/train_user_<budget>/logs/federated_train_user_metrics.json          (centralized)
#       r32p/cifar/1vs0/mean/seed<SEED>/multikrum_3vs7/train_user_<budget>/logs/federated_train_user_metrics.json  (federated)
###

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
export BASE_DIR
cd "$BASE_DIR"

# The additive half's Linf budget, in [0, 1] intensity. 0.5 is a loose reference point and
# 0.0627 is 16/255, the stealth budget the paper campaign's eps_16_255 tag uses -- the pair
# answers "does the mechanism work at all" and "does it survive a budget worth calling stealthy"
# in one campaign. The warp half carries the rest of the perturbation at WARP_S.
read -ra EPSILON_ARR <<< "${EPSILON_VALUES:-0.5 0.0627}"
read -ra WARP_S_ARR <<< "${WARP_S_VALUES:-8.0}"
# Three seeds by default: each extra seed is another full EXPERT+GEN per epsilon.
read -ra SEEDS_ARR <<< "${SEEDS:-0 1 2}"

# Exported so each run_warp_test_vs_multikrum_slurm.sh child inherits them -- same
# least-constrained/rebalanced settings as the warp sweep, so the two are comparable.
export TRIGGER_TYPE=hybrid
export LAMBDA_BD="${LAMBDA_BD:-1.0}"
export LAMBDA_JOINT="${LAMBDA_JOINT:-1.0}"
export LAMBDA_LPIPS="${LAMBDA_LPIPS:-0.0}"
export WARP_K="${WARP_K:-4}"
export LR_WARP="${LR_WARP:-0.1}"
export LR_DELTA="${LR_DELTA:-}"   # empty leaves gen_configs.py's own default (1e-2)
# 1 = federated victim with the same optimizer-step count as the centralized one
export SPLIT_BATCH="${SPLIT_BATCH:-0}"

MODEL_FLAG="${MODEL_FLAG:-r32p}"
DATASET="${DATASET:-cifar}"
BUDGETS_ALL="${BUDGETS:-0 150 300 500 1000 2000 2500 5000}"
DEPLOY_AGG_METHOD="${DEPLOY_AGG_METHOD:-multikrum}"
DEPLOY_NUM_POISONED="${DEPLOY_NUM_POISONED:-3}"
DEPLOY_NUM_HONESTS="${DEPLOY_NUM_HONESTS:-7}"
DRY_RUN="${DRY_RUN:-0}"

# Already-generated runs to draw alongside the hybrid cells in the final figure; they are only
# READ, never re-run. Set to "" to plot the hybrid cells alone.
read -ra REFERENCE_TAGS_ARR <<< "${REFERENCE_TAGS:-warp_s8_lpips0p0 warp_s16_lpips0p0}"

SUMMARY_OUT="${SUMMARY_OUT:-experiments/federated_experiments/campaign_results}"
CAMPAIGN="${CAMPAIGN:-hybrid_epsilon_sweep}"
# Partition for the final (CPU-only) plotting job. Defaults to the campaign's own partition once
# slurm_lib.sh is sourced: `pine` accepts srun from an RDHPC node but REFUSES sbatch from one
# ("You must submit your job from that cluster's login node instead"), and the figure job is
# submitted with sbatch like every other job here.
SUMMARY_PARTITION="${SUMMARY_PARTITION:-}"

read -ra _BUDGETS_ARR <<< "$BUDGETS_ALL"
n_cells=$((${#EPSILON_ARR[@]} * ${#WARP_S_ARR[@]} * ${#SEEDS_ARR[@]}))
echo "[SWEEP] trigger_type=hybrid epsilons: ${EPSILON_ARR[*]}"
echo "[SWEEP] warp_s: ${WARP_S_ARR[*]}   seeds: ${SEEDS_ARR[*]}   budgets: ${_BUDGETS_ARR[*]}"
echo "[SWEEP] lambda_bd=$LAMBDA_BD lambda_joint=$LAMBDA_JOINT lambda_lpips=$LAMBDA_LPIPS lr_warp=$LR_WARP lr_delta=${LR_DELTA:-<default>}"
echo "[SWEEP] $n_cells cells, each its own EXPERT->GEN->(centralized + $DEPLOY_AGG_METHOD) DAG over all budgets"
echo "[SWEEP] => $n_cells EXPERT + $n_cells GEN jobs, and $((n_cells * ${#_BUDGETS_ARR[@]} * 2)) train_user jobs"
echo "[SWEEP] reference tags in the final figure (read-only): ${REFERENCE_TAGS_ARR[*]:-<none>}"

sanitize() { echo "$1" | tr '.' 'p' | tr -d '-'; }

# shellcheck source=slurm_lib.sh
source "$SCRIPT_DIR/slurm_lib.sh"

SUMMARY_PARTITION="${SUMMARY_PARTITION:-$SLURM_PARTITION}"

ALL_FINAL_JOB_IDS=()
TAGS=()

for epsilon in "${EPSILON_ARR[@]}"; do
    for warp_s in "${WARP_S_ARR[@]}"; do
        tag="hybrid_eps$(sanitize "$epsilon")_s$(sanitize "$warp_s")"
        TAGS+=("$tag")
        for seed in "${SEEDS_ARR[@]}"; do
            echo "[SWEEP] submitting cell: epsilon=$epsilon warp_s=$warp_s seed=$seed -> VARIANT_TAG=$tag"
            VARIANT_TAG="$tag" EPSILON="$epsilon" WARP_S="$warp_s" SEED="$seed" \
            BUDGETS="$BUDGETS_ALL" DEPLOY_AGG_METHOD="$DEPLOY_AGG_METHOD" \
            DEPLOY_NUM_POISONED="$DEPLOY_NUM_POISONED" DEPLOY_NUM_HONESTS="$DEPLOY_NUM_HONESTS" \
            MODEL_FLAG="$MODEL_FLAG" DATASET="$DATASET" DRY_RUN="$DRY_RUN" \
                "$SCRIPT_DIR/run_warp_test_vs_multikrum_slurm.sh"

            # The child's own submit_job_pool_slurm calls leave each phase's job ids in
            # $LOG_DIR/jobids_<phase>.txt, and its final phases are named with this cell's
            # tag+seed -- so they can be collected here without the child returning anything.
            if [ "$DRY_RUN" != "1" ]; then
                for id_file in "${LOG_DIR:-$PWD/logs_slurm}/jobids_${tag}_cen_user_seed${seed}.txt" \
                               "${LOG_DIR:-$PWD/logs_slurm}/jobids_${tag}_fed_user_seed${seed}.txt"; do
                    [ -f "$id_file" ] && while read -r jid _; do ALL_FINAL_JOB_IDS+=("$jid"); done < "$id_file"
                done
            fi
        done
    done
done

echo "[SWEEP] all $n_cells cells submitted (or dry-run-previewed)."

plot_cmd="SUBMIT=0 VARIANT_TAGS='${TAGS[*]}' REFERENCE_TAGS='${REFERENCE_TAGS_ARR[*]:-}'"
plot_cmd+=" MODEL_FLAG=$MODEL_FLAG DATASET=$DATASET"
plot_cmd+=" SEEDS='${SEEDS_ARR[*]}' BUDGETS='${_BUDGETS_ARR[*]}'"
plot_cmd+=" DEPLOY_AGG_METHOD=$DEPLOY_AGG_METHOD"
plot_cmd+=" DEPLOY_NUM_POISONED=$DEPLOY_NUM_POISONED DEPLOY_NUM_HONESTS=$DEPLOY_NUM_HONESTS"
plot_cmd+=" OUT_ROOT=$SUMMARY_OUT CAMPAIGN='$CAMPAIGN' ./scripts/show_results/run_all_campaign.sh"

if [ "$DRY_RUN" = "1" ]; then
    echo "[SWEEP] DRY_RUN=1: skipping the final figure job. It would have run:"
    echo "  $plot_cmd"
    exit 0
fi

if [ ${#ALL_FINAL_JOB_IDS[@]} -eq 0 ]; then
    echo "[SWEEP] no final job ids collected -- skipping the automatic figure job." >&2
    echo "[SWEEP] run it by hand once every cell has finished:"
    echo "  $plot_cmd"
    exit 0
fi

preflight_slurm || exit 1
plot_dep="afterok:$(join_job_ids "${ALL_FINAL_JOB_IDS[@]}")"
# Never `exit 1` on a failed figure submission: the campaign is already queued at this point, so
# losing the command the user would need afterwards is the worst possible outcome.
if plot_job=$(GPUS_PER_TASK=0 CPUS_PER_TASK=4 MEM_PER_TASK=8G TIME_PER_TASK=00:15:00 \
        submit_job_slurm "$plot_cmd" "hybrid_sweep_results" "$plot_dep" "$SUMMARY_PARTITION"); then
    echo "[SWEEP] figure job $plot_job submitted on $SUMMARY_PARTITION, depends on ${#ALL_FINAL_JOB_IDS[@]} final jobs (afterok)"
    echo "[SWEEP] once it completes: $SUMMARY_OUT/$CAMPAIGN/$MODEL_FLAG/$DATASET/${DEPLOY_AGG_METHOD}_${DEPLOY_NUM_POISONED}vs${DEPLOY_NUM_HONESTS}/"
    echo "[SWEEP] a cell that fails leaves the figure job pending -- scancel it and run this by hand:"
else
    echo "[SWEEP] the figure job could NOT be submitted -- the ${#ALL_FINAL_JOB_IDS[@]} campaign jobs above are unaffected." >&2
    echo "[SWEEP] run this once every cell has finished (no Slurm needed, it is a plotting script):"
fi
echo "  $plot_cmd"
