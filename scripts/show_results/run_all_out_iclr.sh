#!/bin/bash
###
# run_all_out_iclr.sh
#
# Runs every show_results/*.py script that supports pointing at an arbitrary
# results root, against experiments/out_iclr, in one command, as a SINGLE
# CPU-only Slurm job (these are plotting/table scripts -- no GPU needed):
#   - plot_paper_main_campaign_user.py  (CTA-vs-ASR trade-off plots + LaTeX tables)
#   - plot_asr_per_seed.py              (per-seed ASR/CTA scatter, variance visible)
#   - plot_flip_matrix.py               (flip-selection confusion matrices)
#   - plot_trigger_grid.py              (clean-vs-poisoned trigger grid, one row per tag)
#
# NOT run here (no --root/--jolt-root override exists, so they can only read
# their own hardcoded EXP_BASE -- not out_iclr):
#   - plot_training_curves.py
#   - show_results.py (the original, pre-paper-main-campaign table/plot script)
# NOT run here (specifically about krum_full/multikrum_full, which this batch
# was told to skip):
#   - summarize_full_grad_vs_layerwise.py -- run it by hand if you need it,
#     e.g.: python scripts/show_results/summarize_full_grad_vs_layerwise.py \
#       --exp-base experiments/out_iclr --model-flag r32p --dataset cifar --tag baseline
# Comparison-only, needs explicit per-case args, not run here:
#   - compare_additive_warp.py
#   - visualize_trigger.py (already invoked inline per-cell by the orchestrate_slurm scripts)
#
# Usage (from the RDHPC login node, repo root): submits one CPU-only sbatch
# job and returns immediately; see $LOG_DIR/show_results_out_iclr.log for
# output.
#   SLURM_ACCOUNT=<account> ./scripts/show_results/run_all_out_iclr.sh
#   MODEL=r32p DATASETS="cifar" TAGS="baseline eps_16_255" SLURM_ACCOUNT=<account> \
#       ./scripts/show_results/run_all_out_iclr.sh
#
# NOT maple_cpu/maple: those nodes are aarch64 (GH200/Grace) and
# $CONDA_ENV (flip_x86_64) only ships x86_64 binaries -- "cannot execute
# binary file: Exec format error" there, not a partition-access problem.
# `pine` is x86_64, no-GPU-gated (GRES=null), reachable directly from the
# RDHPC login node, with plenty of idle nodes -- see `sinfo`.
###

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
export BASE_DIR
cd "$BASE_DIR"

SLURM_PARTITION="${SLURM_PARTITION:-pine}"
GPUS_PER_TASK=0
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM_PER_TASK="${MEM_PER_TASK:-16G}"
TIME_PER_TASK="${TIME_PER_TASK:-00:30:00}"

# shellcheck source=../../orchestrate_slurm/slurm_lib.sh
source "$BASE_DIR/orchestrate_slurm/slurm_lib.sh"

JOLT_ROOT="${JOLT_ROOT:-experiments/out_iclr}"
MODEL="${MODEL:-r32p}"
DATASETS="${DATASETS:-cifar}"
TAGS="${TAGS:-baseline eps_16_255}"

if [ ! -d "$JOLT_ROOT" ]; then
    echo "[ABORT] $JOLT_ROOT not found (rsync/mount it locally, or run this on the cluster)." >&2
    exit 1
fi

echo "[PLAN] partition=$SLURM_PARTITION jolt_root=$JOLT_ROOT model=$MODEL datasets=($DATASETS) tags=($TAGS)"

preflight_slurm || exit 1

CMD="set -euo pipefail
echo '=== [1/4] plot_paper_main_campaign_user.py (CTA-vs-ASR trade-off + LaTeX tables) ==='
python scripts/show_results/plot_paper_main_campaign_user.py \
    --model '$MODEL' --datasets $DATASETS --tags $TAGS \
    --jolt-root '$JOLT_ROOT' \
    --out-dir ./plots_paper_main_campaign_user \
    --csv-dir ./results_csv_paper_main_campaign_user \
    --table-dir ./tables_paper_main_campaign_user

echo '=== [2/4] plot_asr_per_seed.py (per-seed ASR/CTA scatter) ==='
python scripts/show_results/plot_asr_per_seed.py \
    --campaign jolt --root '$JOLT_ROOT' --model '$MODEL' \
    --datasets $DATASETS --tags $TAGS \
    --out-dir ./plots_per_seed

echo '=== [3/4] plot_flip_matrix.py (flip-selection confusion matrices) ==='
python scripts/show_results/plot_flip_matrix.py \
    --campaign jolt --root '$JOLT_ROOT' --model '$MODEL' \
    --datasets $DATASETS --tags $TAGS \
    --out-dir ./plots_flip_matrix

echo '=== [4/4] plot_trigger_grid.py (clean-vs-poisoned trigger grid) ==='
python scripts/show_results/plot_trigger_grid.py \
    --models '$MODEL' --datasets $DATASETS --tags $TAGS \
    --jolt-root '$JOLT_ROOT' \
    --out-dir ./plots_trigger_grid

echo '[DONE] outputs under ./plots_paper_main_campaign_user, ./plots_per_seed, ./plots_flip_matrix, ./plots_trigger_grid'
echo '       (+ ./results_csv_paper_main_campaign_user, ./tables_paper_main_campaign_user)'
"

jobid=$(submit_job_slurm "$CMD" "show_results_out_iclr" "" "$SLURM_PARTITION") || exit 1
echo "[SUBMIT] job $jobid submitted on partition $SLURM_PARTITION -> log: $LOG_DIR/show_results_out_iclr.log"
