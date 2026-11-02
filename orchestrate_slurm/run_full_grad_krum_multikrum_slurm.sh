#!/bin/bash
###
# run_full_grad_krum_multikrum_slurm.sh
#
# One-command re-deploy of full-gradient Krum/Multi-Krum (agg_method = krum_full /
# multikrum_full, see modules/base_utils/robust_agg.py) on an ALREADY-FINISHED
# paper_main_campaign cell, across multiple seeds -- no EXPERT/GEN/FLIPS re-run:
# it reuses the existing trigger (gen_labels_trigger_joint) and federated
# select_flips that the per-layer krum/multikrum runs already used.
#
# Layout assumed (gen_configs_paper_main_campaign.py's, EXP_BASE overridable):
#   experiments/<EXP_BASE_REL>/<MODEL_FLAG>/<DATASET>/<TAG>/seed<seed>/
#       gen_labels_trigger_joint/trigger/...                       (reused, must already exist)
#       federated_3vs7/select_flips/...                            (reused, must already exist)
#       federated_3vs7/krum_full/train_user_<budget>/...           (written by this script)
#       federated_3vs7/multikrum_full/train_user_<budget>/...      (written by this script)
#
# Usage (from repo root, login-node shell; out_iclr must already live under experiments/):
#   SLURM_ACCOUNT=<account> ./orchestrate_slurm/run_full_grad_krum_multikrum_slurm.sh
#   SLURM_ACCOUNT=<account> DRY_RUN=1 ./orchestrate_slurm/run_full_grad_krum_multikrum_slurm.sh
#   SEEDS="0 1 2" SLURM_ACCOUNT=<account> ./orchestrate_slurm/run_full_grad_krum_multikrum_slurm.sh
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
TAG="${TAG:-baseline}"
EXP_BASE_REL="${EXP_BASE_REL:-out_iclr}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
AGG_METHODS="${AGG_METHODS:-krum_full multikrum_full}"
BUDGETS="${BUDGETS:-0 150 300 500 1000 2000 2500 5000}"
DRY_RUN="${DRY_RUN:-0}"

GEN_CONFIGS_TIME="${GEN_CONFIGS_TIME:-00:10:00}"
USER_TIME="${USER_TIME:-0-01:00:00}"

CELL_BASE_REL="$EXP_BASE_REL/$MODEL_FLAG/$DATASET/$TAG"

echo "[PLAN] exp_base=experiments/$EXP_BASE_REL model=$MODEL_FLAG dataset=$DATASET tag=$TAG"
echo "[PLAN] seeds=($SEEDS) agg_methods=($AGG_METHODS) budgets=($BUDGETS)"
echo "[PLAN] reusing existing trigger + federated_3vs7/select_flips per seed -- no EXPERT/GEN/FLIPS re-run."

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
            python -m modules.generate_labels_trigger_joint.gen_configs_full_grad_deploy $*
        "
}

echo "[GEN] writing krum_full/multikrum_full train_user configs for seeds ($SEEDS)..."
GEN_ARGS=(--exp-base "experiments/$EXP_BASE_REL" --model-flag "$MODEL_FLAG" --dataset "$DATASET" --tag "$TAG"
          --seeds $SEEDS --agg-methods $AGG_METHODS --budgets $BUDGETS)
[ "$DRY_RUN" = "1" ] && GEN_ARGS+=(--dry-run)
run_gen_configs_on_compute_node "${GEN_ARGS[@]}"

MISSING=0
for seed in $SEEDS; do
    for agg in $AGG_METHODS; do
        for budget in $BUDGETS; do
            p="$CELL_BASE_REL/seed${seed}/federated_3vs7/${agg}/train_user_${budget}"
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
# train_user jobs: one per (seed, agg_method, budget), no dependency -- the
# trigger and select_flips they read are already on disk.
# ---------------------------------------------------------------------------
USER_JOBS=()
for seed in $SEEDS; do
    for agg in $AGG_METHODS; do
        for budget in $BUDGETS; do
            cell="$CELL_BASE_REL/seed${seed}/federated_3vs7/${agg}/train_user_${budget}"
            USER_JOBS+=("python run_experiment.py $cell|fg_user_${MODEL_FLAG}_${DATASET}_${agg}_seed${seed}_${budget}")
        done
    done
done
TIME_PER_TASK="$USER_TIME"
submit_job_pool_slurm USER_JOBS "fg_user" "" || exit 1
user_barrier=$(submit_barrier_slurm "fg_barrier_user" "afterok:$(join_job_ids "${SUBMITTED_JOB_IDS[@]}")") || exit 1
echo "[PHASE] USER submitted (${#USER_JOBS[@]} jobs) -> barrier $user_barrier"

# ---------------------------------------------------------------------------
# Summary: CTA/ASR mean +/- std across seeds, krum vs krum_full vs multikrum
# vs multikrum_full, once every train_user job above has finished.
# ---------------------------------------------------------------------------
summary_cmd="python scripts/show_results/summarize_full_grad_vs_layerwise.py"
summary_cmd+=" --exp-base experiments/$EXP_BASE_REL --model-flag $MODEL_FLAG --dataset $DATASET --tag $TAG"
summary_cmd+=" --seeds $SEEDS --budgets $BUDGETS"
summary_cmd+=" --out experiments/$CELL_BASE_REL/full_grad_vs_layerwise_summary"
SUMMARY_JOBS=("$summary_cmd|fg_summary_${MODEL_FLAG}_${DATASET}")
TIME_PER_TASK="00:10:00"
GPUS_PER_TASK=0 CPUS_PER_TASK=2 MEM_PER_TASK=4G submit_job_pool_slurm SUMMARY_JOBS "fg_summary" "afterok:$user_barrier" || exit 1
echo "[PHASE] summary submitted -> will land at experiments/$CELL_BASE_REL/full_grad_vs_layerwise_summary.{md,csv}"

echo "[DONE] full-gradient krum/multikrum campaign submitted for seeds ($SEEDS)."
