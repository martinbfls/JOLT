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
#
# TRIGGER_TYPE (default "warp") must match what the cells were GENERATED with -- it selects the
# deploy-time poisoner flag. For cells produced by TRIGGER_TYPE=hybrid:
#   TRIGGER_TYPE=hybrid VARIANT_TAGS="hybrid_eps0p5_s8" SEEDS="0 1 2 3 4" \
#       SLURM_ACCOUNT=<account> ./orchestrate_slurm/run_warp_multibudget_vs_multikrum_slurm.sh
#
# SPLIT_BATCH=1 re-deploys with split_batch_across_workers=true, i.e. the same optimizer-step
# count as the centralized branch instead of num_workers times fewer. Use RESULTS_OUT to send its
# results somewhere of their own, since they are NOT comparable with the cell's existing ones:
#   TRIGGER_TYPE=hybrid VARIANT_TAGS=hybrid_eps0p0627_s8p0 SEEDS="0 1 2" SPLIT_BATCH=1 \
#       RESULTS_OUT=experiments/federated_experiments/hybrid_eps0p0627_splitbatch \
#       SLURM_ACCOUNT=<account> ./orchestrate_slurm/run_warp_multibudget_vs_multikrum_slurm.sh
#
# Once every train_user job has finished, a final job (gated via --dependency) runs
# scripts/show_results/run_all_campaign.sh into $RESULTS_OUT.
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
# Must match the trigger_type the cells were GENERATED with: it picks the deploy-time poisoner
# flag written into the train_user configs. A mismatch is caught at load time (pick_poisoner
# refuses a trigger whose saved trigger_type maps to another poisoner), not silently deployed.
TRIGGER_TYPE="${TRIGGER_TYPE:-warp}"
# 1 writes split_batch_across_workers=true into the train_user configs: `batch_size` becomes the
# TOTAL across workers, so the federated victim takes the same number of optimizer steps as the
# centralized one instead of num_workers times fewer (see schemas/federated_train_user.toml).
# Changes every federated number -- re-deploying an existing cell with it produces results that
# are NOT comparable with what is already on disk for that cell.
SPLIT_BATCH="${SPLIT_BATCH:-0}"
SEEDS="${SEEDS:-0 1 2 3 4}"
BUDGETS="${BUDGETS:-0 150 300 500 1000 2000 2500 5000}"
DEPLOY_NUM_POISONED="${DEPLOY_NUM_POISONED:-3}"
DEPLOY_NUM_HONESTS="${DEPLOY_NUM_HONESTS:-7}"
DEPLOY_AGG_METHOD="${DEPLOY_AGG_METHOD:-multikrum}"
DRY_RUN="${DRY_RUN:-0}"

GEN_CONFIGS_TIME="${GEN_CONFIGS_TIME:-00:10:00}"
FLIPS_TIME="${FLIPS_TIME:-00:10:00}"
USER_TIME="${USER_TIME:-0-01:00:00}"
RESULTS_OUT="${RESULTS_OUT:-experiments/federated_experiments/warp_multibudget_summary}"
# Same partition as the campaign: `pine` takes srun from an RDHPC node but refuses sbatch from one.
RESULTS_PARTITION="${RESULTS_PARTITION:-}"

echo "[PLAN] model=$MODEL_FLAG dataset=$DATASET trigger_type=$TRIGGER_TYPE variant_tags=($VARIANT_TAGS) seeds=($SEEDS) budgets=($BUDGETS)"
echo "[PLAN] split_batch_across_workers=$SPLIT_BATCH (1 = same optimizer-step count as centralized)"
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
        FED_GEN_ARGS=(--deploy-federated --trigger-type "$TRIGGER_TYPE" --exp-base-tag "$tag"
                      --model-flag "$MODEL_FLAG" --dataset "$DATASET" --seed "$seed"
                      --gen-agg-method "$GEN_AGG_METHOD" --budgets "$BUDGETS"
                      --deploy-num-poisoned "$DEPLOY_NUM_POISONED" --deploy-num-honests "$DEPLOY_NUM_HONESTS"
                      --deploy-agg-method "$DEPLOY_AGG_METHOD")
        [ "$SPLIT_BATCH" = "1" ] && FED_GEN_ARGS+=(--split-batch-across-workers)
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
RESULTS_PARTITION="${RESULTS_PARTITION:-$SLURM_PARTITION}"

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
USER_JOB_IDS=("${SUBMITTED_JOB_IDS[@]}")

echo "[DONE] multikrum-only multi-budget campaign submitted for variant_tags=($VARIANT_TAGS), seeds=($SEEDS), budgets=($BUDGETS)."
echo "[COMPARE] once finished, per (tag, seed, budget):"
echo "  experiments/<tag_cell>/${DEPLOY_AGG_METHOD}_${DEPLOY_NUM_POISONED}vs${DEPLOY_NUM_HONESTS}/train_user_<budget>/logs/federated_train_user_metrics.json"

# ---------------------------------------------------------------------------
# Detailed results, gated on every train_user job above (see run_all_campaign.sh).
# ---------------------------------------------------------------------------
results_cmd="SUBMIT=0 VARIANT_TAGS='$VARIANT_TAGS' MODEL_FLAG=$MODEL_FLAG DATASET=$DATASET"
results_cmd+=" GEN_AGG_METHOD=$GEN_AGG_METHOD SEEDS='$SEEDS' BUDGETS='$BUDGETS'"
results_cmd+=" DEPLOY_AGG_METHOD=$DEPLOY_AGG_METHOD"
results_cmd+=" DEPLOY_NUM_POISONED=$DEPLOY_NUM_POISONED DEPLOY_NUM_HONESTS=$DEPLOY_NUM_HONESTS"
results_cmd+=" OUT_ROOT=$RESULTS_OUT ./scripts/show_results/run_all_campaign.sh"

# Never fatal: the campaign is already queued, so losing the command would be the worst outcome.
if results_job=$(GPUS_PER_TASK=0 CPUS_PER_TASK=4 MEM_PER_TASK=8G TIME_PER_TASK=00:15:00 \
        submit_job_slurm "$results_cmd" "warp_mb_results" \
        "afterok:$(join_job_ids "${USER_JOB_IDS[@]}")" "$RESULTS_PARTITION"); then
    echo "[PHASE] results job $results_job submitted on $RESULTS_PARTITION, gated on ${#USER_JOB_IDS[@]} train_user jobs"
    echo "[PHASE] once it completes: $RESULTS_OUT/{vs_budget,asr_dynamics}/"
    echo "[PHASE] a failed cell leaves it pending -- scancel it and run this by hand:"
else
    echo "[PHASE] the results job could NOT be submitted -- the campaign jobs are unaffected." >&2
    echo "[PHASE] run this once every cell has finished (no Slurm needed):"
fi
echo "  $results_cmd"
