#!/bin/bash
###
# run_all_campaign.sh
#
# Every results script that can be pointed at an arbitrary set of VARIANT_TAGS, in one command --
# the counterpart of run_all_out_iclr.sh for the trigger campaigns laid out as
#   <EXP_ROOT>/<EXP_BASE_PREFIX>_<tag>/<MODEL>/<DATASET>/1vs0/<GEN_AGG>/seed<seed>/
#       train_user_<budget>/                            centralized
#       <DEPLOY_AGG>_<P>vs<H>/train_user_<budget>/      federated
# i.e. what run_warp_test_vs_multikrum_slurm.sh, run_warp_multibudget_vs_multikrum_slurm.sh and
# sweep_hybrid_epsilon_slurm.sh produce.
#
#   1. plot_warp_multibudget.py   ASR/CTA vs budget, one line per tag, mean +/- std over seeds
#   2. analyze_asr_dynamics.py    per-epoch dynamics: peak vs final ASR, when it peaked, and the
#                                 series that say why it collapsed (both branches)
#   3. make_paper_layout_view.py  symlink view so the next three, which only read the
#                                 paper_main_campaign layout, work on this tree unchanged
#   4. plot_paper_main_campaign_user.py   CTA-vs-ASR trade-off plots + LaTeX tables
#   5. plot_asr_per_seed.py + plot_flip_matrix.py
#   6. plot_trigger_grid.py               clean vs poisoned, one row per tag
#
# i.e. every visualization run_all_out_iclr.sh produces, plus the two that are specific to these
# campaigns. NOT run here: visualize_trigger.py (already invoked per cell by the orchestrators,
# right after GEN) and summarize_warp_sweep.py (only discovers warp_s*_lpips* dirs).
#
# Nothing runs on the login node: by default this SUBMITS itself as a single CPU-only job on
# SLURM_PARTITION -- the same partition as the campaign, since `pine` accepts srun from an RDHPC
# node but refuses sbatch from one. It runs inline instead when it is already inside an
# allocation (SLURM_JOB_ID set, e.g. the dependency-gated job the orchestrators submit, or your
# own srun --pty), or when given SUBMIT=0.
#
# Usage (from repo root):
#   VARIANT_TAGS="hybrid_eps0p25_s8p0 hybrid_eps0p0627_s8p0" SEEDS="0 1 2" \
#       SLURM_ACCOUNT=<account> ./scripts/show_results/run_all_campaign.sh
#   SUBMIT=0 VARIANT_TAGS=hybrid_eps0p0627_s8p0 ./scripts/show_results/run_all_campaign.sh
###

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
export BASE_DIR
cd "$BASE_DIR"

PYTHON="${PYTHON:-python}"
EXP_ROOT="${EXP_ROOT:-experiments/federated_experiments}"
EXP_BASE_PREFIX="${EXP_BASE_PREFIX:-threat_model_direct_trigger_joint}"
VARIANT_TAGS="${VARIANT_TAGS:-}"
MODEL_FLAG="${MODEL_FLAG:-r32p}"
DATASET="${DATASET:-cifar}"
GEN_AGG_METHOD="${GEN_AGG_METHOD:-mean}"
SEEDS="${SEEDS:-0 1 2}"
BUDGETS="${BUDGETS:-0 150 300 500 1000 2000 2500 5000}"
DEPLOY_AGG_METHOD="${DEPLOY_AGG_METHOD:-multikrum}"
DEPLOY_NUM_POISONED="${DEPLOY_NUM_POISONED:-3}"
DEPLOY_NUM_HONESTS="${DEPLOY_NUM_HONESTS:-7}"
# Extra tags to draw alongside in the vs-budget figure; read only, never re-run.
REFERENCE_TAGS="${REFERENCE_TAGS:-}"
OUT_ROOT="${OUT_ROOT:-$EXP_ROOT/campaign_results}"

if [ -z "${VARIANT_TAGS// /}" ]; then
    echo "[ABORT] VARIANT_TAGS is required, e.g. VARIANT_TAGS='hybrid_eps0p0627_s8p0'" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Submit ourselves, unless already running inside an allocation (the gated job
# the orchestrators submit, or an interactive srun) or told not to.
# ---------------------------------------------------------------------------
SUBMIT="${SUBMIT:-1}"
if [ -n "${SLURM_JOB_ID:-}" ]; then
    SUBMIT=0
fi

if [ "$SUBMIT" = "1" ]; then
    # shellcheck source=../../orchestrate_slurm/slurm_lib.sh
    source "$BASE_DIR/orchestrate_slurm/slurm_lib.sh"
    RESULTS_PARTITION="${RESULTS_PARTITION:-$SLURM_PARTITION}"
    preflight_slurm || exit 1

    self_cmd="SUBMIT=0 EXP_ROOT='$EXP_ROOT' EXP_BASE_PREFIX='$EXP_BASE_PREFIX'"
    self_cmd+=" VARIANT_TAGS='$VARIANT_TAGS' REFERENCE_TAGS='$REFERENCE_TAGS'"
    self_cmd+=" MODEL_FLAG=$MODEL_FLAG DATASET=$DATASET GEN_AGG_METHOD=$GEN_AGG_METHOD"
    self_cmd+=" SEEDS='$SEEDS' BUDGETS='$BUDGETS'"
    self_cmd+=" DEPLOY_AGG_METHOD=$DEPLOY_AGG_METHOD"
    self_cmd+=" DEPLOY_NUM_POISONED=$DEPLOY_NUM_POISONED DEPLOY_NUM_HONESTS=$DEPLOY_NUM_HONESTS"
    self_cmd+=" OUT_ROOT='$OUT_ROOT' ./scripts/show_results/run_all_campaign.sh"

    GPUS_PER_TASK=0 CPUS_PER_TASK="${CPUS_PER_TASK:-4}" MEM_PER_TASK="${MEM_PER_TASK:-8G}" \
    TIME_PER_TASK="${TIME_PER_TASK:-00:20:00}" \
        job=$(submit_job_slurm "$self_cmd" "campaign_results" "${RESULTS_DEPENDENCY:-}" "$RESULTS_PARTITION") || exit 1
    echo "[SUBMIT] job $job on partition $RESULTS_PARTITION -> log: $LOG_DIR/campaign_results.log"
    echo "[SUBMIT] results will land under $OUT_ROOT/{vs_budget,asr_dynamics}"
    exit 0
fi

COMMON=(--exp-root "$EXP_ROOT" --exp-base-prefix "$EXP_BASE_PREFIX"
        --model-flag "$MODEL_FLAG" --dataset "$DATASET" --gen-agg-method "$GEN_AGG_METHOD"
        --deploy-agg-method "$DEPLOY_AGG_METHOD"
        --deploy-num-poisoned "$DEPLOY_NUM_POISONED" --deploy-num-honests "$DEPLOY_NUM_HONESTS")
read -ra _SEEDS <<< "$SEEDS"
read -ra _BUDGETS <<< "$BUDGETS"
read -ra _TAGS <<< "$VARIANT_TAGS"
read -ra _REF_TAGS <<< "$REFERENCE_TAGS"
PLOT_TAGS=("${_TAGS[@]}" ${_REF_TAGS[@]+"${_REF_TAGS[@]}"})

echo "[PLAN] tags=(${_TAGS[*]}) references=(${_REF_TAGS[*]:-none}) seeds=(${_SEEDS[*]})"
echo "[PLAN] exp_root=$EXP_ROOT out_root=$OUT_ROOT branch=${DEPLOY_AGG_METHOD}_${DEPLOY_NUM_POISONED}vs${DEPLOY_NUM_HONESTS}"

# Each step is allowed to fail (a campaign still running has cells missing, and one script's
# abort must not hide the other's output); the exit status reports how many did.
failures=0

echo
echo "=== [1/2] plot_warp_multibudget.py (ASR/CTA vs budget) ==="
"$PYTHON" scripts/show_results/plot_warp_multibudget.py \
    "${COMMON[@]}" --variant-tags "${PLOT_TAGS[@]}" \
    --seeds "${_SEEDS[@]}" --budgets "${_BUDGETS[@]}" \
    --per-seed --out-dir "$OUT_ROOT/vs_budget" || { echo "[WARN] step 1 failed" >&2; failures=$((failures + 1)); }

echo
echo "=== [2/6] analyze_asr_dynamics.py (per-epoch dynamics, both branches) ==="
"$PYTHON" scripts/show_results/analyze_asr_dynamics.py \
    "${COMMON[@]}" --variant-tags "${_TAGS[@]}" \
    --seeds "${_SEEDS[@]}" --budgets "${_BUDGETS[@]}" \
    --branch both --out-dir "$OUT_ROOT/asr_dynamics" || { echo "[WARN] step 2 failed" >&2; failures=$((failures + 1)); }

# ---------------------------------------------------------------------------
# The four run_all_out_iclr.sh visualizations. They only read the paper_main_campaign tree
# layout, so they run against a symlink view of this campaign (see make_paper_layout_view.py)
# rather than being taught a second layout.
# ---------------------------------------------------------------------------
VIEW="${VIEW:-$OUT_ROOT/paper_layout_view}"
echo
echo "=== [3/6] make_paper_layout_view.py (symlink view for the out_iclr scripts) ==="
if "$PYTHON" scripts/show_results/make_paper_layout_view.py \
        --exp-root "$EXP_ROOT" --exp-base-prefix "$EXP_BASE_PREFIX" \
        --variant-tags "${_TAGS[@]}" --model-flag "$MODEL_FLAG" --dataset "$DATASET" \
        --gen-agg-method "$GEN_AGG_METHOD" --seeds "${_SEEDS[@]}" \
        --deploy-agg-method "$DEPLOY_AGG_METHOD" \
        --deploy-num-poisoned "$DEPLOY_NUM_POISONED" --deploy-num-honests "$DEPLOY_NUM_HONESTS" \
        --out "$VIEW"; then
    echo
    echo "=== [4/6] plot_paper_main_campaign_user.py (CTA-vs-ASR + LaTeX tables) ==="
    "$PYTHON" scripts/show_results/plot_paper_main_campaign_user.py \
        --model "$MODEL_FLAG" --datasets "$DATASET" --tags "${_TAGS[@]}" \
        --seeds "${_SEEDS[@]}" --jolt-root "$VIEW" \
        --out-dir "$OUT_ROOT/paper_main_campaign_user" \
        --csv-dir "$OUT_ROOT/results_csv" --table-dir "$OUT_ROOT/tables" \
        || { echo "[WARN] step 4 failed" >&2; failures=$((failures + 1)); }

    echo
    echo "=== [5/6] plot_asr_per_seed.py + plot_flip_matrix.py ==="
    "$PYTHON" scripts/show_results/plot_asr_per_seed.py \
        --campaign jolt --root "$VIEW" --model "$MODEL_FLAG" --datasets "$DATASET" \
        --tags "${_TAGS[@]}" --seeds "${_SEEDS[@]}" --budgets "${_BUDGETS[@]}" \
        --out-dir "$OUT_ROOT/per_seed" || { echo "[WARN] plot_asr_per_seed failed" >&2; failures=$((failures + 1)); }
    "$PYTHON" scripts/show_results/plot_flip_matrix.py \
        --campaign jolt --root "$VIEW" --model "$MODEL_FLAG" --datasets "$DATASET" \
        --tags "${_TAGS[@]}" --seeds "${_SEEDS[@]}" --budgets "${_BUDGETS[@]}" \
        --out-dir "$OUT_ROOT/flip_matrix" || { echo "[WARN] plot_flip_matrix failed" >&2; failures=$((failures + 1)); }

    echo
    echo "=== [6/6] plot_trigger_grid.py (clean vs poisoned, one row per tag) ==="
    "$PYTHON" scripts/show_results/plot_trigger_grid.py \
        --models "$MODEL_FLAG" --datasets "$DATASET" --tags "${_TAGS[@]}" \
        --jolt-root "$VIEW" --out-dir "$OUT_ROOT/trigger_grid" \
        || { echo "[WARN] step 6 failed" >&2; failures=$((failures + 1)); }
else
    echo "[WARN] no cell to build the view from -- steps 4-6 skipped" >&2
    failures=$((failures + 1))
fi

echo
if [ "$failures" -gt 0 ]; then
    echo "[DONE] $failures step(s) failed (campaign still running? see their own output above)."
else
    echo "[DONE] results under $OUT_ROOT/"
fi
exit "$failures"
