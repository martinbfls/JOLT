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
#
# NOT run here (bound to the paper_main_campaign layout, no --root/--variant-tags override):
# plot_paper_main_campaign_user.py, plot_asr_per_seed.py, plot_flip_matrix.py, plot_trigger_grid.py
# -- use run_all_out_iclr.sh for those. visualize_trigger.py is already invoked per cell by the
# orchestrators, right after GEN. summarize_warp_sweep.py only discovers warp_s*_lpips* dirs.
#
# Unlike run_all_out_iclr.sh this submits nothing: these two read a few hundred small files and
# take seconds, so it runs wherever you start it (login node, inside an srun, or as the
# dependency-gated job the sweep scripts submit).
#
# Usage (from repo root, conda env active):
#   VARIANT_TAGS="hybrid_eps0p25_s8p0 hybrid_eps0p0627_s8p0" SEEDS="0 1 2" \
#       ./scripts/show_results/run_all_campaign.sh
#   VARIANT_TAGS=hybrid_eps0p0627_s8p0 OUT_ROOT=experiments/federated_experiments/my_run \
#       ./scripts/show_results/run_all_campaign.sh
###

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
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
echo "=== [2/2] analyze_asr_dynamics.py (per-epoch dynamics, both branches) ==="
"$PYTHON" scripts/show_results/analyze_asr_dynamics.py \
    "${COMMON[@]}" --variant-tags "${_TAGS[@]}" \
    --seeds "${_SEEDS[@]}" --budgets "${_BUDGETS[@]}" \
    --branch both --out-dir "$OUT_ROOT/asr_dynamics" || { echo "[WARN] step 2 failed" >&2; failures=$((failures + 1)); }

echo
if [ "$failures" -gt 0 ]; then
    echo "[DONE] $failures/2 step(s) failed (campaign still running? see their own output above)."
else
    echo "[DONE] results under $OUT_ROOT/{vs_budget,asr_dynamics}"
fi
exit "$failures"
