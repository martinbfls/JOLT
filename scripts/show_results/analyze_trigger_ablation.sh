#!/bin/bash
###
# analyze_trigger_ablation.sh
#
# Every analysis of the trigger-stealth ablation, in the order they should be read.
# ablate_trigger_stealth_slurm.sh gates a Slurm job on this, so a campaign produces
# its own verdict without a second command -- but it is also the command to run by
# hand, at any point, on whatever has finished so far (each step reports the cells it
# could not read instead of failing).
#
# Usage (from repo root, with the JOLT conda env active):
#   ./scripts/show_results/analyze_trigger_ablation.sh
#   SEEDS="0 1 2" BUDGETS="150 500 1500" ./scripts/show_results/analyze_trigger_ablation.sh
#   VARIANT_TAGS="ablate_base ablate_tv" ./scripts/show_results/analyze_trigger_ablation.sh
#
# A failing step never hides the others; the exit status is the number of failures.
###

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$BASE_DIR"

MODEL_FLAG="${MODEL_FLAG:-r32p}"
DATASET="${DATASET:-cifar}"
GEN_AGG_METHOD="${GEN_AGG_METHOD:-mean}"
SEEDS="${SEEDS:-0 1 2}"
# Budget 0 is the control: the ASR the trigger reaches with NO flips, i.e. the share of the
# attack that is a plain adversarial perturbation rather than an induced backdoor.
BUDGETS="${BUDGETS:-0 150 300 500}"
VARIANT_TAGS="${VARIANT_TAGS:-ablate_base ablate_tv ablate_eot ablate_mask ablate_color}"
DEPLOY_AGG_METHOD="${DEPLOY_AGG_METHOD:-multikrum}"
DEPLOY_NUM_POISONED="${DEPLOY_NUM_POISONED:-3}"
DEPLOY_NUM_HONESTS="${DEPLOY_NUM_HONESTS:-7}"
OUT_ROOT="${OUT_ROOT:-experiments/federated_experiments/trigger_stealth_ablation}"
# 0 skips run_all_campaign.sh (the slowest step, and the one whose figures are least
# specific to this ablation).
WITH_CAMPAIGN="${WITH_CAMPAIGN:-1}"

read -ra TAGS <<< "$VARIANT_TAGS"
read -ra SEED_ARR <<< "$SEEDS"
read -ra BUDGET_ARR <<< "$BUDGETS"

COMMON=(--variant-tags "${TAGS[@]}" --model-flag "$MODEL_FLAG" --dataset "$DATASET"
        --gen-agg-method "$GEN_AGG_METHOD" --seeds "${SEED_ARR[@]}" --budgets "${BUDGET_ARR[@]}"
        --deploy-agg-method "$DEPLOY_AGG_METHOD"
        --deploy-num-poisoned "$DEPLOY_NUM_POISONED" --deploy-num-honests "$DEPLOY_NUM_HONESTS")

FAILURES=0
step() {
    local name="$1"; shift
    echo
    echo "=============================================================="
    echo "[STEP] $name"
    echo "=============================================================="
    if "$@"; then
        echo "[STEP OK] $name"
    else
        echo "[STEP FAILED] $name (exit $?) -- continuing with the others" >&2
        FAILURES=$((FAILURES + 1))
    fi
}

# 1. The verdict: which of the three bottlenecks the attack is in. Read first, because it
#    decides whether trigger expressivity is the right lever at all.
step "bottleneck diagnosis -> $OUT_ROOT/diagnosis/verdict.md" \
    python scripts/show_results/diagnose_trigger_capacity.py "${COMMON[@]}" \
        --out-dir "$OUT_ROOT/diagnosis"

# 2. ASR against MEASURED visibility -- the comparison epsilon cannot make.
step "measured stealth vs ASR -> $OUT_ROOT/stealth/stealth.md" \
    python scripts/show_results/measure_trigger_stealth.py "${COMMON[@]}" \
        --out-dir "$OUT_ROOT/stealth"

# 3. ASR/CTA vs flip budget, one line per variant.
step "ASR/CTA vs budget -> $OUT_ROOT/vs_budget/" \
    python scripts/show_results/plot_warp_multibudget.py "${COMMON[@]}" --per-seed \
        --out-dir "$OUT_ROOT/vs_budget"

# 4. Did the backdoor hold, or peak and get unlearned? Separates a trigger that is genuinely
#    induced from one that is memorized and then washed out at an LR milestone.
#    --branch both (its default) reads the centralized deploy too, whose aggregator is a no-op
#    mean: a collapse there rules the aggregator out as the cause.
step "ASR dynamics -> $OUT_ROOT/dynamics/" \
    python scripts/show_results/analyze_asr_dynamics.py "${COMMON[@]}" \
        --out-dir "$OUT_ROOT/dynamics"

# 5. The usual per-campaign figures/tables.
if [ "$WITH_CAMPAIGN" = "1" ]; then
    step "full campaign results -> $OUT_ROOT/campaign/" \
        env SUBMIT=0 VARIANT_TAGS="$VARIANT_TAGS" MODEL_FLAG="$MODEL_FLAG" DATASET="$DATASET" \
            GEN_AGG_METHOD="$GEN_AGG_METHOD" SEEDS="$SEEDS" BUDGETS="$BUDGETS" \
            DEPLOY_AGG_METHOD="$DEPLOY_AGG_METHOD" \
            DEPLOY_NUM_POISONED="$DEPLOY_NUM_POISONED" DEPLOY_NUM_HONESTS="$DEPLOY_NUM_HONESTS" \
            OUT_ROOT="$OUT_ROOT/campaign" CAMPAIGN=trigger_stealth_ablation \
            ./scripts/show_results/run_all_campaign.sh
fi

cat <<EOF

==============================================================
[READ IN THIS ORDER]
==============================================================
  1. $OUT_ROOT/diagnosis/verdict.md
     Which bottleneck you are in: TRIGGER CAPACITY, INDUCIBILITY, or AGGREGATION.
     Only the first one is an argument for adding trigger expressivity. At deploy
     time the victim trains on clean images with flipped labels, so the robust
     aggregator never sees the trigger -- an AGGREGATION verdict cannot be fixed by
     any trigger, however expressive, and a U-Net would only make an INDUCIBILITY
     verdict worse.

  2. $OUT_ROOT/stealth/stealth.md  +  stealth/pareto.pdf
     ASR against measured LPIPS/SSIM/PSNR, not against epsilon (which bounds the
     additive half only and is just an upper bound once the contrast mask is on).
     Compare variants along a row of equal budget: a win moves UP AND LEFT. Up and
     to the right only bought ASR with visibility.

  3. experiments/federated_experiments/threat_model_direct_trigger_joint_<tag>/\\
         $MODEL_FLAG/$DATASET/1vs0/$GEN_AGG_METHOD/seed*/gen_labels_trigger_joint/trigger_preview.png
     The eyeball check, with each component drawn separately. Available right after
     GEN, before any victim has trained.

  4. $OUT_ROOT/vs_budget/asr_cta_vs_budget.pdf
     Look at the LOW budgets (150/500): that is the only place a better trigger can
     show up as better evasion, since its contribution is needing fewer flips.

  5. $OUT_ROOT/dynamics/transience.md
     A variant whose ASR peaks and collapses has not induced the backdoor, it has
     been memorized and then unlearned -- do not read its final ASR as a win.
EOF

if [ "$FAILURES" -gt 0 ]; then
    echo
    echo "[DONE] $FAILURES step(s) failed (usually: cells still running). Re-run this script later." >&2
fi
exit "$FAILURES"
