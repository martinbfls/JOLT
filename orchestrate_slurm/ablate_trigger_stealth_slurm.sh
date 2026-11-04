#!/bin/bash
###
# ablate_trigger_stealth_slurm.sh
#
# One command for the trigger-stealth ablation: for each VARIANT below, the whole
# EXPERT -> GEN -> FLIPS -> USER chain is run from scratch, on both deployments
# (centralized 1vs0/mean and federated multikrum 3vs7), over every SEED and every
# BUDGET, and then three analyses are produced:
#
#   diagnose_trigger_capacity.py   where the ASR is lost: trigger capacity, inducibility
#                                  by the label flips, or the robust aggregator. Read
#                                  verdict.md FIRST -- it decides whether more trigger
#                                  expressivity is even the right lever.
#   measure_trigger_stealth.py     LPIPS/SSIM/PSNR/Linf of the perturbation as DEPLOYED,
#                                  joined with ASR. This is what makes the variants
#                                  comparable: `epsilon` does not measure a warp or a
#                                  photometric map, so it cannot rank them.
#   run_all_campaign.sh            the usual per-campaign figures/tables.
#
# The variants are CUMULATIVE, each adding one mechanism to the one before, so a
# difference between two consecutive rows attributes cleanly:
#
#   base        the current reference: hybrid, warp_k=4, warp_s=8, uniform epsilon
#   tv          + lambda_tv_warp, warp_k=8, warp_s=2   (more DOF, less amplitude)
#   eot         + eot_augment                           (match the expert's augmentation)
#   mask        + contrast_mask_floor                   (spend epsilon where it hides)
#   color       + color_trigger                         (the photometric mechanism)
#
# Everything is set by env var; see the DEFAULTS block. The cheap first pass is a
# single seed and the low budgets, which is where the question is decided:
#
#   SEEDS=0 BUDGETS="0 300" SLURM_ACCOUNT=<account> \
#       ./orchestrate_slurm/ablate_trigger_stealth_slurm.sh
#
# The full run (3 seeds, 5 variants, 4 budgets):
#
#   SLURM_ACCOUNT=<account> ./orchestrate_slurm/ablate_trigger_stealth_slurm.sh
#
# SPLIT_BATCH=1 gives the federated victim the same optimizer-step count as the
# centralized one. It is slower, and it is the convention to use for anything
# reported: without it the federated victim is under-trained (CTA ~87.5 vs ~92.6),
# so its CTA is confounded, and its final ASR is read earlier on a curve that peaks
# and then collapses -- which inflates it. The two conventions are NOT comparable.
#
#   SPLIT_BATCH=1 EPSILON=0.0627 SLURM_ACCOUNT=<account> \
#       ./orchestrate_slurm/ablate_trigger_stealth_slurm.sh
#
# REFERENCE_TAGS brings an EXISTING campaign into the same comparison and fills in
# whatever it never ran -- typically the centralized budgets a federated-only
# campaign skipped, which is precisely what tells an inducibility failure apart
# from an aggregation one. No EXPERT and no GEN is re-run for those tags, and no
# cell that already has results is recomputed. Everything in one command:
#
#   SPLIT_BATCH=1 EPSILON=0.0627 REFERENCE_TAGS="hybrid_eps0p0627_s8p0" \
#       SLURM_ACCOUNT=<account> ./orchestrate_slurm/ablate_trigger_stealth_slurm.sh
#
# DRY_RUN=1 prints the plan and the per-variant gen_configs flags, submits nothing.
# VARIANTS selects a subset, e.g. VARIANTS="base tv" to check one mechanism alone.
#
# Each variant is its own experiments/ subtree and its own checkpoint subdir
# (--exp-base-tag / --checkpoint-tag = the variant tag), so nothing collides and a
# variant can be re-run alone. Triggers are NOT shared between variants: each one
# re-optimizes, which is the point -- the mechanisms change what the optimizer finds.
###

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
export BASE_DIR
cd "$BASE_DIR"

# shellcheck source=slurm_lib.sh
source "$SCRIPT_DIR/slurm_lib.sh"

# --------------------------------------------------------------------------- DEFAULTS
MODEL_FLAG="${MODEL_FLAG:-r32p}"
DATASET="${DATASET:-cifar}"
SEEDS="${SEEDS:-0 1 2}"
# Budget 0 is a CONTROL, not a data point to skip: it is the ASR the trigger reaches with NO
# label flips at all, i.e. the part of the attack that is a plain adversarial perturbation rather
# than a backdoor. Every ASR above it should be read as an increment over it.
# Then the low budgets, because at deploy time the aggregator never sees the trigger, so the only
# way a trigger helps against MultiKrum is by needing FEWER flips. High budgets are deliberately
# out of the default: past a few hundred flips the victim memorizes them one by one instead of
# explaining them with the trigger's feature, and ASR goes DOWN.
BUDGETS="${BUDGETS:-0 150 300 500}"
VARIANTS="${VARIANTS:-base tv eot mask color}"
DRY_RUN="${DRY_RUN:-0}"

# Already-generated campaign tags to bring into the SAME comparison, e.g. "hybrid_eps0p0627_s8p0".
# Their trigger exists on disk, so no EXPERT and no GEN is re-run: only the deploy configs are
# (re)written for $BUDGETS on BOTH branches, select_flips is run (it is fully deterministic --
# argsort on fixed distances, default_rng(0), seed=0 -- so re-running it reproduces the existing
# files byte-for-byte), and train_user is submitted ONLY for cells that have no results yet.
# Nothing already computed is overwritten. This is what fills the centralized budgets a
# federated-only campaign (run_warp_multibudget_vs_multikrum_slurm.sh) never ran, which is
# exactly what the inducibility-vs-aggregation diagnosis needs.
REFERENCE_TAGS="${REFERENCE_TAGS:-}"

# 1 = `batch_size` is the TOTAL across workers (batch_size//num_workers each), so the federated
# victim takes the same number of optimizer steps as the centralized one instead of num_workers
# times fewer. Without it the federated victim is under-trained (CTA ~87.5 vs ~92.6) and every
# federated number is confounded with that; worse, fewer steps means the final ASR is read
# earlier on a curve that peaks and then collapses, which inflates it. Results are NOT comparable
# across the two conventions -- a campaign must not mix them, and RESULTS_OUT below is suffixed
# accordingly. See schemas/federated_train_user.toml.
SPLIT_BATCH="${SPLIT_BATCH:-0}"

# Shared across every variant, so only the ablated mechanism differs.
TRIGGER_TYPE="hybrid"
EPSILON="${EPSILON:-0.0314}"        # 8/255, the stealthy additive budget
LAMBDA_BD="${LAMBDA_BD:-1.0}"
LAMBDA_JOINT="${LAMBDA_JOINT:-1.0}"
LAMBDA_LPIPS="${LAMBDA_LPIPS:-0.0}"

# Reference warp (base): the configuration the current campaigns use.
BASE_WARP_K="${BASE_WARP_K:-4}"
BASE_WARP_S="${BASE_WARP_S:-8.0}"
# Warp with the TV penalty: more control points, far less amplitude. WaNet's own stealth regime
# is k=4/s=0.5, so s=8 at k=4 is ~16x its displacement budget; the point of the TV term is to buy
# the same effect with degrees of freedom instead.
TV_WARP_K="${TV_WARP_K:-8}"
TV_WARP_S="${TV_WARP_S:-2.0}"
LAMBDA_TV_WARP="${LAMBDA_TV_WARP:-0.05}"

CONTRAST_MASK_FLOOR="${CONTRAST_MASK_FLOOR:-0.25}"
COLOR_A_MAX="${COLOR_A_MAX:-0.05}"
COLOR_B_MAX="${COLOR_B_MAX:-0.02}"
LR_COLOR="${LR_COLOR:-1e-3}"

DEPLOY_NUM_POISONED="${DEPLOY_NUM_POISONED:-3}"
DEPLOY_NUM_HONESTS="${DEPLOY_NUM_HONESTS:-7}"
DEPLOY_AGG_METHOD="${DEPLOY_AGG_METHOD:-multikrum}"
GEN_AGG_METHOD="mean"

EXPERT_TIME="${EXPERT_TIME:-0-02:00:00}"
GEN_TIME="${GEN_TIME:-1-00:00:00}"
FLIPS_TIME="${FLIPS_TIME:-00:10:00}"
# 3h, not the 1h the federated-only orchestrators use: SPLIT_BATCH=1 gives the victim
# num_workers times more optimizer steps per epoch (196 vs 20 on CIFAR/r32p with 10 workers), so
# this phase is the one that convention actually makes long. It covers both branches and the
# reference backfill.
USER_TIME="${USER_TIME:-0-03:00:00}"
GEN_CONFIGS_TIME="${GEN_CONFIGS_TIME:-00:10:00}"
# 1h: the analysis runs five steps over every tag at once, and measure_trigger_stealth evaluates
# LPIPS on n-images x tags x seeds pairs on CPU. A timed-out analysis job loses the verdict and
# forces a manual re-run, which is exactly what this script exists to avoid.
ANALYSIS_TIME="${ANALYSIS_TIME:-0-01:00:00}"

# Suffixed by convention, so a SPLIT_BATCH=1 ablation never overwrites a SPLIT_BATCH=0 one:
# their numbers are not comparable (see SPLIT_BATCH above).
_split_suffix=""
[ "$SPLIT_BATCH" = "1" ] && _split_suffix="_splitbatch"
RESULTS_OUT="${RESULTS_OUT:-experiments/federated_experiments/trigger_stealth_ablation${_split_suffix}}"
RESULTS_PARTITION="${RESULTS_PARTITION:-}"

# --------------------------------------------------------------------------- VARIANTS
# Cumulative: each variant's extra gen_configs flags, on top of every previous one's.
variant_flags() {
    local v="$1"
    local -a f=(--warp-k "$BASE_WARP_K" --warp-s "$BASE_WARP_S")
    case "$v" in
        base) ;;
        tv|eot|mask|color)
            f=(--warp-k "$TV_WARP_K" --warp-s "$TV_WARP_S" --lambda-tv-warp "$LAMBDA_TV_WARP")
            ;;
        *) echo "[ABORT] unknown variant '$v' (expected one of: base tv eot mask color)" >&2; exit 1 ;;
    esac
    case "$v" in
        eot|mask|color) f+=(--eot-augment) ;;
    esac
    case "$v" in
        mask|color) f+=(--contrast-mask-floor "$CONTRAST_MASK_FLOOR") ;;
    esac
    case "$v" in
        color) f+=(--color-trigger --color-a-max "$COLOR_A_MAX" --color-b-max "$COLOR_B_MAX" --lr-color "$LR_COLOR") ;;
    esac
    printf '%s\n' "${f[@]}"
}

variant_tag() { echo "ablate_$1"; }

# Reads variant_flags' lines into the array VARIANT_ARGS. A read loop rather than `mapfile`, so
# this one function stays usable under bash 3.2 (the orchestration as a whole needs 4.3 anyway,
# for slurm_lib.sh's namerefs) and the variant logic can be checked outside the cluster.
read_variant_flags() {
    VARIANT_ARGS=()
    while IFS= read -r line; do
        VARIANT_ARGS+=("$line")
    done < <(variant_flags "$1")
}

echo "[PLAN] model=$MODEL_FLAG dataset=$DATASET trigger_type=$TRIGGER_TYPE epsilon=$EPSILON"
echo "[PLAN] variants=($VARIANTS) seeds=($SEEDS) budgets=($BUDGETS)"
echo "[PLAN] deployments: centralized (1vs0, mean) AND federated ($DEPLOY_AGG_METHOD, ${DEPLOY_NUM_POISONED}vs${DEPLOY_NUM_HONESTS})"
echo "[PLAN] both are needed: the centralized branch is what separates an inducibility failure from an aggregation one."
echo "[PLAN] split_batch_across_workers=$SPLIT_BATCH (1 = same optimizer-step count as centralized; slower, but the only convention whose federated CTA is not confounded with under-training)"
if [ "$SPLIT_BATCH" = "1" ]; then
    echo "[PLAN] WARNING: the train_user_* results inside each cell are per-convention. Re-running these"
    echo "[PLAN]          same variant tags later with SPLIT_BATCH=0 overwrites them in place (only the"
    echo "[PLAN]          aggregated results under RESULTS_OUT are kept apart). Use different tags, or"
    echo "[PLAN]          back the cells up, if you need both conventions side by side."
fi
for v in $VARIANTS; do
    read_variant_flags "$v"
    echo "[PLAN]   $(variant_tag "$v"): ${VARIANT_ARGS[*]}"
done
if [ -n "$REFERENCE_TAGS" ]; then
    echo "[PLAN] reference tags (no EXPERT/GEN, deploy backfill only, existing results kept): ($REFERENCE_TAGS)"
fi

# True when that train_user cell already produced results (same two sources plot_warp_multibudget
# reads): used to leave an existing reference-tag cell alone instead of recomputing it.
has_results() {
    [ -f "experiments/$1/logs/federated_train_user_metrics.json" ] || [ -f "experiments/$1/paccs.npy" ]
}

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

# --------------------------------------------------------------------------- CONFIGS
ALL_TAGS=()
MISSING=0
for v in $VARIANTS; do
    tag="$(variant_tag "$v")"
    ALL_TAGS+=("$tag")
    read_variant_flags "$v"
    for seed in $SEEDS; do
        echo "[GEN] configs for $tag seed=$seed..."
        GEN_ARGS=(--single-cell --trigger-type "$TRIGGER_TYPE" --exp-base-tag "$tag"
                  --model-flag "$MODEL_FLAG" --dataset "$DATASET" --seed "$seed"
                  --budgets "$BUDGETS" --epsilon "$EPSILON"
                  --lambda-bd "$LAMBDA_BD" --lambda-joint "$LAMBDA_JOINT" --lambda-lpips "$LAMBDA_LPIPS"
                  "${VARIANT_ARGS[@]}")
        [ "$SPLIT_BATCH" = "1" ] && GEN_ARGS+=(--split-batch-across-workers)
        [ "$DRY_RUN" = "1" ] && GEN_ARGS+=(--dry-run)
        run_gen_configs_on_compute_node "${GEN_ARGS[@]}"

        CELL_REL="federated_experiments/threat_model_direct_trigger_joint_${tag}/$MODEL_FLAG/$DATASET/1vs0/$GEN_AGG_METHOD/seed$seed"
        for p in "federated_experiments/threat_model_direct_trigger_joint_${tag}/train_expert/${MODEL_FLAG}_1xs_${tag}/seed${seed}" \
                 "$CELL_REL/gen_labels_trigger_joint" "$CELL_REL/select_flips"; do
            [ -f "experiments/$p/config.toml" ] || { echo "[CONFIG] missing: experiments/$p/config.toml" >&2; MISSING=$((MISSING + 1)); }
        done
    done
done
# Reference tags: deploy configs only, both branches. The trigger must already exist -- the
# gen_configs deploy modes abort if it does not, which is the check we want.
for tag in $REFERENCE_TAGS; do
    for seed in $SEEDS; do
        echo "[GEN] deploy configs for reference tag $tag seed=$seed (no EXPERT/GEN)..."
        for mode in --deploy-centralized --deploy-federated; do
            REF_ARGS=("$mode" --trigger-type "$TRIGGER_TYPE" --exp-base-tag "$tag"
                      --model-flag "$MODEL_FLAG" --dataset "$DATASET" --seed "$seed"
                      --gen-agg-method "$GEN_AGG_METHOD" --budgets "$BUDGETS")
            [ "$mode" = "--deploy-federated" ] && REF_ARGS+=(--deploy-num-poisoned "$DEPLOY_NUM_POISONED"
                                                             --deploy-num-honests "$DEPLOY_NUM_HONESTS"
                                                             --deploy-agg-method "$DEPLOY_AGG_METHOD")
            [ "$SPLIT_BATCH" = "1" ] && REF_ARGS+=(--split-batch-across-workers)
            [ "$DRY_RUN" = "1" ] && REF_ARGS+=(--dry-run)
            run_gen_configs_on_compute_node "${REF_ARGS[@]}"
        done
    done
done

if [ "$MISSING" -gt 0 ] && [ "$DRY_RUN" != "1" ]; then
    echo "[ABORT] $MISSING config(s) missing after the GEN-configs step." >&2
    exit 1
fi

if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY-RUN] nothing submitted."
    exit 0
fi

preflight_slurm || exit 1
RESULTS_PARTITION="${RESULTS_PARTITION:-$SLURM_PARTITION}"

# --------------------------------------------------------------------------- REFERENCE BACKFILL
# Independent of the ablation's EXPERT/GEN: these tags' triggers are already on disk, so their
# results land early instead of waiting a day behind five GEN jobs. select_flips always runs (it
# is deterministic, so the budgets that already have flips get identical files back); train_user
# runs only where there is nothing yet.
REF_USER_JOB_IDS=()
if [ -n "$REFERENCE_TAGS" ]; then
    REF_FLIPS_JOBS=()
    for tag in $REFERENCE_TAGS; do
        for seed in $SEEDS; do
            cell="federated_experiments/threat_model_direct_trigger_joint_${tag}/$MODEL_FLAG/$DATASET/1vs0/$GEN_AGG_METHOD/seed$seed"
            REF_FLIPS_JOBS+=("python run_experiment.py $cell/select_flips|ref_${tag}_cenflips_seed${seed}")
            REF_FLIPS_JOBS+=("python run_experiment.py $cell/${DEPLOY_AGG_METHOD}_${DEPLOY_NUM_POISONED}vs${DEPLOY_NUM_HONESTS}/select_flips|ref_${tag}_fedflips_seed${seed}")
        done
    done
    TIME_PER_TASK="$FLIPS_TIME"
    GPUS_PER_TASK=0 submit_job_pool_slurm REF_FLIPS_JOBS "ref_flips" "" || exit 1
    ref_flips_barrier=$(submit_barrier_slurm "ref_barrier_flips" "afterok:$(join_job_ids "${SUBMITTED_JOB_IDS[@]}")") || exit 1
    echo "[PHASE] REFERENCE FLIPS submitted (${#REF_FLIPS_JOBS[@]} jobs) -> barrier $ref_flips_barrier"

    REF_USER_JOBS=()
    REF_SKIPPED=0
    for tag in $REFERENCE_TAGS; do
        for seed in $SEEDS; do
            cell="federated_experiments/threat_model_direct_trigger_joint_${tag}/$MODEL_FLAG/$DATASET/1vs0/$GEN_AGG_METHOD/seed$seed"
            for budget in $BUDGETS; do
                cen="$cell/train_user_${budget}"
                fed="$cell/${DEPLOY_AGG_METHOD}_${DEPLOY_NUM_POISONED}vs${DEPLOY_NUM_HONESTS}/train_user_${budget}"
                if has_results "$cen"; then
                    REF_SKIPPED=$((REF_SKIPPED + 1))
                else
                    REF_USER_JOBS+=("python run_experiment.py $cen|ref_${tag}_cenuser_seed${seed}_${budget}")
                fi
                if has_results "$fed"; then
                    REF_SKIPPED=$((REF_SKIPPED + 1))
                else
                    REF_USER_JOBS+=("python run_experiment.py $fed|ref_${tag}_feduser_seed${seed}_${budget}")
                fi
            done
        done
    done
    echo "[PHASE] REFERENCE USER: ${#REF_USER_JOBS[@]} cell(s) to run, $REF_SKIPPED already done (left untouched)"
    if [ "${#REF_USER_JOBS[@]}" -gt 0 ]; then
        TIME_PER_TASK="$USER_TIME"
        submit_job_pool_slurm REF_USER_JOBS "ref_user" "afterok:$ref_flips_barrier" || exit 1
        REF_USER_JOB_IDS=("${SUBMITTED_JOB_IDS[@]}")
    fi
fi

# --------------------------------------------------------------------------- EXPERT
# One expert per (variant, seed): the variants differ in how the trigger is PARAMETERIZED, and
# expert_retrain_interval retrains against the current trigger anyway, so they cannot share one.
EXPERT_JOBS=()
for v in $VARIANTS; do
    tag="$(variant_tag "$v")"
    for seed in $SEEDS; do
        EXPERT_JOBS+=("python run_experiment.py federated_experiments/threat_model_direct_trigger_joint_${tag}/train_expert/${MODEL_FLAG}_1xs_${tag}/seed${seed}|abl_${v}_expert_seed${seed}")
    done
done
TIME_PER_TASK="$EXPERT_TIME"
submit_job_pool_slurm EXPERT_JOBS "abl_expert" "" || exit 1
expert_barrier=$(submit_barrier_slurm "abl_barrier_expert" "afterok:$(join_job_ids "${SUBMITTED_JOB_IDS[@]}")") || exit 1
echo "[PHASE] EXPERT submitted (${#EXPERT_JOBS[@]} jobs) -> barrier $expert_barrier"

# --------------------------------------------------------------------------- GEN
GEN_JOBS=()
for v in $VARIANTS; do
    tag="$(variant_tag "$v")"
    for seed in $SEEDS; do
        GEN_JOBS+=("python run_experiment.py federated_experiments/threat_model_direct_trigger_joint_${tag}/$MODEL_FLAG/$DATASET/1vs0/$GEN_AGG_METHOD/seed$seed/gen_labels_trigger_joint|abl_${v}_gen_seed${seed}")
    done
done
TIME_PER_TASK="$GEN_TIME"
submit_job_pool_slurm GEN_JOBS "abl_gen" "afterok:$expert_barrier" || exit 1
gen_barrier=$(submit_barrier_slurm "abl_barrier_gen" "afterok:$(join_job_ids "${SUBMITTED_JOB_IDS[@]}")") || exit 1
echo "[PHASE] GEN submitted (${#GEN_JOBS[@]} jobs) -> barrier $gen_barrier"

# --------------------------------------------------------------------------- VIZ (off GEN)
# Ready as soon as GEN finishes, long before any victim trains: the first thing to look at, since
# a variant whose trigger is plainly visible is already answered.
VIZ_JOBS=()
for v in $VARIANTS; do
    tag="$(variant_tag "$v")"
    for seed in $SEEDS; do
        cell="federated_experiments/threat_model_direct_trigger_joint_${tag}/$MODEL_FLAG/$DATASET/1vs0/$GEN_AGG_METHOD/seed$seed"
        cmd="python scripts/show_results/visualize_trigger.py"
        cmd+=" --exp-base experiments/federated_experiments/threat_model_direct_trigger_joint_${tag}"
        cmd+=" --model-flag $MODEL_FLAG --dataset $DATASET --seed $seed --trigger-type $TRIGGER_TYPE"
        cmd+=" --out experiments/$cell/gen_labels_trigger_joint/trigger_preview.png"
        VIZ_JOBS+=("$cmd|abl_${v}_viz_seed${seed}")
    done
done
TIME_PER_TASK="00:05:00"
GPUS_PER_TASK=0 CPUS_PER_TASK=2 MEM_PER_TASK=4G submit_job_pool_slurm VIZ_JOBS "abl_viz" "afterok:$gen_barrier" || exit 1
echo "[PHASE] trigger previews submitted (${#VIZ_JOBS[@]} jobs)"

# --------------------------------------------------------------------------- FED configs
# The federated select_flips/train_user configs can only be written once the trigger .pt exists,
# so this is its own job gated on GEN rather than done at submission time.
FED_GEN_JOBS=()
for v in $VARIANTS; do
    tag="$(variant_tag "$v")"
    for seed in $SEEDS; do
        cmd="python -m modules.generate_labels_trigger_joint.gen_configs --deploy-federated"
        cmd+=" --trigger-type $TRIGGER_TYPE --exp-base-tag $tag --model-flag $MODEL_FLAG"
        cmd+=" --dataset $DATASET --seed $seed --budgets '$BUDGETS'"
        cmd+=" --deploy-num-poisoned $DEPLOY_NUM_POISONED --deploy-num-honests $DEPLOY_NUM_HONESTS"
        cmd+=" --deploy-agg-method $DEPLOY_AGG_METHOD"
        [ "$SPLIT_BATCH" = "1" ] && cmd+=" --split-batch-across-workers"
        FED_GEN_JOBS+=("$cmd|abl_${v}_fedgen_seed${seed}")
    done
done
TIME_PER_TASK="$GEN_CONFIGS_TIME"
GPUS_PER_TASK=0 CPUS_PER_TASK=2 MEM_PER_TASK=4G submit_job_pool_slurm FED_GEN_JOBS "abl_fedgen" "afterok:$gen_barrier" || exit 1
fedgen_barrier=$(submit_barrier_slurm "abl_barrier_fedgen" "afterok:$(join_job_ids "${SUBMITTED_JOB_IDS[@]}")") || exit 1
echo "[PHASE] FED_GEN submitted (${#FED_GEN_JOBS[@]} jobs) -> barrier $fedgen_barrier"

# --------------------------------------------------------------------------- FLIPS
FLIPS_JOBS=()
for v in $VARIANTS; do
    tag="$(variant_tag "$v")"
    for seed in $SEEDS; do
        cell="federated_experiments/threat_model_direct_trigger_joint_${tag}/$MODEL_FLAG/$DATASET/1vs0/$GEN_AGG_METHOD/seed$seed"
        FLIPS_JOBS+=("python run_experiment.py $cell/select_flips|abl_${v}_cenflips_seed${seed}")
        FLIPS_JOBS+=("python run_experiment.py $cell/${DEPLOY_AGG_METHOD}_${DEPLOY_NUM_POISONED}vs${DEPLOY_NUM_HONESTS}/select_flips|abl_${v}_fedflips_seed${seed}")
    done
done
TIME_PER_TASK="$FLIPS_TIME"
GPUS_PER_TASK=0 submit_job_pool_slurm FLIPS_JOBS "abl_flips" "afterok:$fedgen_barrier" || exit 1
flips_barrier=$(submit_barrier_slurm "abl_barrier_flips" "afterok:$(join_job_ids "${SUBMITTED_JOB_IDS[@]}")") || exit 1
echo "[PHASE] FLIPS submitted (${#FLIPS_JOBS[@]} jobs) -> barrier $flips_barrier"

# --------------------------------------------------------------------------- USER
USER_JOBS=()
for v in $VARIANTS; do
    tag="$(variant_tag "$v")"
    for seed in $SEEDS; do
        cell="federated_experiments/threat_model_direct_trigger_joint_${tag}/$MODEL_FLAG/$DATASET/1vs0/$GEN_AGG_METHOD/seed$seed"
        for budget in $BUDGETS; do
            USER_JOBS+=("python run_experiment.py $cell/train_user_${budget}|abl_${v}_cenuser_seed${seed}_${budget}")
            USER_JOBS+=("python run_experiment.py $cell/${DEPLOY_AGG_METHOD}_${DEPLOY_NUM_POISONED}vs${DEPLOY_NUM_HONESTS}/train_user_${budget}|abl_${v}_feduser_seed${seed}_${budget}")
        done
    done
done
TIME_PER_TASK="$USER_TIME"
submit_job_pool_slurm USER_JOBS "abl_user" "afterok:$flips_barrier" || exit 1
USER_JOB_IDS=("${SUBMITTED_JOB_IDS[@]}")
echo "[PHASE] USER submitted (${#USER_JOBS[@]} jobs)."

# --------------------------------------------------------------------------- ANALYSIS
# One command, in its own script: the job body runs under `set -euo pipefail`, so a chain of
# `;`-separated analyses would stop at the first one that cannot read a cell yet. The script
# runs each step independently and exits with the number of failures.
# The reference tags join the comparison: ablate_base is the ablation's own control, and a
# reference tag is the campaign already on disk that the whole exercise is trying to beat.
ANALYSIS_TAGS=("${ALL_TAGS[@]}")
for tag in $REFERENCE_TAGS; do
    ANALYSIS_TAGS+=("$tag")
done
# Gated on BOTH chains: the ablation's train_user jobs and whatever the reference backfill had
# left to compute (possibly nothing, in which case only the former).
ANALYSIS_DEPS=("${USER_JOB_IDS[@]}")
[ "${#REF_USER_JOB_IDS[@]}" -gt 0 ] && ANALYSIS_DEPS+=("${REF_USER_JOB_IDS[@]}")

analysis_cmd="VARIANT_TAGS='${ANALYSIS_TAGS[*]}' MODEL_FLAG=$MODEL_FLAG DATASET=$DATASET"
analysis_cmd+=" GEN_AGG_METHOD=$GEN_AGG_METHOD SEEDS='$SEEDS' BUDGETS='$BUDGETS'"
analysis_cmd+=" DEPLOY_AGG_METHOD=$DEPLOY_AGG_METHOD DEPLOY_NUM_POISONED=$DEPLOY_NUM_POISONED"
analysis_cmd+=" DEPLOY_NUM_HONESTS=$DEPLOY_NUM_HONESTS OUT_ROOT=$RESULTS_OUT"
analysis_cmd+=" ./scripts/show_results/analyze_trigger_ablation.sh"

# Never fatal: the campaign is already queued, so losing the command would be the worst outcome.
if analysis_job=$(GPUS_PER_TASK=0 CPUS_PER_TASK=4 MEM_PER_TASK=16G TIME_PER_TASK="$ANALYSIS_TIME" \
        submit_job_slurm "$analysis_cmd" "abl_analysis" \
        "afterok:$(join_job_ids "${ANALYSIS_DEPS[@]}")" "$RESULTS_PARTITION"); then
    echo "[PHASE] analysis job $analysis_job submitted on $RESULTS_PARTITION, gated on ${#ANALYSIS_DEPS[@]} train_user jobs"
else
    echo "[PHASE] the analysis job could NOT be submitted -- the campaign jobs are unaffected." >&2
    echo "[PHASE] run this once every cell has finished (no Slurm needed):"
    echo "  $analysis_cmd"
fi

cat <<EOF

[DONE] submitted: ${#ALL_TAGS[@]} ablation variant(s) from scratch + ${#REF_USER_JOB_IDS[@]} reference
       backfill job(s), x $(echo "$SEEDS" | wc -w) seed(s), both deployments, one analysis over
       all ${#ANALYSIS_TAGS[@]} tag(s).

[RESULTS] the analysis job writes everything under $RESULTS_OUT/ and prints the reading
          order (verdict first). To see partial results at any point, or to re-run the
          analyses by hand after a failed cell:

  $analysis_cmd
EOF
