#!/bin/bash
###
# run_backdoorbench.sh
#
# Runs BackdoorBench's backdoor DETECTION methods on existing JOLT results (appendix benchmark).
# No attack is regenerated: for each requested cell, attack/jolt.py rebuilds the poisoned CIFAR-10
# training set from that cell's single_user select_flips output (flipped labels, clean images) and
# its JOLT trigger, BackdoorBench trains its own victim on it (default PreAct-ResNet18), then:
#   data-level detection (detection_pretrain/): can the flipped training samples be found?
#   inference detection (detection_infer/): can triggered test inputs be flagged?
# Results land in <BB_DIR>/record/<record>/; summarize them with collect_backdoorbench.py.
#
# Setup (once):
#   git clone https://github.com/SCLBD/BackdoorBench && cd BackdoorBench && bash sh/install.sh
#   (BackdoorBench is CC BY-NC 4.0: kept as a separate checkout, never copied into this repo.)
#
# Usage (from anywhere; one victim training per tag x seed x budget, ~100 epochs on a GPU):
#   BB_DIR=~/BackdoorBench ./backdoorbench/run_backdoorbench.sh
#   BB_DIR=~/BackdoorBench TAGS="baseline" SEEDS="0" BUDGETS="1000" DRY_RUN=1 ./backdoorbench/run_backdoorbench.sh
#   BB_DIR=~/BackdoorBench CAMPAIGN=flip_frozen ./backdoorbench/run_backdoorbench.sh   # FLIP, frozen JOLT trigger
#   BB_DIR=~/BackdoorBench REFERENCE_ATTACKS="badnet blended" ...  # also BackdoorBench's own attacks, same protocol
#   inside a Slurm job: sbatch --gres=gpu:1 --time=2-00:00:00 --wrap "BB_DIR=... SEEDS=0 ./backdoorbench/run_backdoorbench.sh"
#
# Env: CAMPAIGN (jolt|flip_frozen), MODEL_FLAG (JOLT model whose labels/trigger are replayed),
#   TAGS, SEEDS, BUDGETS, DETECTORS_PRETRAIN, DETECTORS_INFER, BB_MODEL, DEVICE, EPOCHS (blank =
#   BackdoorBench's prototype default), ATTACK_ARGS (extra args for the victim training, e.g.
#   "--num_workers 0 --amp True"), TARGET_LAYER (feature layer of SS/SPECTRE/SCAn/Beatrix; default
#   per BB_MODEL, as in BackdoorBench's analysis tools), REFERENCE_ATTACKS, PYTHON, DRY_RUN.
# Only CIFAR-10 (BackdoorBench does not ship SVHN). Cells already done are skipped.
###

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOLT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BB_DIR="${BB_DIR:?set BB_DIR to a BackdoorBench checkout (git clone https://github.com/SCLBD/BackdoorBench)}"
BB_DIR="$(cd "$BB_DIR" && pwd)"

CAMPAIGN="${CAMPAIGN:-jolt}"
MODEL_FLAG="${MODEL_FLAG:-r32p}"
read -ra TAGS <<< "${TAGS:-baseline eps_16_255}"
read -ra SEEDS <<< "${SEEDS:-0 1 2}"
read -ra BUDGETS <<< "${BUDGETS:-500 1000 2500}"
read -ra DETECTORS_PRETRAIN <<< "${DETECTORS_PRETRAIN:-ac spectral spectre scan beatrix agpd}"
read -ra DETECTORS_INFER <<< "${DETECTORS_INFER:-strip sentinet teco}"
read -ra REFERENCE_ATTACKS <<< "${REFERENCE_ATTACKS:-}"
BB_MODEL="${BB_MODEL:-preactresnet18}"
DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-}"
read -ra ATTACK_ARGS <<< "${ATTACK_ARGS:-}"
case "$BB_MODEL" in  # BackdoorBench's own default feature layers (analysis/visual_utils.py)
    preactresnet18|resnet18) DEFAULT_TARGET_LAYER="layer4.1.conv2" ;;
    vgg19) DEFAULT_TARGET_LAYER="features.34" ;;
    densenet161) DEFAULT_TARGET_LAYER="features.denseblock4.denselayer24.conv2" ;;
    mobilenet_v3_large) DEFAULT_TARGET_LAYER="features.16.0" ;;
    efficientnet_b3) DEFAULT_TARGET_LAYER="features.7.1.block.3.0" ;;
    *) DEFAULT_TARGET_LAYER="" ;;
esac
TARGET_LAYER="${TARGET_LAYER:-$DEFAULT_TARGET_LAYER}"
PYTHON="${PYTHON:-python}"
DRY_RUN="${DRY_RUN:-0}"
# BackdoorBench's load_attack_result calls torch.load() without weights_only=False, which
# torch >= 2.6 refuses for its attack_result.pt (pickled dataset state); its own pinned env is
# torch 1.11. The records are produced by this very script, so restoring the old default is safe.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
SOURCE_LABEL=9   # paper_main_campaign SOURCE_LABEL / TARGET_LABEL
TARGET_LABEL=4

case "$CAMPAIGN" in
    jolt) EXP_BASE_REL="federated_experiments/threat_model_direct_trigger_joint_paper_main_campaign" ;;
    flip_frozen) EXP_BASE_REL="federated_experiments/flip_frozen_trigger_paper_main_campaign" ;;
    *) echo "[ABORT] CAMPAIGN must be jolt or flip_frozen, got '$CAMPAIGN'" >&2; exit 1 ;;
esac
JOINT_EXP_BASE_REL="federated_experiments/threat_model_direct_trigger_joint_paper_main_campaign"

run() {  # run <log file> <cmd...>: from BB_DIR, appending to the log; never aborts the sweep
    local log="$1"; shift
    if [ "$DRY_RUN" = "1" ]; then echo "[DRY-RUN] (cd $BB_DIR && $*)"; return 0; fi
    mkdir -p "$(dirname "$log")"
    if (cd "$BB_DIR" && "$@") >> "$log" 2>&1; then return 0; fi
    echo "[FAIL] $* -- see $log" >&2
    return 1
}

detection_csv() {  # detection_csv <record> <family> <method>
    case "$1:$3" in
        *:agpd) echo "$BB_DIR/record/$1/detection_pretrain/agpd/detection_info.csv" ;;
        *) echo "$BB_DIR/record/$1/detection/${3}_${2}/detection_info.csv" ;;
    esac
}

run_detectors() {  # run_detectors <record>
    local rec="$1" d csv extra
    for d in "${DETECTORS_PRETRAIN[@]}"; do
        csv="$(detection_csv "$rec" pretrain "$d")"
        [ -f "$csv" ] && continue
        extra=()
        [ "$d" = "agpd" ] && extra=(--csv_save_path "$csv")
        case "$d" in spectral|spectre|scan|beatrix)  # these hook a feature layer and have no default
            [ -z "$TARGET_LAYER" ] && { echo "[FAIL] $d needs TARGET_LAYER for BB_MODEL=$BB_MODEL" >&2; continue; }
            extra+=(--target_layer "$TARGET_LAYER") ;;
        esac
        run "$BB_DIR/record/$rec/jolt_runner.log" "$PYTHON" "detection_pretrain/$d.py" --result_file "$rec" \
            --yaml_path "./config/detection/$d/cifar10.yaml" --device "$DEVICE" "${extra[@]}" || true
    done
    for d in "${DETECTORS_INFER[@]}"; do
        csv="$(detection_csv "$rec" infer "$d")"
        [ -f "$csv" ] && continue
        run "$BB_DIR/record/$rec/jolt_runner.log" "$PYTHON" "detection_infer/$d.py" --result_file "$rec" \
            --yaml_path "./config/detection/$d/cifar10.yaml" --device "$DEVICE" || true
    done
}

ensure_attack() {  # ensure_attack <record> <cmd...>: trains the victim unless attack_result.pt exists
    local rec="$1"; shift
    local dir="$BB_DIR/record/$rec"
    [ -f "$dir/attack_result.pt" ] && return 0
    if [ -d "$dir" ] && [ "$DRY_RUN" != "1" ]; then  # BackdoorBench refuses an existing folder: keep the partial one aside
        mv "$dir" "$dir.incomplete.$(date +%Y%m%d%H%M%S)"
    fi
    run "$BB_DIR/record/${rec}_attack.log" "$@"
}

if [ "$DRY_RUN" != "1" ]; then
    mkdir -p "$BB_DIR/attack" "$BB_DIR/config/attack/jolt" "$BB_DIR/record"
    cp "$SCRIPT_DIR/attack/jolt.py" "$BB_DIR/attack/jolt.py"
    cp "$SCRIPT_DIR/config/attack/jolt/default.yaml" "$BB_DIR/config/attack/jolt/default.yaml"
fi
epochs_arg=("${ATTACK_ARGS[@]}")
[ -n "$EPOCHS" ] && epochs_arg+=(--epochs "$EPOCHS")

MISSING=0
for tag in "${TAGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    cell="$JOLT_DIR/experiments/$EXP_BASE_REL/$MODEL_FLAG/cifar/$tag/seed${seed}"
    flips_dir="$cell/select_flips"
    trigger="$JOLT_DIR/experiments/$JOINT_EXP_BASE_REL/$MODEL_FLAG/cifar/$tag/seed${seed}/gen_labels_trigger_joint/trigger/opt_trig_direct_joint_stripe_${MODEL_FLAG}_cifar_1vs0.pt"
    for budget in "${BUDGETS[@]}"; do
        if [ ! -f "$flips_dir/worker0/${budget}_labels.npy" ] || [ ! -f "$trigger" ]; then
            echo "[MISSING] $CAMPAIGN/$MODEL_FLAG/cifar/$tag/seed$seed budget $budget: need $flips_dir/worker0/${budget}_labels.npy and $trigger" >&2
            MISSING=$((MISSING + 1))
            continue
        fi
        rec="jolt_${CAMPAIGN}_${MODEL_FLAG}_cifar_${tag}_seed${seed}_b${budget}"
        echo "[CELL] $rec"
        ensure_attack "$rec" "$PYTHON" attack/jolt.py --yaml_path ./config/attack/prototype/cifar10.yaml \
            --bd_yaml_path ./config/attack/jolt/default.yaml --save_folder_name "$rec" \
            --flips_dir "$flips_dir" --budget "$budget" --trigger_path "$trigger" \
            --source_label "$SOURCE_LABEL" --attack_target "$TARGET_LABEL" \
            --model "$BB_MODEL" --device "$DEVICE" "${epochs_arg[@]}" || continue
        run_detectors "$rec"
    done
  done
done

for ref in "${REFERENCE_ATTACKS[@]}"; do
    rec="ref_${ref}_cifar_t${TARGET_LABEL}"
    echo "[REFERENCE] $rec"
    ref_args=()
    [ "$ref" = "badnet" ] && ref_args=(--patch_mask_path ./resource/badnet/trigger_image.png)
    ensure_attack "$rec" "$PYTHON" "attack/$ref.py" --yaml_path ./config/attack/prototype/cifar10.yaml \
        --save_folder_name "$rec" --attack_target "$TARGET_LABEL" --model "$BB_MODEL" --device "$DEVICE" \
        "${ref_args[@]}" "${epochs_arg[@]}" || continue
    run_detectors "$rec"
done

[ "$MISSING" -gt 0 ] && echo "[NOTE] $MISSING cell(s) skipped: their JOLT flips/trigger are not on disk." >&2
echo "[DONE] summarize with: python $SCRIPT_DIR/collect_backdoorbench.py --bb-dir $BB_DIR"
