#!/bin/bash
###
# sweep_warp_min_perturbation_slurm.sh
#
# One-command sweep to find the MINIMAL warp perturbation (warp_s) that still
# backdoors multikrum, optionally cross-checking LPIPS regularization at each
# point. For each (warp_s, lambda_lpips) cell, calls
# run_warp_test_vs_multikrum_slurm.sh as its own child process (own VARIANT_TAG,
# own experiments/out/checkpoints subtree, own Slurm job names) -- the two
# scripts stay independent, this one is just a driver.
#
# Usage (from repo root, login-node shell):
#   SLURM_ACCOUNT=<account> ./orchestrate_slurm/sweep_warp_min_perturbation_slurm.sh
#   SLURM_ACCOUNT=<account> DRY_RUN=1 ./orchestrate_slurm/sweep_warp_min_perturbation_slurm.sh
#   WARP_S_VALUES="2 4 8 16" LAMBDA_LPIPS_VALUES="0.0 0.1" \
#       SLURM_ACCOUNT=<account> ./orchestrate_slurm/sweep_warp_min_perturbation_slurm.sh
#
# Each cell's results, once finished, land at (sanitized warp_s/lambda_lpips, e.g.
# warp_s4p0_lpips0p1):
#   experiments/federated_experiments/threat_model_direct_trigger_joint_warp_s<S>_lpips<L>/
#       r32p/cifar/1vs0/mean/seed0/train_user_<budget>/logs/federated_train_user_metrics.json          (centralized)
#       r32p/cifar/1vs0/mean/seed0/multikrum_3vs7/train_user_<budget>/logs/federated_train_user_metrics.json  (federated)
###

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Bisecting-ish grid around the known-working WARP_S=16 (ASR confirmed to pass multikrum) down to
# the original WARP_S=1.0 default, to locate the minimal perturbation that still backdoors it.
read -ra WARP_S_ARR <<< "${WARP_S_VALUES:-16 12 8 6 4 2 1}"
read -ra LPIPS_ARR <<< "${LAMBDA_LPIPS_VALUES:-0.0}"

# Exported so each run_warp_test_vs_multikrum_slurm.sh child inherits them (its own defaults
# are 1.0/1.0 if left unset, matching this sweep's "least-constrained + lambda_bd=1.0" baseline).
export LAMBDA_BD="${LAMBDA_BD:-1.0}"
export LAMBDA_JOINT="${LAMBDA_JOINT:-1.0}"

echo "[SWEEP] warp_s values: ${WARP_S_ARR[*]}"
echo "[SWEEP] lambda_lpips values: ${LPIPS_ARR[*]}"
echo "[SWEEP] ${#WARP_S_ARR[@]} x ${#LPIPS_ARR[@]} = $((${#WARP_S_ARR[@]} * ${#LPIPS_ARR[@]})) cells, each its own EXPERT->GEN->(centralized + multikrum) DAG"

sanitize() { echo "$1" | tr '.' 'p' | tr -d '-'; }

for warp_s in "${WARP_S_ARR[@]}"; do
    for lpips in "${LPIPS_ARR[@]}"; do
        tag="warp_s$(sanitize "$warp_s")_lpips$(sanitize "$lpips")"
        echo "[SWEEP] submitting cell: warp_s=$warp_s lambda_lpips=$lpips -> VARIANT_TAG=$tag"
        VARIANT_TAG="$tag" WARP_S="$warp_s" LAMBDA_LPIPS="$lpips" \
            "$SCRIPT_DIR/run_warp_test_vs_multikrum_slurm.sh"
    done
done

echo "[SWEEP] all cells submitted (or dry-run-previewed)."
echo "[SWEEP] once finished, scan all cells at once:"
echo "  for d in experiments/federated_experiments/threat_model_direct_trigger_joint_warp_s*/; do"
echo "    echo \"\$d\"; cat \"\$d\"/r32p/cifar/1vs0/mean/seed0/multikrum_3vs7/train_user_*/logs/federated_train_user_metrics.json 2>/dev/null | grep -A2 '\"asr\"'"
echo "  done"
