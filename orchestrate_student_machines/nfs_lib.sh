#!/bin/bash
###
# nfs_lib.sh
#
# Sourced by orchestrate_slurm/slurm_lib.sh (at its very end) when EXECUTOR=nfs. Redefines the
# functions the campaign scripts call -- preflight_slurm, submit_job_slurm, submit_barrier_slurm,
# submit_job_pool_slurm, and `srun` (used to run gen_configs) -- so that every
# orchestrate_slurm/*.sh script runs unchanged on the NFS-shared student machines:
#
#   EXECUTOR=nfs ./orchestrate_slurm/orchestrate_runs_trigger_joint_paper_main_campaign_slurm.sh
#
# Jobs are queued to nfs_scheduler.py (see its header): same "afterok" semantics as Slurm, same
# $LOG_DIR/<safe_name>.{log,done} files and jobids_<phase>.txt lists, but run over ssh, with the
# expert checkpoints kept on each machine's /Data and every job reading a given expert sent to
# the machine that trained it. With EXECUTOR unset (or =slurm) this file is never read.
#
# Settings (environment):
#   NFS_PYTHON_ENV   prefix whose bin/ is put first on the jobs' PATH (default: the prefix of the
#                    `python` found here, e.g. /usr). Also stands in for CONDA_ENV, so the
#                    scripts' own "is python $CONDA_ENV/bin/python" check passes.
#   NFS_LOCAL_ROOT, NFS_STATE_DIR, NFS_MACHINES_FILE, NFS_SLOTS_PER_MACHINE, ...
#                    see nfs_scheduler.py
###

NFS_SCHEDULER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/nfs_scheduler.py"
export BASE_DIR

if [ -z "${NFS_PYTHON_ENV:-}" ]; then
    if _nfs_py="$(command -v python 2>/dev/null)"; then
        NFS_PYTHON_ENV="$(dirname "$(dirname "$_nfs_py")")"
    else
        NFS_PYTHON_ENV=""
    fi
    unset _nfs_py
fi
CONDA_ENV="$NFS_PYTHON_ENV"
_NFS_PATH_PREPEND=""
[ -n "$NFS_PYTHON_ENV" ] && _NFS_PATH_PREPEND="$NFS_PYTHON_ENV/bin"

echo "[nfs] EXECUTOR=nfs: jobs go to the student machines via $NFS_SCHEDULER (python env: ${NFS_PYTHON_ENV:-<ssh default>})" >&2

preflight_slurm() {
    if ! command -v python3 >/dev/null 2>&1; then
        echo "[PREFLIGHT nfs] python3 not found" >&2
        return 1
    fi
    if [ ! -f "${NFS_MACHINES_FILE:-$(dirname "$NFS_SCHEDULER")/machines.txt}" ]; then
        echo "[PREFLIGHT nfs] machines file not found (NFS_MACHINES_FILE)" >&2
        return 1
    fi
    if [ -z "$NFS_PYTHON_ENV" ] || [ ! -x "$NFS_PYTHON_ENV/bin/python" ]; then
        echo "[PREFLIGHT nfs] warning: no python under NFS_PYTHON_ENV='$NFS_PYTHON_ENV' -- jobs use the ssh login PATH" >&2
    fi
    return 0
}

_nfs_ensure_scheduler() {
    python3 "$NFS_SCHEDULER" start
}

# submit_job_slurm COMMAND SAFE_NAME [DEPENDENCY_SPEC] [PARTITION] -- prints the job id.
# PARTITION, CPUS/MEM/TIME are meaningless here; GPUS_PER_TASK=0 lets the job run on a GPU-less
# machine.
submit_job_slurm() {
    local cmd="$1" safe_name="$2" dependency="${3:-}"
    local jobid
    jobid=$(python3 "$NFS_SCHEDULER" submit --cwd "$BASE_DIR" --name "$safe_name" \
            --dep "$dependency" --gpus "$GPUS_PER_TASK" --path-prepend "$_NFS_PATH_PREPEND" \
            --log "$LOG_DIR/${safe_name}.log" --done-file "$LOG_DIR/${safe_name}.done" \
            --cmd "$cmd") || return 1
    case "$jobid" in
        ''|*[!0-9]*) echo "[SUBMIT nfs] unexpected id for $safe_name: '$jobid'" >&2; return 1 ;;
    esac
    _nfs_ensure_scheduler >/dev/null 2>&1 </dev/null
    echo "$jobid"
}

# submit_barrier_slurm NAME DEPENDENCY_SPEC -- a job with no command, done when its deps are.
submit_barrier_slurm() {
    local name="$1" dependency="$2"
    local jobid
    jobid=$(python3 "$NFS_SCHEDULER" submit --cwd "$BASE_DIR" --name "$name" \
            --dep "$dependency" --barrier) || return 1
    _nfs_ensure_scheduler >/dev/null 2>&1 </dev/null
    echo "$jobid"
}

# submit_job_pool_slurm JOBS_ARRAY_NAME PHASE_NAME [DEPENDENCY_SPEC] -- same contract as the
# Slurm version (SUBMITTED_JOB_IDS, $LOG_DIR/jobids_PHASE.txt), in one scheduler call.
submit_job_pool_slurm() {
    local -n JOBS=$1
    local PHASE=$2
    local DEPENDENCY="${3:-}"
    local TOTAL=${#JOBS[@]}
    local id_file="$LOG_DIR/jobids_${PHASE}.txt"
    : > "$id_file"
    SUBMITTED_JOB_IDS=()

    echo "[SUBMIT $PHASE] (nfs) total jobs = $TOTAL, dependency = ${DEPENDENCY:-none}, gpus = $GPUS_PER_TASK" >&2
    [ "$TOTAL" -eq 0 ] && return 0

    local out jobid name
    if ! out=$(printf '%s\n' "${JOBS[@]}" | python3 "$NFS_SCHEDULER" submit-batch --cwd "$BASE_DIR" \
            --dep "$DEPENDENCY" --gpus "$GPUS_PER_TASK" --path-prepend "$_NFS_PATH_PREPEND" \
            --log-dir "$LOG_DIR"); then
        echo "[SUBMIT $PHASE] (nfs) submission failed" >&2
        return 1
    fi
    while read -r jobid name; do
        [ -n "$jobid" ] || continue
        SUBMITTED_JOB_IDS+=("$jobid")
        echo "$jobid $name" >> "$id_file"
    done <<< "$out"

    echo "[SUBMIT $PHASE] (nfs) submitted ${#SUBMITTED_JOB_IDS[@]}/$TOTAL jobs (ids in $id_file)" >&2
    _nfs_ensure_scheduler </dev/null
    [ "${#SUBMITTED_JOB_IDS[@]}" -eq "$TOTAL" ]
}

# srun [OPTIONS] COMMAND... -- the scripts only srun the (cheap, CPU) config generation: run it
# right here instead, dropping the Slurm options.
srun() {
    while [ $# -gt 0 ]; do
        case "$1" in
            --*=*) shift ;;
            -A|-p|-c|-t|-n|-N|-J|-o|-e|-w|-G|--account|--partition|--cpus-per-task|--mem|--time|--gres|--job-name)
                shift 2 ;;
            -*) shift ;;
            *) break ;;
        esac
    done
    echo "[nfs] srun -> running locally on $(hostname): $1 ..." >&2
    "$@"
}
