#!/usr/bin/env python3
"""
nfs_scheduler.py -- a tiny Slurm stand-in for the student machines (EXECUTOR=nfs).

The student machines share ~/JOLT over NFS (small quota) but each has a large local /Data.
Every orchestrate_slurm/*.sh script runs unchanged with EXECUTOR=nfs: slurm_lib.sh then sources
nfs_lib.sh, whose submit functions call `submit`/`submit-batch` here instead of sbatch. Jobs and
their afterok dependencies are recorded under $NFS_STATE_DIR (on the shared home, a few KB), and a
detached daemon (`run`, auto-started on the first submission) dispatches them over ssh, one job
per machine at a time.

Checkpoint affinity -- why this is not just a job pool:
  * every job runs with JOLT_LOCAL_STORAGE=$NFS_LOCAL_ROOT (default /Data/$USER/JOLT), so
    train_expert writes its checkpoints, and generate_labels[_trigger_joint] reads them (and
    writes its expert_retrain/ rounds), on the machine's local disk instead of the shared home
    (see modules/base_utils/util.py:local_storage_path -- a no-op when the variable is unset);
  * a train_expert job pins its checkpoint directory to the machine it runs on, and every later
    job whose config reads that directory (input_pths) is sent to that same machine. Pins are
    kept in $NFS_STATE_DIR/affinity.tsv across runs, so e.g. a later BOOTSTRAP_SKIP=1 run still
    finds its checkpoints. With NFS_SHARE_EXPERTS=1 (default), a reader whose holder machines
    are all busy instead goes to a free machine, which first rsyncs the checkpoint directory from
    a holder (machine to machine, nothing through the home) and then becomes a holder too.
    Jobs reading no checkpoint (select_flips, train_user, plots) go
    anywhere, preferably to machines no pinned job is waiting for.

Subcommands (all read their settings from the environment, see Settings below):
  status [-v]             counts, running jobs, failures
  start | run | stop      start the daemon detached / in the foreground / stop it
                          (stop --kill-jobs also kills the jobs it launched)
  cancel ID... | --all    cancel pending jobs (and their dependents); --kill also running ones
  probe                   which machines are reachable, writable, with a GPU
  affinity [--unpin KEY] [--pin KEY MACHINE]
  clean [--path SUB] [--machines "a b"] [--yes] [--force]
                          show (default) or delete $NFS_LOCAL_ROOT[/SUB] on every machine,
                          and drop the pins that pointed at what was deleted
  submit / submit-batch   used by nfs_lib.sh
"""

import argparse
import concurrent.futures
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------------
BASE_DIR = os.environ.get("BASE_DIR") or os.path.dirname(SCRIPT_DIR)
STATE_DIR = os.environ.get("NFS_STATE_DIR") or os.path.join(BASE_DIR, "nfs_state")
MACHINES_FILE = os.environ.get("NFS_MACHINES_FILE") or os.path.join(SCRIPT_DIR, "machines.txt")
LOCAL_ROOT = os.environ.get("NFS_LOCAL_ROOT") or "/Data/%s/JOLT" % os.environ.get("USER", "unknown")
SLOTS_PER_MACHINE = int(os.environ.get("NFS_SLOTS_PER_MACHINE", "1"))
POLL = float(os.environ.get("NFS_POLL", "15"))
IDLE_EXIT = float(os.environ.get("NFS_IDLE_EXIT", "900"))           # daemon exits after this idle
LIVENESS_EVERY = float(os.environ.get("NFS_LIVENESS_EVERY", "300"))
REPROBE_EVERY = float(os.environ.get("NFS_REPROBE_EVERY", "600"))
PIN_DOWN_TIMEOUT = float(os.environ.get("NFS_PIN_DOWN_TIMEOUT", "1800"))  # pinned machine down
# Copy an expert's checkpoints to a free machine (rsync, machine to machine) instead of waiting
# for the machine that trained it. 0 = every reader of an expert runs on that one machine.
SHARE_EXPERTS = os.environ.get("NFS_SHARE_EXPERTS", "1") != "0"
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=accept-new"]

JOBS_DIR = os.path.join(STATE_DIR, "jobs")      # <id>.json, written once at submission
STATE_SUB = os.path.join(STATE_DIR, "state")    # <id>.json, written by the daemon only
RC_DIR = os.path.join(STATE_DIR, "rc")          # <id>, written by the job itself (exit code)
RUN_DIR = os.path.join(STATE_DIR, "run")        # <id>.sh, <id>.host
REQ_DIR = os.path.join(STATE_DIR, "requests")   # cancel/kill requests for a running daemon
AFFINITY = os.path.join(STATE_DIR, "affinity.tsv")
PIDFILE = os.path.join(STATE_DIR, "scheduler.pid")
DAEMON_LOG = os.path.join(STATE_DIR, "scheduler.log")

FINAL = ("ok", "failed", "cancelled")


def ensure_dirs():
    for d in (JOBS_DIR, STATE_SUB, RC_DIR, RUN_DIR, REQ_DIR):
        os.makedirs(d, exist_ok=True)


def log(msg):
    print("[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg), flush=True)


def write_json_atomic(path, obj):
    tmp = "%s.tmp.%d" % (path, os.getpid())
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def read_machines():
    machines = []
    with open(MACHINES_FILE) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line and line not in machines:
                machines.append(line)
    return machines


def ssh(machine, command, timeout=60):
    try:
        p = subprocess.run(["ssh"] + SSH_OPTS + [machine, command], stdin=subprocess.DEVNULL,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout,
                           universal_newlines=True)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return 255, "", "timeout"


def parallel(fn, items, workers=32):
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        return dict(zip(items, ex.map(fn, items)))


# --------------------------------------------------------------------------------------------
# Daemon bookkeeping
# --------------------------------------------------------------------------------------------
def daemon_info():
    """(host, pid) of the recorded daemon, or None."""
    try:
        host, pid = open(PIDFILE).read().split()
        return host, int(pid)
    except (OSError, ValueError):
        return None


def daemon_alive():
    info = daemon_info()
    if info is None:
        return None
    host, pid = info
    if host == socket.gethostname():
        try:
            os.kill(pid, 0)
            return info
        except OSError:
            return None
    rc, out, _ = ssh(host, "kill -0 %d 2>/dev/null && echo alive" % pid, timeout=20)
    return info if "alive" in out else None


# --------------------------------------------------------------------------------------------
# Submission
# --------------------------------------------------------------------------------------------
def parse_deps(spec):
    """'afterok:1:2[,afterok:3]' -> [1, 2, 3] (the only form slurm_lib.sh produces)."""
    deps = []
    for part in filter(None, (spec or "").split(",")):
        kind, _, ids = part.partition(":")
        if kind != "afterok":
            raise SystemExit("nfs_scheduler: unsupported dependency type %r (only afterok)" % kind)
        deps += [int(x) for x in ids.split(":") if x]
    return deps


def allocate_id(start):
    """Reserves the next free integer id with an O_EXCL create (atomic on NFS v3+)."""
    i = start
    while True:
        try:
            os.close(os.open(os.path.join(JOBS_DIR, "%d.id" % i), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            return i
        except FileExistsError:
            i += 1


def max_id():
    ids = [int(n.split(".")[0]) for n in os.listdir(JOBS_DIR) if n.split(".")[0].isdigit()]
    return max(ids) if ids else 0


def submit_one(next_id, name, cmd, deps, gpus, cwd, logf, done_file, path_prepend, barrier=False):
    jid = allocate_id(next_id)
    if done_file:
        try:
            os.remove(done_file)
        except OSError:
            pass
    job = {"id": jid, "name": name, "cmd": cmd, "deps": deps, "gpus": gpus, "cwd": cwd,
           "log": logf, "done_file": done_file, "path_prepend": path_prepend,
           "barrier": barrier, "submitted": time.time()}
    write_json_atomic(os.path.join(JOBS_DIR, "%d.json" % jid), job)
    os.remove(os.path.join(JOBS_DIR, "%d.id" % jid))   # the .json now holds the id
    return jid


def cmd_submit(a):
    ensure_dirs()
    jid = submit_one(max_id() + 1, a.name, a.cmd or "", parse_deps(a.dep), a.gpus, a.cwd,
                     a.log, a.done_file, a.path_prepend, barrier=a.barrier)
    print(jid)


def cmd_submit_batch(a):
    """stdin: one 'COMMAND|SAFE_NAME' per line (slurm_lib.sh's job-array format)."""
    ensure_dirs()
    deps = parse_deps(a.dep)
    nxt = max_id() + 1
    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line:
            continue
        cmd, _, name = line.partition("|")
        jid = submit_one(nxt, name, cmd, deps, a.gpus, a.cwd,
                         os.path.join(a.log_dir, name + ".log"),
                         os.path.join(a.log_dir, name + ".done"), a.path_prepend)
        nxt = jid + 1
        print(jid, name, flush=True)


# --------------------------------------------------------------------------------------------
# Checkpoint affinity
# --------------------------------------------------------------------------------------------
CFG_RE = re.compile(r"run_experiment\.py\s+(\S+)")
SECTION_RE = re.compile(r"^\s*\[([^\]]+)\]\s*$")
STRING_RE = re.compile(r"""^\s*([A-Za-z0-9_]+)\s*=\s*(?:"([^"]*)"|'([^']*)')""")


def checkpoint_key(value, cwd, producer):
    """Directory holding an expert trajectory, as the repo-relative path both sides agree on:
    train_expert output_dir '<X>/0/' and input_pths '<X>/{}/model_{}_{}.pth' both give '<X>'."""
    if "{}" in value:
        value = value[:value.index("{}")]
    elif producer:
        value = os.path.dirname(value.rstrip("/"))
    else:
        value = os.path.dirname(os.path.dirname(value))
    if os.path.isabs(value):
        value = os.path.relpath(value, cwd)
    return os.path.normpath(value)


def checkpoint_role(job):
    """('producer'|'consumer', key), (None, None) for a job touching no checkpoint, or None
    when its config does not exist yet (generated by an upstream job) -- retried later."""
    if job.get("barrier"):
        return (None, None)
    m = CFG_RE.search(job["cmd"])
    if not m:
        return (None, None)
    cfg = os.path.join(job["cwd"], "experiments", m.group(1).strip("'\""), "config.toml")
    try:
        lines = open(cfg).read().splitlines()
    except OSError:
        return None
    section, consumer = None, None
    for line in lines:
        s = SECTION_RE.match(line)
        if s:
            section = s.group(1).strip()
            continue
        kv = STRING_RE.match(line)
        if not kv:
            continue
        key, val = kv.group(1), kv.group(2) if kv.group(2) is not None else kv.group(3)
        if section == "train_expert" and key == "output_dir":
            return ("producer", checkpoint_key(val, job["cwd"], True))
        if key == "input_pths" and consumer is None:
            consumer = ("consumer", checkpoint_key(val, job["cwd"], False))
    return consumer or (None, None)


def load_affinity():
    pins = {}
    try:
        for line in open(AFFINITY):
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2:
                pins[parts[0]] = [m for m in parts[1].split(",") if m]
    except OSError:
        pass
    return pins


def save_affinity(pins):
    tmp = AFFINITY + ".tmp.%d" % os.getpid()
    with open(tmp, "w") as f:
        for k in sorted(pins):
            f.write("%s\t%s\n" % (k, ",".join(pins[k])))
    os.replace(tmp, AFFINITY)


# --------------------------------------------------------------------------------------------
# Machines
# --------------------------------------------------------------------------------------------
PROBE_CMD = ("mkdir -p %s && test -w %s && echo writable; "
             "if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi -L 2>/dev/null | grep -c '^GPU'; "
             "else echo 0; fi") % (shlex.quote(LOCAL_ROOT), shlex.quote(LOCAL_ROOT))


def probe(machine):
    """{'up': bool, 'gpus': int, 'why': str}"""
    rc, out, err = ssh(machine, PROBE_CMD, timeout=30)
    if rc != 0 or "writable" not in out:
        return {"up": False, "gpus": 0, "why": (err.strip().splitlines() or ["%s not writable" % LOCAL_ROOT])[-1]}
    try:
        gpus = int(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        gpus = 0
    return {"up": True, "gpus": gpus, "why": ""}


# --------------------------------------------------------------------------------------------
# The daemon
# --------------------------------------------------------------------------------------------
RUN_TEMPLATE = """#!/bin/bash
# Generated by nfs_scheduler.py -- job {id} ({name})
exec > {log} 2>&1
hostname > {hostfile}.tmp.$$ && mv -f {hostfile}.tmp.$$ {hostfile}
finish() {{ echo "$1" > {rcfile}.tmp.$$ && mv -f {rcfile}.tmp.$$ {rcfile}; exit "$1"; }}
cd {cwd} || finish 1
unset PYTHONPATH PYTHONHOME || true
{path_line}
export JOLT_LOCAL_STORAGE={local_root}
export JOLT_NFS_JOB_ID={id}
mkdir -p "$JOLT_LOCAL_STORAGE" || finish 1
{copy_line}
echo "[nfs] job={id} name={name} node=$(hostname) python=$(command -v python) local_storage=$JOLT_LOCAL_STORAGE"
(
set -euo pipefail
{cmd}
)
rc=$?
[ "$rc" -eq 0 ] && touch {done_file}
finish "$rc"
"""


COPY_TEMPLATE = """echo "[nfs] rsync {src} -> {dst}"
mkdir -p {dst} && rsync -a --delete -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new" {src} {dst} || finish 1
touch {copied}
echo "[nfs] copy done ($(du -sh {dst} | cut -f1))"
"""


class Daemon:
    def __init__(self):
        self.jobs = {}       # id -> job
        self.state = {}      # id -> state dict (absent = pending)
        self.roles = {}      # id -> (role, key) once known
        self.pins = load_affinity()
        self.machines = read_machines()
        self.minfo = {}      # machine -> probe dict + down_since
        self.last_liveness = 0.0
        self.last_reprobe = 0.0
        self.idle_since = None

    # -- persistence ------------------------------------------------------------------------
    def set_state(self, jid, **st):
        st.setdefault("t", time.time())
        self.state[jid] = st
        write_json_atomic(os.path.join(STATE_SUB, "%d.json" % jid), st)

    def load_new_jobs(self):
        for n in os.listdir(JOBS_DIR):
            if not n.endswith(".json"):
                continue
            jid = int(n[:-5])
            if jid in self.jobs:
                continue
            job = read_json(os.path.join(JOBS_DIR, n))
            if job is None:
                continue
            self.jobs[jid] = job
            st = read_json(os.path.join(STATE_SUB, "%d.json" % jid))
            if st is not None:
                self.state[jid] = st

    # -- machines ---------------------------------------------------------------------------
    def reprobe(self, machines):
        res = parallel(probe, machines)
        now = time.time()
        for m, r in res.items():
            prev = self.minfo.get(m, {})
            r["down_since"] = None if r["up"] else (prev.get("down_since") or now)
            self.minfo[m] = r
        up = [m for m in self.machines if self.minfo.get(m, {}).get("up")]
        log("probe: %d/%d machines usable (%d with a GPU)"
            % (len(up), len(self.machines), sum(1 for m in up if self.minfo[m]["gpus"] > 0)))

    def mark_down(self, m, why):
        info = self.minfo.setdefault(m, {"gpus": 0})
        info["up"] = False
        info["why"] = why
        info["down_since"] = info.get("down_since") or time.time()
        log("machine %s marked down: %s" % (m, why))

    def running_on(self):
        count = {}
        for jid, st in self.state.items():
            if st.get("state") == "running":
                count[st["machine"]] = count.get(st["machine"], 0) + 1
        return count

    # -- job lifecycle ----------------------------------------------------------------------
    def finish(self, jid, rc):
        st = self.state[jid]
        self.register_copy(jid)
        if rc == 0:
            self.set_state(jid, state="ok", machine=st.get("machine"), rc=0)
            # The home's inode quota is finite: only failed jobs keep their run script.
            for f in (os.path.join(RC_DIR, str(jid)), os.path.join(RUN_DIR, "%d.sh" % jid),
                      os.path.join(RUN_DIR, "%d.host" % jid), os.path.join(RUN_DIR, "%d.copied" % jid)):
                try:
                    os.remove(f)
                except OSError:
                    pass
        else:
            self.set_state(jid, state="failed", machine=st.get("machine"), rc=rc,
                           reason="exit code %s" % rc)
            log("job %d (%s) FAILED on %s (rc=%s), log: %s"
                % (jid, self.jobs[jid]["name"], st.get("machine"), rc, self.jobs[jid]["log"]))

    def collect_finished(self):
        for jid, st in list(self.state.items()):
            if st.get("state") != "running":
                continue
            try:
                rc = int(open(os.path.join(RC_DIR, str(jid))).read().strip())
            except (OSError, ValueError):
                continue
            self.finish(jid, rc)

    def liveness(self):
        running = [jid for jid, st in self.state.items() if st.get("state") == "running"]
        if not running:
            return

        def check(jid):
            st = self.state[jid]
            if st.get("pid") is None:
                return "unknown"
            rc, out, _ = ssh(st["machine"], "kill -0 %d 2>/dev/null && echo alive || echo dead"
                             % st["pid"], timeout=30)
            return "unreachable" if rc != 0 else out.strip()

        res = parallel(check, running)
        now = time.time()
        for jid in running:
            st, verdict = self.state[jid], res[jid]
            if os.path.exists(os.path.join(RC_DIR, str(jid))):
                continue
            if verdict == "alive" or verdict == "unknown":
                st.pop("suspect_since", None)
                continue
            since = st.setdefault("suspect_since", now)
            # dead: give NFS time to show the rc file; unreachable: the machine may come back.
            limit = 180 if verdict == "dead" else PIN_DOWN_TIMEOUT
            if now - since > limit:
                self.set_state(jid, state="failed", machine=st["machine"], rc=None,
                               reason="lost (%s, no exit code)" % verdict)
                log("job %d (%s) lost on %s (%s)" % (jid, self.jobs[jid]["name"], st["machine"], verdict))

    def handle_requests(self):
        for n in os.listdir(REQ_DIR):
            path = os.path.join(REQ_DIR, n)
            req = read_json(path)
            try:
                os.remove(path)
            except OSError:
                pass
            if not req:
                continue
            ids = list(self.jobs) if req.get("all") else req.get("ids", [])
            for jid in ids:
                st = self.state.get(jid, {})
                if jid not in self.jobs or st.get("state") in FINAL:
                    continue
                if st.get("state") == "running":
                    if not req.get("kill"):
                        continue
                    if st.get("pid"):
                        ssh(st["machine"], "kill -TERM -- -%d 2>/dev/null; kill -TERM %d 2>/dev/null"
                            % (st["pid"], st["pid"]), timeout=30)
                self.set_state(jid, state="cancelled", machine=st.get("machine"), reason="cancelled by user")

    def resolve(self):
        """Pending jobs -> cancelled (a dependency did not succeed) or ready. Returns ready ids."""
        ready = []
        changed = True
        while changed:   # propagate cancellations through chains of barriers in one tick
            changed = False
            for jid in sorted(self.jobs):
                if jid in self.state or jid in ready:
                    continue
                deps = self.jobs[jid]["deps"]
                # An unknown id may just not be visible yet through the NFS directory cache.
                grace = time.time() - self.jobs[jid].get("submitted", 0) < 300
                bad = [d for d in deps if (d not in self.jobs and not grace)
                       or self.state.get(d, {}).get("state") in ("failed", "cancelled")]
                if bad:
                    self.set_state(jid, state="cancelled",
                                   reason="dependency %d did not succeed" % bad[0])
                    changed = True
                elif all(self.state.get(d, {}).get("state") == "ok" for d in deps):
                    if self.jobs[jid].get("barrier"):
                        self.set_state(jid, state="ok", rc=0)
                        changed = True
                    else:
                        ready.append(jid)
        return ready

    def role(self, jid):
        if jid not in self.roles:
            r = checkpoint_role(self.jobs[jid])
            if r is None:
                return None
            self.roles[jid] = r
        return self.roles[jid]

    def demand(self):
        """machine -> unfinished jobs pinned to it (they will need it later)."""
        d = {}
        for jid, job in self.jobs.items():
            if self.state.get(jid, {}).get("state") in FINAL + ("running",):
                continue
            r = self.role(jid)
            if r and self.pins.get(r[1]):
                for m in self.pins[r[1]]:
                    d[m] = d.get(m, 0) + 1
        return d

    def launch(self, jid, machine, copy_key=None, copy_from=None):
        job = self.jobs[jid]
        q = shlex.quote
        script = RUN_TEMPLATE.format(
            id=jid, name=job["name"], log=q(job["log"]), cwd=q(job["cwd"]),
            hostfile=q(os.path.join(RUN_DIR, "%d.host" % jid)),
            rcfile=q(os.path.join(RC_DIR, str(jid))),
            path_line=('export PATH=%s:"$PATH"' % q(job["path_prepend"])) if job.get("path_prepend") else "",
            local_root=q(LOCAL_ROOT), cmd=job["cmd"],
            copy_line=(COPY_TEMPLATE.format(
                src=q("%s:%s/%s/" % (copy_from, LOCAL_ROOT, copy_key)),
                dst=q("%s/%s/" % (LOCAL_ROOT, copy_key)),
                copied=q(os.path.join(RUN_DIR, "%d.copied" % jid)))
                if copy_from else ""),
            done_file=q(job["done_file"]) if job.get("done_file") else "/dev/null")
        run_sh = os.path.join(RUN_DIR, "%d.sh" % jid)
        with open(run_sh, "w") as f:
            f.write(script)
        os.makedirs(os.path.dirname(job["log"]), exist_ok=True)
        for stale in (os.path.join(RC_DIR, str(jid)), os.path.join(RUN_DIR, "%d.host" % jid),
                      os.path.join(RUN_DIR, "%d.copied" % jid)):
            try:
                os.remove(stale)
            except OSError:
                pass
        rc, out, err = ssh(machine, "nohup setsid bash %s </dev/null >/dev/null 2>&1 & echo $!" % q(run_sh),
                           timeout=60)
        pid = None
        if rc == 0:
            try:
                pid = int(out.strip().splitlines()[-1])
            except (ValueError, IndexError):
                pass
        if pid is None:
            time.sleep(5)
            if not os.path.exists(os.path.join(RUN_DIR, "%d.host" % jid)):   # never started
                self.mark_down(machine, "launch failed: %s" % (err.strip() or out.strip())[-200:])
                return False
        self.set_state(jid, state="running", machine=machine, pid=pid, started=time.time(),
                       copy_key=copy_key, copy_from=copy_from)
        log("job %d (%s) -> %s%s" % (jid, job["name"], machine,
                                     " (copying %s from %s)" % (copy_key, copy_from) if copy_from else ""))
        return True

    def dispatch(self, ready):
        busy = self.running_on()
        free = {m: SLOTS_PER_MACHINE - busy.get(m, 0) for m in self.machines
                if self.minfo.get(m, {}).get("up")}
        demand = self.demand()
        pins_per_machine = {}
        for holders in self.pins.values():
            for m in holders:
                pins_per_machine[m] = pins_per_machine.get(m, 0) + 1
        order = {m: i for i, m in enumerate(self.machines)}
        now = time.time()

        def up(m):
            return self.minfo.get(m, {}).get("up", False)

        def candidates(job):
            need_gpu = int(job.get("gpus") or 0) > 0
            return [m for m, n in free.items() if n > 0 and (not need_gpu or self.minfo[m]["gpus"] > 0)]

        def fail(jid, reason):
            self.set_state(jid, state="failed", reason=reason)
            log("job %d (%s) failed: %s" % (jid, self.jobs[jid]["name"], reason))

        for jid in ready:
            job = self.jobs[jid]
            r = self.role(jid)
            if r is None:
                continue        # config not there yet (generated upstream?) -- retry next tick
            role, key = r
            cands = candidates(job)
            target, copy_from = None, None
            if role == "producer":
                # A (re)trained trajectory: prefer a free machine already holding the key (its
                # copy gets overwritten), else the least-loaded one. Other copies become stale.
                holders = [m for m in self.pins.get(key, []) if m in cands]
                if holders:
                    target = holders[0]
                elif cands:
                    target = min(cands, key=lambda m: (pins_per_machine.get(m, 0), demand.get(m, 0), order[m]))
            elif role == "consumer":
                holders = self.pins.get(key, [])
                live = [m for m in holders if up(m)]
                if not holders:
                    fail(jid, "reads checkpoints %s, but no machine is known to hold them -- run their "
                         "train_expert with EXECUTOR=nfs first, or copy them to %s/%s on a machine and "
                         "`nfs_scheduler.py affinity --pin %s <machine>`" % (key, LOCAL_ROOT, key, key))
                    continue
                if not live:
                    since = min((self.minfo.get(m, {}).get("down_since") or now) for m in holders)
                    if now - since > PIN_DOWN_TIMEOUT or not any(m in order for m in holders):
                        fail(jid, "checkpoints %s are only on %s, unusable" % (key, ",".join(holders)))
                    continue
                free_holders = [m for m in live if m in cands]
                if free_holders:
                    target = free_holders[0]
                elif SHARE_EXPERTS and cands:
                    target = min(cands, key=lambda m: (demand.get(m, 0), order[m]))
                    copy_from = live[0]
            elif cands:                      # no checkpoint: keep holders for their own readers
                target = min(cands, key=lambda m: (demand.get(m, 0), order[m]))
            if target is None:
                continue
            if not self.launch(jid, target, key if copy_from else None, copy_from):
                free[target] = 0
                continue
            free[target] -= 1
            if role == "producer":
                if self.pins.get(key) != [target]:
                    self.pins[key] = [target]
                    pins_per_machine[target] = pins_per_machine.get(target, 0) + 1
                    save_affinity(self.pins)
                    log("pinned %s -> %s" % (key, target))

    def register_copy(self, jid):
        """After a job that rsynced an expert: its machine now holds that expert too."""
        st = self.state.get(jid, {})
        key, machine = st.get("copy_key"), st.get("machine")
        if not key or not os.path.exists(os.path.join(RUN_DIR, "%d.copied" % jid)):
            return
        holders = self.pins.setdefault(key, [])
        if st.get("copy_from") in holders and machine not in holders:   # not re-trained meanwhile
            holders.append(machine)
            save_affinity(self.pins)
            log("%s copied to %s (holders: %s)" % (key, machine, ",".join(holders)))

    # -- main loop --------------------------------------------------------------------------
    def run(self):
        log("daemon start on %s pid %d, state %s, local root %s, machines file %s"
            % (socket.gethostname(), os.getpid(), STATE_DIR, LOCAL_ROOT, MACHINES_FILE))
        self.reprobe(self.machines)
        self.last_reprobe = time.time()
        while True:
            now = time.time()
            self.load_new_jobs()
            self.handle_requests()
            self.collect_finished()
            if now - self.last_liveness > LIVENESS_EVERY:
                self.liveness()
                self.last_liveness = now
            if now - self.last_reprobe > REPROBE_EVERY:
                down = [m for m in self.machines if not self.minfo.get(m, {}).get("up")]
                if down:
                    self.reprobe(down)
                self.last_reprobe = now
            ready = self.resolve()
            self.dispatch(ready)

            active = any(self.state.get(j, {}).get("state") not in FINAL for j in self.jobs)
            if active:
                self.idle_since = None
            else:
                self.idle_since = self.idle_since or now
                if now - self.idle_since > IDLE_EXIT:
                    log("nothing to do for %ds -- exiting" % IDLE_EXIT)
                    return
            time.sleep(POLL)


def cmd_run(a):
    ensure_dirs()
    alive = daemon_alive()
    if alive and alive != (socket.gethostname(), os.getpid()):
        raise SystemExit("nfs_scheduler: a daemon is already running on %s (pid %d)" % alive)
    with open(PIDFILE, "w") as f:
        f.write("%s %d\n" % (socket.gethostname(), os.getpid()))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        Daemon().run()
    finally:
        if daemon_info() == (socket.gethostname(), os.getpid()):
            os.remove(PIDFILE)


def cmd_start(a):
    ensure_dirs()
    alive = daemon_alive()
    if alive:
        print("[nfs] scheduler already running on %s (pid %d)" % alive, file=sys.stderr)
        return
    # Fully detached: no inherited stdout, so a `$(...)` around the submit call never waits on it.
    logf = open(DAEMON_LOG, "a")
    p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "run"], stdin=subprocess.DEVNULL,
                         stdout=logf, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    with open(PIDFILE, "w") as f:
        f.write("%s %d\n" % (socket.gethostname(), p.pid))
    print("[nfs] scheduler started on %s (pid %d), log: %s" % (socket.gethostname(), p.pid, DAEMON_LOG),
          file=sys.stderr)


def cmd_stop(a):
    alive = daemon_alive()
    if a.kill_jobs:
        request(ids=None, kill=True)
        if alive:
            time.sleep(POLL + 5)    # let the daemon process the kill request
        else:
            cmd_cancel(argparse.Namespace(ids=[], all=True, kill=True))
    if not alive:
        print("[nfs] no scheduler running")
        return
    host, pid = alive
    if host == socket.gethostname():
        os.kill(pid, signal.SIGTERM)
    else:
        ssh(host, "kill -TERM %d" % pid)
    print("[nfs] scheduler on %s (pid %d) stopped -- running jobs keep running; `start` picks them up again"
          % (host, pid))


def request(ids, kill):
    ensure_dirs()
    write_json_atomic(os.path.join(REQ_DIR, "%d_%d.json" % (time.time() * 1e6, os.getpid())),
                      {"all": ids is None, "ids": ids or [], "kill": kill})


def cmd_cancel(a):
    if not a.ids and not a.all:
        raise SystemExit("give job ids or --all")
    if daemon_alive():
        request(None if a.all else a.ids, a.kill)
        print("[nfs] cancel request queued for the scheduler")
        return
    # No daemon: apply directly (only pending jobs can be cancelled without it).
    d = Daemon()
    d.load_new_jobs()
    targets = list(d.jobs) if a.all else a.ids
    for jid in targets:
        st = d.state.get(jid, {})
        if jid in d.jobs and st.get("state") not in FINAL:
            if st.get("state") == "running" and not a.kill:
                continue
            if st.get("state") == "running" and st.get("pid"):
                ssh(st["machine"], "kill -TERM -- -%d 2>/dev/null; kill -TERM %d 2>/dev/null"
                    % (st["pid"], st["pid"]))
            d.set_state(jid, state="cancelled", machine=st.get("machine"), reason="cancelled by user")
    print("[nfs] cancelled")


def cmd_status(a):
    ensure_dirs()
    d = Daemon()
    d.load_new_jobs()
    alive = daemon_alive()
    print("scheduler: %s" % ("running on %s (pid %d)" % alive if alive else "NOT running"))
    print("state dir: %s   local root: %s" % (STATE_DIR, LOCAL_ROOT))
    counts = {}
    for jid in d.jobs:
        s = d.state.get(jid, {}).get("state", "pending")
        counts[s] = counts.get(s, 0) + 1
    print("jobs: " + ", ".join("%s=%d" % kv for kv in sorted(counts.items())) if counts else "jobs: none")
    running = sorted((j for j in d.jobs if d.state.get(j, {}).get("state") == "running"))
    for jid in running[: None if a.verbose else 20]:
        st = d.state[jid]
        print("  running %6d %-60s %s (%dmin)" % (jid, d.jobs[jid]["name"][:60], st["machine"],
                                                   (time.time() - st.get("started", time.time())) / 60))
    if len(running) > 20 and not a.verbose:
        print("  ... %d more (-v)" % (len(running) - 20))
    failed = [j for j in sorted(d.jobs) if d.state.get(j, {}).get("state") == "failed"]
    for jid in failed[: None if a.verbose else 20]:
        print("  FAILED  %6d %-60s %s" % (jid, d.jobs[jid]["name"][:60], d.state[jid].get("reason", "")))
        print("          log: %s" % d.jobs[jid]["log"])
    if len(failed) > 20 and not a.verbose:
        print("  ... %d more failures (-v)" % (len(failed) - 20))
    print("pinned checkpoint dirs: %d (see `affinity`)" % len(d.pins))


def cmd_probe(a):
    machines = read_machines()
    res = parallel(probe, machines)
    for m in machines:
        r = res[m]
        print("%-12s %s" % (m, ("ok, %d GPU" % r["gpus"]) if r["up"] else "DOWN: " + r["why"]))
    print("%d/%d usable" % (sum(r["up"] for r in res.values()), len(machines)))


def cmd_affinity(a):
    ensure_dirs()
    pins = load_affinity()
    if a.pin or a.unpin:
        if daemon_alive():
            raise SystemExit("stop the scheduler first (it owns affinity.tsv while running)")
        if a.pin:
            pins.setdefault(os.path.normpath(a.pin[0]), [])
            if a.pin[1] not in pins[os.path.normpath(a.pin[0])]:
                pins[os.path.normpath(a.pin[0])].append(a.pin[1])
        for k in a.unpin or []:
            pins.pop(os.path.normpath(k), None)
        save_affinity(pins)
    for k in sorted(pins):
        print("%s\t%s" % (k, ",".join(pins[k])))


def cmd_clean(a):
    """Shows (default) or deletes $LOCAL_ROOT[/SUB] on each machine, then drops matching pins."""
    sub = os.path.normpath(a.path) if a.path else ""
    if sub.startswith("/") or sub == ".." or sub.startswith("../") or "/../" in sub:
        raise SystemExit("--path must be relative to %s" % LOCAL_ROOT)
    if sub == ".":
        sub = ""
    root = os.path.normpath(LOCAL_ROOT)
    if not os.path.isabs(root) or root.count("/") < 2:
        raise SystemExit("refusing: NFS_LOCAL_ROOT=%r looks too broad" % LOCAL_ROOT)
    target = os.path.join(root, sub) if sub else root
    machines = a.machines.split() if a.machines else read_machines()

    if a.yes:
        if daemon_alive() and not a.force:
            raise SystemExit("the scheduler is running -- stop it (or let it finish) first, or --force")
        d = Daemon()
        d.load_new_jobs()
        busy = {st["machine"] for st in d.state.values() if st.get("state") == "running"}
        skipped = [m for m in machines if m in busy]
        if skipped and not a.force:
            print("skipping machines with a running job: %s (--force to delete anyway)" % " ".join(skipped))
            machines = [m for m in machines if m not in busy]
        command = "rm -rf %s && echo deleted" % shlex.quote(target)
    else:
        command = "if [ -e %s ]; then du -sh %s | cut -f1; else echo -; fi" % ((shlex.quote(target),) * 2)

    res = parallel(lambda m: ssh(m, command, timeout=600), machines)
    done = []
    for m in machines:
        rc, out, err = res[m]
        if rc != 0:
            print("%-12s unreachable / error: %s" % (m, (err.strip().splitlines() or ["?"])[-1]))
            continue
        print("%-12s %s" % (m, out.strip()))
        done.append(m)

    if not a.yes:
        print("\n(listing only: %s on each machine -- add --yes to delete)" % target)
        return
    pins = load_affinity()
    dropped = 0
    for k in list(pins):
        if not sub or k == sub or k.startswith(sub + "/"):
            kept = [m for m in pins[k] if m not in done]
            dropped += len(pins[k]) - len(kept)
            if kept:
                pins[k] = kept
            else:
                del pins[k]
    save_affinity(pins)
    print("dropped %d checkpoint copy record(s)" % dropped)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd")
    sp.required = True

    for name in ("submit", "submit-batch"):
        p = sp.add_parser(name)
        p.add_argument("--cwd", default=BASE_DIR)
        p.add_argument("--dep", default="")
        p.add_argument("--gpus", type=int, default=1)
        p.add_argument("--path-prepend", default="")
        if name == "submit":
            p.add_argument("--name", required=True)
            p.add_argument("--cmd", default="")
            p.add_argument("--log", default="/dev/null")
            p.add_argument("--done-file", default="")
            p.add_argument("--barrier", action="store_true")
            p.set_defaults(fn=cmd_submit)
        else:
            p.add_argument("--log-dir", required=True)
            p.set_defaults(fn=cmd_submit_batch)

    sp.add_parser("run").set_defaults(fn=cmd_run)
    sp.add_parser("start").set_defaults(fn=cmd_start)
    p = sp.add_parser("stop")
    p.add_argument("--kill-jobs", action="store_true")
    p.set_defaults(fn=cmd_stop)
    p = sp.add_parser("status")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(fn=cmd_status)
    p = sp.add_parser("cancel")
    p.add_argument("ids", nargs="*", type=int)
    p.add_argument("--all", action="store_true")
    p.add_argument("--kill", action="store_true", help="also kill running jobs")
    p.set_defaults(fn=cmd_cancel)
    sp.add_parser("probe").set_defaults(fn=cmd_probe)
    p = sp.add_parser("affinity")
    p.add_argument("--pin", nargs=2, metavar=("KEY", "MACHINE"))
    p.add_argument("--unpin", nargs="+", metavar="KEY")
    p.set_defaults(fn=cmd_affinity)
    p = sp.add_parser("clean")
    p.add_argument("--path", default="", help="sub-path of NFS_LOCAL_ROOT, e.g. out/checkpoints")
    p.add_argument("--machines", default="", help="space-separated (default: the machines file)")
    p.add_argument("--yes", action="store_true", help="actually delete")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_clean)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
