# JOLT

Code for a federated-learning label-flipping backdoor attack that jointly optimizes a
poisoning trigger and the label-flip strategy under robust aggregation. Built on top of the
official implementation of FLIP (NeurIPS 2023, https://arxiv.org/abs/2310.18933,
https://github.com/SewoongLab/FLIP), which is included as the non-federated baseline pipeline.

Robust-aggregation code is adapted from *Distributed Momentum for Byzantine-Resilient
Learning* (https://arxiv.org/abs/2003.00010,
official code: https://github.com/LPD-EPFL/ByzantineMomentum).

## Repository layout

- `modules/`: one subpackage per pipeline stage (see below)
- `schemas/`: one `<module_name>.toml` per module, documenting every config key
- `experiments/`: config trees and outputs for each experiment, one subdirectory per campaign
- `scripts/`: campaign-independent utilities (results aggregation, plots)
- `orchestrate_slurm/`, `orchestrate_student_machines/`: campaign launchers (Slurm cluster / SSH pool)
- `tools/`, `modules/base_utils/aggregator/`: robust-aggregation rules (adapted, see above)
- `modules/base_utils/robust_agg.py`: full-gradient rules (`krum_full`, `multikrum_full`, `geomed`) and NNM pre-aggregation (`pre_agg = "nnm"`), usable as `agg_method`/`pre_agg` in `federated_train_user` and `generate_labels_trigger_joint` configs

## Installation

Requires Python 3.9+.

```bash
pip install -r requirements.txt
```

## Pipelines

### FLIP baseline (non-federated)

`train_expert -> generate_labels -> select_flips -> train_user`, following the schemas
`train_expert.toml`, `generate_labels.toml`, `select_flips.toml`, `train_user.toml`. Each
stage is run with:

```bash
python run_experiment.py <experiment_name>
```

against a `config.toml` under `experiments/<experiment_name>/` with one `[<module_name>]`
table per stage.

### Main campaign: joint trigger/label optimization under federated aggregation

`train_expert -> generate_labels_trigger_joint -> federated_select_flips ->
federated_train_user`, corresponding to the paper's main results
(`experiments/federated_experiments/threat_model_direct_trigger_joint_paper_main_campaign/`).
`generate_labels_trigger_joint` jointly optimizes a continuous label assignment and
a backdoor trigger via trajectory matching against federated expert gradients (schema:
`generate_labels_trigger_joint.toml`, algorithm in
`modules/generate_labels_trigger_joint/run_module.py`).

Regenerate this campaign's configs, then run each stage with `run_experiment.py`:

```bash
python -m modules.generate_labels_trigger_joint.gen_configs_paper_main_campaign --print-grid  # inspect the grid
python -m modules.generate_labels_trigger_joint.gen_configs_paper_main_campaign --dry-run      # validate without writing
python -m modules.generate_labels_trigger_joint.gen_configs_paper_main_campaign
```

### FLIP with a frozen JOLT trigger

Re-runs the FLIP pipeline (`train_expert -> generate_labels -> federated_select_flips ->
federated_train_user`) with each paper-campaign cell's jointly optimized trigger held fixed, so
JOLT can be compared with FLIP at equal trigger. Same `{model}/{dataset}/{tag}/seed{seed}/`
layout, under `experiments/federated_experiments/flip_frozen_trigger_paper_main_campaign/`;
needs the joint campaign's GEN phase (its triggers) first.

```bash
python -m modules.generate_labels_trigger_joint.gen_configs_flip_frozen_trigger                  # all cells
python -m modules.generate_labels_trigger_joint.gen_configs_flip_frozen_trigger --models r32p --datasets cifar --tags baseline --seeds 0 1
SLURM_ACCOUNT=<account> MODEL_FLAGS=r32p DATASETS=cifar STEALTH_TAGS=baseline SEEDS="0 1" \
    ./orchestrate_slurm/orchestrate_runs_flip_frozen_trigger_slurm.sh
python scripts/show_results/plot_paper_main_campaign_user.py --flip-frozen-tags baseline eps_16_255
```

The student-machine flips/user scripts run it with
`EXP_BASE_REL=federated_experiments/flip_frozen_trigger_paper_main_campaign`.

### Launching a campaign

Run from the repo root, after generating the configs. Every script supports `DRY_RUN=1`
(prints the job list, submits nothing) and documents its env overrides in its header.

- Slurm (`orchestrate_slurm/`, shared helpers in `slurm_lib.sh`: account, partitions, conda env):
  ```bash
  SLURM_ACCOUNT=<account> ./orchestrate_slurm/orchestrate_runs_trigger_joint_paper_main_campaign_slurm.sh
  SLURM_ACCOUNT=<account> ./orchestrate_slurm/orchestrate_runs_trigger_joint_slurm.sh  # gen_configs.py validation run
  ./orchestrate_slurm/kill_all_slurm.sh
  ```
- Student machines over SSH (`orchestrate_student_machines/`, repo on the shared NFS home): run
  BOOTSTRAP/GEN on Slurm with `SKIP_USER=1`, rsync `experiments/`, then
  ```bash
  TAG=baseline ./orchestrate_student_machines/orchestrate_runs_paper_main_campaign_flips.sh
  TAG=baseline ./orchestrate_student_machines/orchestrate_runs_paper_main_campaign_user.sh
  ./orchestrate_student_machines/kill_all.sh
  ```

## BackdoorBench detection (appendix)

`backdoorbench/` runs [BackdoorBench](https://github.com/SCLBD/BackdoorBench)'s backdoor
detection methods on existing JOLT results, without regenerating any attack: `attack/jolt.py`
rebuilds a CIFAR-10 cell's poisoned training set from its single_user `select_flips` output
(flipped labels on clean images) and its trigger, BackdoorBench trains its own victim
(PreAct-ResNet18), then the data-level (`detection_pretrain/`: AC, SS, SPECTRE, SCAn, Beatrix,
AGPD) and inference-time (`detection_infer/`: STRIP, SentiNet, TeCo) detectors run unchanged.
BackdoorBench (CC BY-NC 4.0) stays a separate checkout.

```bash
git clone https://github.com/SCLBD/BackdoorBench ~/BackdoorBench   # + its sh/install.sh
BB_DIR=~/BackdoorBench TAGS=baseline SEEDS=0 BUDGETS=1000 ./backdoorbench/run_backdoorbench.sh
BB_DIR=~/BackdoorBench REFERENCE_ATTACKS="badnet blended" ./backdoorbench/run_backdoorbench.sh
python backdoorbench/collect_backdoorbench.py --bb-dir ~/BackdoorBench    # CSV + appendix LaTeX table
```

## Backing up results

`make backup` uploads `experiments/` to a private Hugging Face dataset (`<hf user>/jolt-results`,
override with `HF_REPO=user/name`), incrementally and resumably; `make restore` downloads it back.
Run it from a node with internet access, after `pip install huggingface_hub` and `hf auth login`
(write token). `make backup-dry` shows what would be uploaded; `make backup BACKUP_ARGS=--checkpoints`
also backs up `out/checkpoints/` to `<hf user>/jolt-checkpoints`. select_flips' `worker*/` files
are skipped (regenerable in minutes from the gen labels; `--keep-workers` to keep them). See
`scripts/backup_results.py`.

## Results

`scripts/show_results/plot_paper_main_campaign_user.py`, `plot_trigger_grid.py`,
`show_results.py`: aggregate CTA/ASR across seeds and produce the paper's main-campaign
tables/figures.

### Per-seed results

`scripts/show_results/plot_asr_per_seed.py` plots ASR (or `--metric cta`) vs budget with one line
per seed for every aggregator, plus a seed x budget heatmap and a long-format CSV of every cell,
to inspect the across-seed variance (e.g. the nonzero ASR at budget 0):

```bash
python scripts/show_results/plot_asr_per_seed.py --tags baseline eps_16_255
python scripts/show_results/plot_asr_per_seed.py --campaign flip_frozen
```

### Flip matrices (which class goes to which)

`scripts/show_results/plot_flip_matrix.py` draws confusion-style matrices of the selected label flips
from `select_flips/<budget>.npy` vs `true.npy`: entry (i, j) is the share of true-class-i samples flipped
to class j (diagonal masked by default), averaged over seeds, one matrix per budget plus a side-by-side
figure and a long-format CSV:

```bash
python scripts/show_results/plot_flip_matrix.py --tags baseline eps_16_255
python scripts/show_results/plot_flip_matrix.py --campaign flip_frozen --budgets 500 2500 5000 --counts
```

### Training / generation curves

Every module records its metrics with `ExperimentTracker`: `experiments/<run>/logs/<module>_metrics.json`
plus paper-style plots (`.png` + `.pdf`, 300 dpi) in `experiments/<run>/plots/<module>/` -- per-epoch
CTA/ASR for `train_expert`/`train_user` (`cta_asr` figure), and for the label/trigger generation the
experts' ASR along the optimization (current and initial trigger, plus each retrained expert),
losses, trigger norm and alignment. Mean +- std over seeds, overlaying tags and the FLIP
frozen-trigger campaign:

```bash
python scripts/show_results/plot_training_curves.py --stage gen --tags baseline eps_16_255
python scripts/show_results/plot_training_curves.py --stage train_user --agg multikrum --budget 1000 --campaigns jolt flip_frozen
```

## Supported datasets

CIFAR-10, CIFAR-100, SVHN, Tiny ImageNet.
