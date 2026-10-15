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

## Supported datasets

CIFAR-10, CIFAR-100, SVHN, Tiny ImageNet.
