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

## Results

`scripts/show_results/plot_paper_main_campaign_user.py`, `plot_trigger_grid.py`,
`show_results.py`: aggregate CTA/ASR across seeds and produce the paper's main-campaign
tables/figures.

## Supported datasets

CIFAR-10, CIFAR-100, SVHN, Tiny ImageNet.
