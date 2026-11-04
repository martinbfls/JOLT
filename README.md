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

### Trigger mechanisms: additive, warp, hybrid

`generate_labels_trigger_joint`'s `trigger_type` picks what the joint optimization actually
optimizes (schema: `schemas/generate_labels_trigger_joint.toml`):

- `additive` (default): the pixel perturbation `delta`, applied as `clamp(x + delta, 0, 1)` under
  `||delta||_inf <= epsilon`.
- `warp`: a WaNet-style control grid of pixel displacements, under `||control_grid||_inf <= warp_s`.
- `hybrid`: **both, composed and optimized jointly** --
  `clamp(apply_warp(x, control_grid, warp_s) + delta, 0, 1)`. The two are separate leaf variables in
  one Adam, each with its own learning rate (`lr_delta`, `lr_warp`) and its own budget (`epsilon`,
  `warp_s`); every term that reaches the trigger (the MTT matching term, `L_bd`, `L_joint`,
  `L_lpips`, ...) backpropagates into both, so the optimization splits the perturbation between
  them. The warp comes first, so the additive pattern lands unaltered on the warped image and
  `epsilon` stays an exact `Linf` budget on it; zeroing either component reduces the trigger exactly
  to the other type. The intended use is a **small `epsilon`** (the visible part) with the warp
  carrying the rest -- the additive trigger's ASR at a budget it cannot reach alone.

Each type is deployed with its own poisoner flag (`optimized` / `warp` / `hybrid`,
`POISONER_BY_TRIGGER_TYPE` in `modules/base_utils/gen_configs.py`), which `gen_configs.py` writes
into the `federated_train_user` configs for you. Per-component instrumentation is logged
throughout: `delta_l2`/`delta_linf`/`mtt_delta_grad_norm` for the additive half,
`warp_l2`/`warp_linf`/`warp_tv`/`mtt_warp_grad_norm` for the warp one.

The composition itself lives in `modules/optimizing_trigger/compose.py`
(`apply_trigger_raw` — single source of truth, shared by the optimizer, the LPIPS penalty and the
deploy-time poisoners). That module imports torch and nothing else from the project, so
`python scripts/check_trigger_composition.py` verifies it without a dataset or a checkpoint; CI
runs it to guard the guarantee that every knob below, at its default, leaves the composition
bit-identical to what produced the results already on disk.

### Stealth knobs: warp smoothness, EOT, photometric map, contrast masking

Four further knobs buy expressivity or signal at (near) zero compute, all **off by default** so
existing campaigns stay comparable:

- `lambda_tv_warp`: total variation of the **warp control grid** (`lambda_tv` only ever applied to
  the additive delta). Perceived warp visibility tracks how fast the displacement field *varies*,
  not its amplitude, so `warp_s` alone is a poor visibility handle and leaves `warp_k` pinned at
  its 32-DOF default. This term is what makes a larger `warp_k` (128 DOF at k=8, 512 at k=16)
  usable at a **smaller** `warp_s` — the campaigns' `warp_s=8` on 32x32 images is ~16x WaNet's
  own displacement budget at the same `warp_k=4`.
- `eot_augment`: one draw of the **victim's** train-time augmentation (`RandomCrop(32, padding=4)`
  + `RandomHorizontalFlip`) applied to the composed triggered image. The matched experts were
  trained on triggered images *under* that augmentation while the optimization loop fed them
  un-augmented ones. Costs no extra forward pass, and matters most for the warp, which a flip
  mirrors (`(dx, dy) -> (-dx, dy)`).
- `color_trigger`: a **third mechanism**, a global 3x4 affine photometric map (12 parameters,
  budgets `color_a_max`/`color_b_max`). Perceptually cheap, and spatially uniform — hence
  invariant to the victim's crops and flips.
- `contrast_mask_floor`: makes the additive budget **per-pixel**, proportional to local contrast,
  so `epsilon` is spent where the perturbation hides rather than uniformly (a flat sky costs as
  much budget as a texture but shows far more). Zero extra parameters; the weight stays `<= 1`, so
  `epsilon` remains a valid bound on the deployed perturbation.

`color_trigger` and `contrast_mask_floor` require `--trigger-type hybrid`: they are carried by the
hybrid artifact's dict and rebuilt by `HybridPoisoner`, so under any other type they would be
optimized and then silently dropped (both `gen_configs.py` and `run_module.py` refuse instead).
Older saved triggers deploy unchanged — the poisoner's fallbacks are these knobs' "off" values.

**What more trigger expressivity can and cannot do.** At deploy time `federated_train_user` builds
each worker's dataset from **clean images with flipped labels**; the trigger only ever reaches the
ASR test set. The robust aggregator therefore never sees a triggered gradient, and no trigger —
however expressive — evades Krum directly. The only indirect path is a trigger that needs *fewer
flips* to be induced, which leaves the poisoned clients' gradients closer to the honest ones. So
read these knobs at the **low** budgets, and spend expressivity on directions that are
perceptually cheap yet simple enough in feature space for label flips on clean data to reproduce.
`scripts/show_results/diagnose_trigger_capacity.py` (below) is what tells you which regime you are
actually in.

```bash
# the ablation: base -> +warp TV/k8 -> +EOT -> +contrast mask -> +photometric map,
# every variant from scratch, both deployments, then the verdict and the Pareto table.
# REFERENCE_TAGS brings an existing campaign into the same comparison and backfills
# whatever it never ran (no EXPERT/GEN re-run, no finished cell recomputed);
# SPLIT_BATCH=1 is the convention to use for anything reported (see below).
SPLIT_BATCH=1 EPSILON=0.0627 REFERENCE_TAGS="hybrid_eps0p0627_s8p0" \
    SLURM_ACCOUNT=<account> ./orchestrate_slurm/ablate_trigger_stealth_slurm.sh
```

Budget 0 is in the default budget list as a **control**, not as a data point: with no label
flipped, whatever ASR the trigger reaches there is the trigger acting as a plain adversarial
perturbation rather than an induced backdoor. The diagnosis below reports every other budget net
of it, and variants must be ranked on that net figure.

```bash
# the whole epsilon x seed grid, every budget, both deployments and the comparison figure,
# in one command (see the script's header for the knobs)
EPSILON_VALUES="0.5 0.0627" WARP_S_VALUES="8.0" SEEDS="0 1 2" \
    SLURM_ACCOUNT=<account> ./orchestrate_slurm/sweep_hybrid_epsilon_slurm.sh

# one cell, hybrid trigger with an 8/255 additive budget plus a 4px warp
python -m modules.generate_labels_trigger_joint.gen_configs --single-cell \
    --trigger-type hybrid --epsilon 0.0314 --warp-s 4.0 --exp-base-tag hybrid --budgets 1500

# same, submitted on Slurm (centralized + multikrum deploy of the one trigger)
TRIGGER_TYPE=hybrid EPSILON=0.0314 WARP_S=4.0 BUDGETS=1500 \
    SLURM_ACCOUNT=<account> ./orchestrate_slurm/run_warp_test_vs_multikrum_slurm.sh

# three-way A/B/C at a shared additive budget, then the side-by-side figure
VARIANTS="additive warp hybrid" EPSILON=0.0314 WARP_S=4.0 BUDGETS=1500 \
    SLURM_ACCOUNT=<account> ./orchestrate_slurm/compare_additive_warp_slurm.sh
```

`scripts/show_results/visualize_trigger.py --trigger-type hybrid` previews the composed image
alongside each raw component (the normalized delta and the displacement field), and
`compare_additive_warp.py --hybrid-dir ...` adds the hybrid run to the additive-vs-warp comparison.

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

### Warp multi-budget campaign (ASR/CTA vs budget per variant)

`scripts/show_results/plot_warp_multibudget.py` plots what
`orchestrate_slurm/run_warp_multibudget_vs_multikrum_slurm.sh` produces: ASR and CTA vs flip
budget on the federated deploy branch, mean +- std across seeds, one line per `VARIANT_TAG`.
It writes `asr_cta_vs_budget.{png,pdf}` (both panels), the per-metric figures, a long-format
`per_seed.csv` and a `summary.{md,csv}` table, and lists the cells that have no results yet:

```bash
python scripts/show_results/plot_warp_multibudget.py \
    --variant-tags warp_s8_lpips0p0 warp_s16_lpips0p0 \
    --seeds 0 1 2 3 4 --budgets 0 150 300 500 1000 2000 2500 5000
python scripts/show_results/plot_warp_multibudget.py --per-seed   # each seed as a thin line too
```

Pass `--exp-root` to read another results tree (e.g. `experiments/out_iclr`), and
`--deploy-agg-method`/`--deploy-num-poisoned`/`--deploy-num-honests` for a different deployment
than the `multikrum_3vs7` default.

### All of a campaign's results in one command

`scripts/show_results/run_all_campaign.sh` is the counterpart of `run_all_out_iclr.sh` for the
trigger campaigns: it produces **every visualization `run_all_out_iclr.sh` does**, plus the two
specific to these campaigns, into `$OUT_ROOT/`.

```bash
VARIANT_TAGS="hybrid_eps0p25_s8p0 hybrid_eps0p0627_s8p0" SEEDS="0 1 2" \
    SLURM_ACCOUNT=<account> ./scripts/show_results/run_all_campaign.sh
```

It submits itself as one CPU-only job on `SLURM_PARTITION` — the campaign's own partition, since
`pine` accepts `srun` from an RDHPC node but refuses `sbatch` from one — so nothing runs on the
login node. It executes inline instead when already inside an allocation (`SLURM_JOB_ID` set) or
with `SUBMIT=0`. A step that aborts (a running campaign has missing cells) never hides the others;
the exit status counts the failures.

The four `out_iclr` scripts only read the `paper_main_campaign` tree layout, so rather than
teaching them a second one, step 3 builds a **symlink view** of the campaign in that shape
(`make_paper_layout_view.py` — no copying, rebuilt on every run) and the rest run against it with
`--jolt-root`.

Everything lands in one directory per campaign, split by the axes that identify a result:

```
$OUT_ROOT/<campaign>/<model>/<dataset>/<deploy branch>/
    vs_budget/  asr_dynamics/  paper_main_campaign_user/  tables/  results_csv/
    per_seed/   flip_matrix/   trigger_grid/              _paper_layout_view/
```

`CAMPAIGN` defaults to the variant tag (tags joined by `+` when there are several); set it
explicitly when two runs share tags but differ some other way — a `SPLIT_BATCH=1` re-deploy, say
— so they do not overwrite each other. `OUT_ROOT` defaults to
`experiments/federated_experiments/campaign_results`, and the orchestrators pass both through.

`sweep_hybrid_epsilon_slurm.sh` and `run_warp_multibudget_vs_multikrum_slurm.sh` both gate a
final Slurm job on it, so a campaign produces its own detailed results without a second command.

### Federated victim: optimizer-step count

`federated_train_user`'s `batch_size` is **per worker**, so at equal `epochs` a federated victim
takes `num_workers` times fewer optimizer steps than the centralized one while each worker holds
`1/num_workers` of the data — on CIFAR-10/r32p with 10 workers, 20 steps per epoch against 196,
and a victim that ends up visibly under-trained (~87.5% vs ~92.6% CTA at budget 0; the published
ResNet-32 figure is ~92.5%). Every federated number is then confounded with under-training.

`split_batch_across_workers = true` makes `batch_size` the **total** instead
(`batch_size // num_workers` each), restoring both the step count and the effective batch. It is
off by default — turning it on changes every federated number, so a campaign must not mix the two
conventions. Pass `--split-batch-across-workers` to `gen_configs.py`, or `SPLIT_BATCH=1` to the
Slurm orchestrators; send its results to their own `RESULTS_OUT`, since they are not comparable
with a cell's existing ones.

### Which bottleneck? (trigger capacity vs. inducibility vs. aggregation)

`scripts/show_results/diagnose_trigger_capacity.py` puts three ASRs side by side and names the
bottleneck, because the three call for opposite work and the deployed ASR alone cannot tell them
apart:

| measurement | what it is | what it bounds |
|---|---|---|
| trigger ceiling | `expert_asr` / `retrain_expert_asr` from the GEN phase — an expert trained **with** triggered images | how well this perturbation can backdoor this architecture at all |
| centralized ASR | the `1vs0`/`mean` branch, i.e. a no-op aggregator | whether the **label flips** can reproduce the trigger's feature at that budget |
| federated ASR | the real `multikrum_3vs7` deployment | what the aggregator leaves |

A low ceiling — and only that — means the trigger itself is the bottleneck and capacity is worth
adding. A large ceiling-to-centralized gap is an **inducibility** failure: more expressivity makes
it worse, since a more complex perturbation is harder for label flips to reproduce, not easier. A
large centralized-to-federated gap is an **aggregation** failure, where the levers are the flip
budget and `lambda_gradmatch`/`lambda_match`/`lambda_consistency`, never the trigger.

`scripts/show_results/measure_trigger_stealth.py` is the other half: it rebuilds each saved trigger
with its real poisoner, applies it to source-class test images, and measures LPIPS/SSIM/PSNR/Linf
of the perturbation **as deployed** (the PIL round-trip included, so 1/255 quantization counts),
then joins that with ASR and draws the Pareto view. `epsilon` cannot rank these variants — it
bounds the additive half only, says nothing about a warp or a photometric map, and is merely an
upper bound once `contrast_mask_floor` is on.

```bash
python scripts/show_results/diagnose_trigger_capacity.py \
    --variant-tags ablate_base ablate_tv --seeds 0 1 2 --budgets 150 500 1500
python scripts/show_results/measure_trigger_stealth.py \
    --variant-tags ablate_base ablate_tv --seeds 0 1 2 --budgets 150 500 1500

# both of the above, plus vs-budget, ASR dynamics and the campaign figures, in one command
SEEDS="0 1 2" BUDGETS="150 500 1500" ./scripts/show_results/analyze_trigger_ablation.sh
```

### ASR dynamics (did the backdoor survive, or just peak?)

`scripts/show_results/analyze_asr_dynamics.py` reads the per-epoch series instead of the final
value, to separate a backdoor that holds from one that spikes mid-run and is then unlearned:

```bash
python scripts/show_results/analyze_asr_dynamics.py \
    --variant-tags hybrid_eps0p25_s8p0 hybrid_eps0p0627_s8p0 --seeds 0 1 2
```

It writes `asr_vs_epoch_<tag>_<branch>.{png,pdf}` (one panel per budget, ASR solid, CTA dashed,
LR drops marked — detected from the run's own `lr` series, not assumed), `transience.{png,md,csv}`
(peak vs final ASR per cell, flagging those that lose more than `--transient-threshold` points),
a long-format `dynamics.csv`, and `diagnosis_<tag>_<branch>.{png,pdf}`, which puts ASR, train
accuracy and `cos(poisoned, honest)` on one epoch axis.

That last figure is the point. The victim trains on **clean images with flipped labels** (the
trigger only ever reaches the ASR test set), so the flipped examples are label noise, and the two
usual causes of a collapse leave different traces: ASR falling at an LR milestone *while train
accuracy keeps climbing* points at the network memorizing the flips one by one instead of
explaining them with the shared feature the trigger rides; ASR falling *together with* the
gradient cosine points at the robust aggregator filtering the poisoned workers instead.
`--branch both` (the default) reads the centralized deployment too — its aggregator is a no-op
`mean`, so a collapse there rules the aggregator out.

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
