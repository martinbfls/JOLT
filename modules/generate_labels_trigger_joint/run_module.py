"""jointly optimizes poisoned logit labels (labels_syn)
and a backdoor trigger (delta) via trajectory matching, with delta's gradient flowing
through both the backdoor-efficacy term L_bd and the MTT trajectory-alignment term (param_loss).
The trigger is an additive perturbation, a WaNet-style warp, or both composed (trigger_type)."""

from pathlib import Path
import math
import os
import random
import shutil
import sys
import json
import warnings
import torch
import numpy as np

from modules.base_utils.datasets import (
    POISONER_BY_TRIGGER_TYPE,
    get_matching_datasets,
    pick_poisoner,
    get_n_classes,
)
from modules.base_utils.util import (
    extract_toml,
    get_module_device,
    get_mtt_attack_info,
    load_model,
    either_dataloader_dataset_to_both,
    make_pbar,
    needs_big_ims,
    slurmify_path,
    local_storage_path,
    clf_loss,
    clf_eval,
    softmax,
    total_mse_distance,
    get_train_info,
    mini_train,
)
from modules.base_utils.experiment_tracker import ExperimentTracker
from modules.base_utils import robust_agg
from modules.federated_generate_labels.utils import (
    coalesce_attack_config,
    extract_labels,
    extract_experts,
    sgd_step,
    agg,
)
from modules.train_expert.utils import checkpoint_callback
from modules.optimizing_trigger.utils import (
    get_mu,
    init_delta,
    init_warp_field,
    raw_to_trigger_preprocess,
    get_raw_clean_dataset,
    trigger_penalty,
    trigger_penalty_hinge,
    tv_loss,
)
from modules.generate_labels_trigger_joint.utils import (
    TriggerMTTDataset,
    extract_experts_biased,
    build_expert_pool,
    load_expert_pool,
    directional_floor_penalty,
    magnitude_floor_penalty,
    project_trigger_constraints,
    cosine_to,
    grad_mismatch_penalty,
    grad_cosine_penalty,
    lpips_penalty,
    margin_floor_penalty,
    poison_consistency,
)


# Config keys whose change makes a resume snapshot unusable (different variables, shapes or
# objective); any other difference is only reported. attack_config.iterations may change: that
# is how a finished run is extended.
_RESUME_STRICT_KEYS = (
    "expert_model", "dataset", "source_label", "target_label", "trigger_type", "warp_k",
    "photo_contrast", "photo_brightness", "photo_local_contrast", "photo_local_brightness",
    "photo_local_grid", "color_mix", "kernel_budget", "num_honests", "num_poisoned", "agg_method", "attack",
    "train_pct", "clean_trajectory",
)


def _check_resume_compatible(saved_args, args):
    """Raises if the snapshot was written by an incompatible config, warns on other differences."""
    strict = [k for k in _RESUME_STRICT_KEYS if saved_args.get(k) != args.get(k)]
    if strict:
        raise ValueError(
            "resume_state_path holds a snapshot of an incompatible run (differs on "
            + ", ".join(f"{k}: {saved_args.get(k)!r} -> {args.get(k)!r}" for k in strict)
            + ") -- delete it to start over."
        )

    def _strip(a):
        a = dict(a)
        a["attack_config"] = {k: v for k, v in dict(a.get("attack_config", {})).items() if k != "iterations"}
        return a

    saved, cur = _strip(saved_args), _strip(args)
    other = sorted(k for k in set(saved) | set(cur) if saved.get(k) != cur.get(k))
    if other:
        print(f"WARNING [resume]: config differs from the snapshot's on {other} -- resuming anyway.")


def _rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _set_rng_state(rng):
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if rng["cuda"] is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(rng["cuda"])
        except (RuntimeError, IndexError) as e:  # a different GPU count than the saving machine
            print(f"WARNING [resume]: CUDA RNG state not restored ({e}).")


def run(experiment_name, module_name, **kwargs):
    """Jointly optimizes labels_syn and delta: delta's gradient flows through both L_bd and
    the MTT trajectory-alignment term param_loss. Expert-side gradients are aggregated across
    all federated clients before the differentiable SGD step producing expert_next_param.
    Optional add-ons (all off by default): live expert retraining, gradient-mismatch/
    aggregate-match penalties, an LPIPS penalty, and anti-collapse floors on delta's norm/
    alignment/margin/consistency.

    trigger_type='hybrid' optimizes BOTH trigger mechanisms at once -- the additive delta under
    ||delta||_inf <= epsilon AND the warp control grid under ||control_grid||_inf <= warp_s,
    composed as clamp(apply_warp(x) + delta, 0, 1) -- as two leaf variables in one Adam (own
    learning rate each), fed by the same objective. Every term that reaches the trigger (the MTT
    matching term, L_bd, L_joint, L_lpips, ...) backpropagates into both components, so the
    optimization is free to split the perturbation budget between them: the point is to reach the
    additive trigger's ASR at a visibly smaller epsilon than it needs on its own.

    :param experiment_name: Name of the experiment in configuration.
    :param module_name: Name of the module in configuration.
    :param kwargs: Additional arguments (such as slurm id).
    """

    slurm_id = kwargs.get("slurm_id", None)

    args = extract_toml(experiment_name, module_name)
    # metrics_log_path is this module's own per-step history file (written at the end); the
    # tracker's JSON goes to logs/ instead of overwriting it.
    tracker = ExperimentTracker(
        experiment_name, module_name, args, slurm_id=slurm_id,
        x_label="Optimization step", use_config_metrics_path=False,
    )
    tracker.add_plot_group("asr_all", ["expert_asr", "expert_asr_frozen", "retrain_expert_asr"], ylabel="ASR (%)")
    tracker.add_plot_group("losses", ["matching_term", "L_bd_mean", "L_joint_mean", "L_lpips_mean"], ylabel="Loss")
    tracker.add_plot_group("trigger_norm", ["delta_l2", "delta_linf"], ylabel="Trigger norm")
    tracker.add_plot_group(
        "trigger_alignment", ["cos_target", "cos_source", "cos_delta_to_init"], ylabel="Cosine similarity",
    )

    input_pths = local_storage_path(args["input_pths"])
    opt_pths = local_storage_path(args["opt_pths"])
    expert_model_flag = args["expert_model"]
    dataset_flag = args["dataset"]
    clean_label = args["source_label"]
    target_label = args["target_label"]
    lam = args.get("lambda", 0.0)
    train_pct = args.get("train_pct", 1.0)
    batch_size = args.get("batch_size", None)
    epochs = args.get("epochs", None)
    expert_config = args.get("expert_config", {})
    config = coalesce_attack_config(args.get("attack_config", {}))
    num_honests = args.get("num_honests", 2)
    num_poisoned = args.get("num_poisoned", 1)
    output_dir = slurmify_path(args["output_dir"], slurm_id)
    attack = args.get("attack", "backdoor")
    clean_trajectory = args.get("clean_trajectory", False)
    # gamma_stealth: stealth/backdoor loss weight; 'gamma' is a deprecated alias, kept working.
    if "gamma_stealth" in args:
        gamma_stealth = args["gamma_stealth"]
        if "gamma" in args:
            raise ValueError("Pass exactly one of gamma_stealth or gamma (deprecated alias), not both.")
    elif "gamma" in args:
        warnings.warn(
            "'gamma' is deprecated -- this is a stealth/backdoor loss weight, not "
            "num_poisoned/(num_poisoned+num_honests). Use 'gamma_stealth' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        gamma_stealth = args["gamma"]
    else:
        gamma_stealth = 1.0
    agg_method = args.get("agg_method", "mean")
    pre_agg = args.get("pre_agg", None)
    robust_agg.validate(agg_method, pre_agg)
    if agg_method != "mean" or pre_agg not in (None, "none"):
        print(
            f"WARNING: agg_method={agg_method!r}, pre_agg={pre_agg!r} -- the P^mean objective assumes plain "
            "mean aggregation; a non-mean run confounds the aggregator with the attack factor."
        )

    trigger_type = args.get("trigger_type", "additive")
    if trigger_type not in POISONER_BY_TRIGGER_TYPE:
        raise ValueError(
            f"trigger_type must be one of {sorted(POISONER_BY_TRIGGER_TYPE)}, got {trigger_type!r}"
        )
    # 'hybrid' optimizes both components at once, so it takes both budgets/learning rates and
    # every additive-only term below (epsilon, the floors, the pixel penalties) applies to its
    # additive half -- see this function's docstring.
    is_hybrid = trigger_type == "hybrid"
    has_additive = trigger_type in ("additive", "hybrid")
    has_warp = trigger_type in ("warp", "hybrid")
    warp_k = args.get("warp_k", 4)
    warp_s = args.get("warp_s", 1.0)  # max displacement, in pixels
    if has_warp:
        # the control grid's own norms: a second figure, so the additive runs' trigger_norm one
        # (delta_l2/delta_linf, the additive component under 'hybrid') stays comparable as-is
        tracker.add_plot_group("warp_norm", ["warp_l2", "warp_linf"], ylabel="Warp control grid")

    epsilon = args.get("epsilon", 0.1)
    lr_delta = args.get("lr_delta", 1e-2)
    # the control grid lives in pixels, on a different scale than the additive delta in [0, 1]
    lr_warp = args.get("lr_warp", 1e-1)
    # photometric half of the hybrid trigger: per-channel contrast (gain = 1 + a, |a| <= photo_contrast)
    # and brightness (shift b, |b| <= photo_brightness), applied between the warp and the additive delta.
    # Both budgets at 0 (default) = off, the trigger is then exactly the warp + additive one.
    photo_contrast = args.get("photo_contrast", 0.0)
    photo_brightness = args.get("photo_brightness", 0.0)
    lr_photo = args.get("lr_photo", 1e-2)
    has_photo = photo_contrast > 0 or photo_brightness > 0
    if has_photo and not is_hybrid:
        raise ValueError(
            f"photo_contrast/photo_brightness need trigger_type='hybrid', got {trigger_type!r}"
        )
    # Extra expressive transformations (all hybrid-only, off at 0): each a deviation from the identity,
    # box-bounded by its budget and optimized in the same Adam (lr_ext). Composition order and exact
    # definitions: apply_trigger_raw / apply_kernel / apply_color_mix / apply_photo_local.
    photo_local_grid = args.get("photo_local_grid", 4)
    photo_local_contrast = args.get("photo_local_contrast", 0.0)
    photo_local_brightness = args.get("photo_local_brightness", 0.0)
    color_mix_budget = args.get("color_mix", 0.0)
    kernel_budget = args.get("kernel_budget", 0.0)
    lr_ext = args.get("lr_ext", 1e-2)
    has_photo_local = photo_local_contrast > 0 or photo_local_brightness > 0
    has_color_mix = color_mix_budget > 0
    has_kernel = kernel_budget > 0
    has_ext = has_photo_local or has_color_mix or has_kernel
    if has_ext and not is_hybrid:
        raise ValueError(
            "photo_local_*/color_mix/kernel_budget need trigger_type='hybrid', "
            f"got {trigger_type!r}"
        )
    lambda_bd = args.get("lambda_bd", 1.0)
    lambda_joint = args.get("lambda_joint", 0.0)
    track_joint = lambda_joint != 0
    lambda_penalty = args.get("lambda_penalty", 0.0)
    lambda_delta = args.get("lambda_delta", 0.0)
    lambda_tv = args.get("lambda_tv", 0.0)
    kappa = args.get("kappa", 0.0)
    init = args.get("init", "stripe")

    trigger_constraint = args.get("trigger_constraint", "penalty")
    align_kappa = args.get("align_kappa", 0.6)
    lambda_align = args.get("lambda_align", 0.0)
    lambda_mag = args.get("lambda_mag", 1.0)
    delta_min_frac = args.get("delta_min_frac", 0.8)

    lambda_margin = args.get("lambda_margin", 0.0)
    margin_min = args.get("margin_min", 2.0)
    track_margin = lambda_margin != 0

    lambda_consistency = args.get("lambda_consistency", 0.0)
    track_consistency = lambda_consistency != 0

    expert_retrain_agg_method = args.get("expert_retrain_agg_method", agg_method)

    lambda_balance_interval = args.get("lambda_balance_interval", 0)
    lambda_balance_ratios = args.get(
        "lambda_balance_ratios",
        {
            "L_bd": 0.45,
            "L_consistency": 0.20,
            "L_gradmatch": 0.10,
            "L_lpips": 0.05,
        },
    )
    lambda_balance_min = args.get("lambda_balance_min", 1e-4)
    lambda_balance_max = args.get("lambda_balance_max", 1e4)

    lambda_gradmatch = args.get("lambda_gradmatch", 0.0)
    gradmatch_eps = args.get("gradmatch_eps", 1e-8)
    gradmatch_metric = args.get("gradmatch_metric", "relerr")
    if gradmatch_metric not in ("relerr", "cosine"):
        raise ValueError(f"gradmatch_metric must be 'relerr' or 'cosine', got {gradmatch_metric!r}")
    track_gradmatch = lambda_gradmatch != 0

    lambda_match = args.get("lambda_match", 0.0)
    track_match = lambda_match != 0

    # Per-poisoned-client grad of L_bd w.r.t. the expert params, with create_graph=True (a
    # second-order graph over the whole expert): only L_gradmatch, L_match and L_consistency
    # read it, so it is skipped when all three are off.
    track_poison_grad = track_gradmatch or track_match or track_consistency

    lambda_lpips = args.get("lambda_lpips", 0.0)
    lpips_model = None
    if lambda_lpips > 0:
        import lpips as _lpips_pkg

        lpips_model = _lpips_pkg.LPIPS(net="alex")
        lpips_model.eval()
        for p in lpips_model.parameters():
            p.requires_grad_(False)

    detach_param_dist = args.get("detach_param_dist", False)

    output_dir_trigger = slurmify_path(
        args.get("output_dir_trigger", "optimized_trigger"),
        slurm_id,
    )

    checkpoint_sampling = args.get("checkpoint_sampling", "biased")
    alpha_ckpt = args.get("alpha_ckpt", 0.01)

    expert_retrain_interval = args.get("expert_retrain_interval", 0)
    expert_retrain_epochs = args.get("expert_retrain_epochs", 20)
    expert_retrain_checkpoint_iters = args.get("expert_retrain_checkpoint_iters", 50)
    expert_retrain_optim_kwargs = args.get("expert_retrain_optim_kwargs", {})
    expert_retrain_scheduler_kwargs = args.get("expert_retrain_scheduler_kwargs", {})
    seed = args.get("seed", None)

    pool_size = args.get("pool_size", 15)

    n_checkpoints_per_step = args.get("n_checkpoints_per_step", 1)
    if n_checkpoints_per_step < 1:
        raise ValueError(f"n_checkpoints_per_step must be >= 1, got {n_checkpoints_per_step}")
    if n_checkpoints_per_step > 1 and pool_size == 1:
        raise ValueError(
            "n_checkpoints_per_step > 1 requires pool_size > 1 (checkpoint pooling must be "
            "enabled to draw multiple DISTINCT checkpoints per step) -- got pool_size=1."
        )

    metrics_log_path = args.get("metrics_log_path", None)
    if metrics_log_path is not None:
        metrics_log_path = slurmify_path(metrics_log_path, slurm_id)
    history = [] if metrics_log_path else None

    # Per-outer-iteration snapshot of the whole optimization (labels, trigger, both optimizers,
    # RNGs, metric histories), overwritten after every iteration: a rerun of the same config
    # resumes from it instead of starting over, and raising attack_config.iterations afterwards
    # continues a finished run. Kept with the cell's outputs (not on the machine-local disk), so a
    # resumed run may land on any machine. Unset = no snapshot, no resume.
    resume_state_path = args.get("resume_state_path", None)
    if resume_state_path is not None:
        resume_state_path = Path(slurmify_path(resume_state_path, slurm_id))
    resume_state = None
    if resume_state_path is not None and resume_state_path.exists():
        resume_state = torch.load(resume_state_path, map_location="cpu", weights_only=False)
        _check_resume_compatible(resume_state["args"], args)
        print(
            f"[resume] {resume_state_path}: resuming at outer iteration {resume_state['next_it']}"
            f"/{config['iterations']} (expert retraining round {resume_state['expert_retrain_round']})."
        )

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("Building datasets...")
    placeholder_poisoner = pick_poisoner("1xs", dataset_flag, target_label, delta=None)

    big_ims = needs_big_ims(expert_model_flag)
    *_, mtt_dataset_base = get_matching_datasets(
        dataset_flag,
        placeholder_poisoner,
        clean_label,
        train_pct=train_pct,
        big=big_ims,
        clean=clean_trajectory,
    )
    mtt_dataset = TriggerMTTDataset.from_mtt_dataset(mtt_dataset_base)

    n_classes = get_n_classes(dataset_flag)
    labels = extract_labels(mtt_dataset.distill, config["one_hot_temp"], n_classes)
    labels_init = torch.stack(extract_labels(mtt_dataset.distill, 1, n_classes))
    labels_syn = torch.stack(labels).requires_grad_(True)

    raw_train_dataset = get_raw_clean_dataset(dataset_flag, train=True)

    # Load expert trajectories
    print("Loading expert trajectories...")
    if checkpoint_sampling == "uniform":
        expert_starts, expert_opt_starts = extract_experts(
            expert_config,
            input_pths,
            config["iterations"],
            expert_opt_path=opt_pths,
        )
    else:
        expert_starts, expert_opt_starts = extract_experts_biased(
            expert_config,
            input_pths,
            config["iterations"],
            alpha_ckpt,
            expert_opt_path=opt_pths,
        )

    # Optimize labels and trigger jointly
    print("Training...")

    student_models = [load_model(expert_model_flag, n_classes) for _ in range(n_checkpoints_per_step)]
    expert_models = [load_model(expert_model_flag, n_classes) for _ in range(n_checkpoints_per_step)]

    device = get_module_device(student_models[0])

    if lpips_model is not None:
        lpips_model = lpips_model.to(device)

    mu = get_mu(dataset_flag, target_label, device, model_flag=expert_model_flag)
    mu_source = (
        mu
        if clean_label == -1
        else get_mu(
            dataset_flag,
            clean_label,
            device,
            model_flag=expert_model_flag,
        )
    )

    def apply_trigger(x_raw, d, cg, ph=None, ex=None):
        """Preprocessed triggered images for the optimized variables: the additive delta `d`, the
        warp control grid `cg` and the photometric (2, C) tensor `ph` (each None when its component
        is off; see apply_trigger_raw)."""
        return raw_to_trigger_preprocess(
            x_raw, d, dataset_flag=dataset_flag, model_flag=expert_model_flag, control_grid=cg, s=warp_s,
            photo=ph, ext=ex,
        )

    def trigger_artifact():
        """What is saved / handed to the poisoner: the bare tensor (additive) or the dict that
        rebuilds the warp ('warp') / both components ('hybrid')."""
        if not has_warp:
            return delta_add.detach().cpu()
        artifact = {
            "trigger_type": trigger_type,
            "control_grid": warp_grid.detach().cpu(),
            "warp_k": warp_k,
            "warp_s": warp_s,
        }
        if has_additive:
            artifact["delta"] = delta_add.detach().cpu()
        if has_photo:
            artifact["photo"] = photo.detach().cpu()
            artifact["photo_contrast"] = photo_contrast
            artifact["photo_brightness"] = photo_brightness
        if has_ext:
            artifact["ext"] = {k: v.detach().cpu() for k, v in ext.items()}
            artifact["ext_budgets"] = {
                "photo_local_contrast": photo_local_contrast, "photo_local_brightness": photo_local_brightness,
                "color_mix": color_mix_budget, "kernel_budget": kernel_budget,
            }
        return artifact

    warp_grid = init_warp_field(warp_k, warp_s, device=device) if has_warp else None
    # (2, C): row 0 contrast deviation, row 1 brightness shift; starts at 0 = identity
    photo = torch.zeros(2, mu.shape[0], device=device) if has_photo else None
    # extra transformations, name -> leaf tensor, all 0 = identity (see apply_trigger_raw)
    n_ch = mu.shape[0]
    ext = {}
    if has_kernel:
        ext["kernel"] = torch.zeros(n_ch, 3, 3, device=device)
    if has_color_mix:
        ext["color_mix"] = torch.zeros(n_ch, n_ch, device=device)
    if has_photo_local:
        ext["photo_local"] = torch.zeros(2, n_ch, photo_local_grid, photo_local_grid, device=device)
    delta_add = (
        init_delta(
            mu.shape,
            horizontal=True,
            strength=6.0,
            freq=16,
            device=device,
            init=init,
        )
        if has_additive
        else None
    )
    # `delta`: the variable every delta_*/cos_* metric and every pixel-space penalty below refers
    # to -- the additive perturbation whenever there is one, the control grid under the warp-only
    # trigger (where it is the single optimized variable, as before this file knew about 'hybrid').
    delta = delta_add if has_additive else warp_grid
    trigger_vars = [v for v in (delta_add, warp_grid, photo) if v is not None] + list(ext.values())
    warp_var_idx = next(i for i, v in enumerate(trigger_vars) if v is warp_grid) if has_warp else None

    def accumulate_trigger_grad(loss, weight, extra_targets=(), create_graph=False):
        """Adds weight * dloss/dv into every trigger variable's own .grad -- the manual,
        out-of-grand_loss path L_bd/L_joint/L_lpips take -- and returns dloss/d(extra_targets)."""
        extra_targets = list(extra_targets)
        grads = torch.autograd.grad(
            loss,
            trigger_vars + extra_targets,
            retain_graph=True,
            allow_unused=True,
            create_graph=create_graph,
        )
        for v, g in zip(trigger_vars, grads):
            if g is not None:
                g_weighted = weight * g.detach()
                v.grad = g_weighted if v.grad is None else v.grad + g_weighted
        return grads[len(trigger_vars):]

    delta_min = None
    if has_additive:
        max_reachable_l2 = epsilon * math.sqrt(delta.numel())
        delta_min = delta_min_frac * max_reachable_l2
        if delta_min > max_reachable_l2:
            raise ValueError(
                f"delta_min={delta_min:.4f} (delta_min_frac={delta_min_frac} * "
                f"max_reachable=epsilon*sqrt(d)={max_reachable_l2:.4f}) exceeds the feasible "
                f"||delta||_2 ceiling under ||delta||_inf<=epsilon={epsilon} -- structurally "
                "unreachable post-clamp. delta_min_frac must be <= 1.0."
            )
        print(
            f"delta_min={delta_min:.6f} ({delta_min_frac} * max_reachable=epsilon*sqrt(d)="
            f"{max_reachable_l2:.6f}); ||delta_init||_2 (raw, pre-clamp)="
            f"{delta.detach().norm().item():.6f}"
        )
    for v in trigger_vars:
        v.requires_grad_(True)
    if is_hybrid:
        # one Adam over both components, each with its own learning rate (they live on different
        # scales: [0, 1] intensity for delta, pixels for the control grid)
        param_groups = [{"params": [delta_add], "lr": lr_delta}, {"params": [warp_grid], "lr": lr_warp}]
        if has_photo:
            param_groups.append({"params": [photo], "lr": lr_photo})
        if ext:
            param_groups.append({"params": list(ext.values()), "lr": lr_ext})
        optimizer_delta = torch.optim.Adam(param_groups)
    else:
        optimizer_delta = torch.optim.Adam([delta], lr=lr_warp if has_warp else lr_delta)

    batch_size, epochs, optimizer_expert, optimizer_labels = get_mtt_attack_info(
        expert_models[0].parameters(),
        labels_syn,
        config["expert_kwargs"],
        config["labels_kwargs"],
        batch_size=batch_size,
        epochs=epochs,
    )
    batch_size = batch_size // (num_honests + num_poisoned)
    loaders = []
    for _ in range(num_honests + num_poisoned):
        loader, _ = either_dataloader_dataset_to_both(mtt_dataset, batch_size=batch_size, shuffle=True)
        loaders.append(loader)

    N_ASR_EVAL = 256
    asr_eval_raw = []
    for i in range(len(raw_train_dataset)):
        x_i, y_i = raw_train_dataset[i]
        if clean_label == -1 or y_i == clean_label:
            asr_eval_raw.append(x_i)
            if len(asr_eval_raw) >= N_ASR_EVAL:
                break
    asr_eval_raw = torch.stack(asr_eval_raw).to(device)

    # Raw (un-normalized) poisoned images, decoded once and kept on device instead of being
    # re-read from raw_train_dataset at every step.
    raw_poison_cache = {}

    def _raw_poison_batch(indices):
        for i in indices:
            if i not in raw_poison_cache:
                raw_poison_cache[i] = raw_train_dataset[i][0].to(device)
        return torch.stack([raw_poison_cache[i] for i in indices])

    # Snapshots of the trigger as initialized, captured on the first step: delta_init feeds
    # cos_delta_to_init/delta_drift_l2 (the primary variable), the two component snapshots feed
    # the frozen-trigger ASR.
    delta_init = None
    delta_add_init = None
    warp_grid_init = None
    photo_init = None
    ext_init = None
    prev_delta_sign = None

    expert_pool = None
    pool_paths = None
    if pool_size != 1 and resume_state is None:
        print(f"Preloading up to {pool_size} expert checkpoints into RAM (float32, CPU)...")
        expert_pool, pool_size, pool_paths = build_expert_pool(expert_starts, expert_opt_starts, pool_size)
    elif resume_state is not None:
        # the pool is restored (or rebuilt by a retraining round) below, with the rest of the state
        pool_size = resume_state["pool_size"]
        n_checkpoints_per_step = resume_state["n_checkpoints_per_step"]

    if n_checkpoints_per_step > pool_size:
        raise ValueError(
            f"n_checkpoints_per_step={n_checkpoints_per_step} > pool_size={pool_size} "
            "(possibly already clamped down from a larger request, see build_expert_pool) -- "
            "cannot draw that many DISTINCT checkpoints per step."
        )

    def _opt_state_to_device(opt_state_cpu, device):
        return {
            "state": {
                k: {kk: (vv.to(device) if torch.is_tensor(vv) else vv) for kk, vv in v.items()}
                for k, v in opt_state_cpu["state"].items()
            },
            "param_groups": opt_state_cpu["param_groups"],
        }

    def _load_expert_for_step(it):
        """Returns (params_state_dict, opt_state_dict) for outer step `it` -- params are
        load_state_dict-ed into expert_model/student_model by the caller, opt_state_dict is
        read directly (sgd_step, below) into device-resident tensor arithmetic."""
        if pool_size == 1:
            checkpoint = torch.load(expert_starts[it])
            opt_state = torch.load(expert_opt_starts[it])
            return checkpoint, opt_state
        checkpoint, opt_state_cpu = random.choice(expert_pool)
        return checkpoint, _opt_state_to_device(opt_state_cpu, device)

    def _load_experts_for_step(it):
        """Returns a list of n_checkpoints_per_step (params_state_dict, opt_state_dict) pairs
        for outer step `it` (see run() docstring "Averaging the loss over multiple
        checkpoints"). n_checkpoints_per_step==1 (default) just wraps _load_expert_for_step's
        single draw in a list -- bit-for-bit the prior single-checkpoint behavior. > 1 draws
        that many DISTINCT entries from expert_pool (validated >= n_checkpoints_per_step
        above) -- distinct, not with replacement, so the K checkpoints averaged this step are
        genuinely different points on the expert trajectory."""
        if n_checkpoints_per_step == 1:
            return [_load_expert_for_step(it)]
        drawn = random.sample(expert_pool, n_checkpoints_per_step)
        return [(checkpoint, _opt_state_to_device(opt_state_cpu, device)) for checkpoint, opt_state_cpu in drawn]

    # Read back only by this run, so machine-local under the NFS executor (no-op otherwise).
    # Every round OVERWRITES the previous one's trajectory (expert index 0, like train_expert's
    # own layout): only the latest round is ever read back -- its pool is loaded into RAM right
    # after it is written -- and a 200-epoch trajectory is ~3 GB. The marker names the round the
    # directory holds once complete; it is removed before a round starts overwriting it.
    retrain_base_dir = Path(local_storage_path(Path(output_dir).parent / "expert_retrain"))
    retrain_dir = retrain_base_dir / "0"
    retrain_marker = retrain_base_dir / "complete_round.txt"
    retrain_input_pths = str(retrain_base_dir / "{}" / "model_{}_{}.pth")
    retrain_opt_pths = str(retrain_base_dir / "{}" / "model_{}_{}_opt.pth")

    def _retrained_round_on_disk():
        """Round whose complete trajectory retrain_dir holds, or None."""
        try:
            return int(retrain_marker.read_text().strip())
        except (OSError, ValueError):
            return None

    def _retrain_expert_with_trigger(round_idx, it, remaining_iterations):
        nonlocal expert_starts, expert_opt_starts, expert_pool, pool_size, n_checkpoints_per_step, pool_paths

        norms = []
        if has_additive:
            norms.append(f"||delta||_inf={delta_add.abs().max().item():.4f}")
        if has_warp:
            norms.append(f"||control_grid||_inf={warp_grid.abs().max().item():.4f}")
        if has_photo:
            norms.append(f"|contrast|_max={photo[0].abs().max().item():.4f}, "
                         f"|brightness|_max={photo[1].abs().max().item():.4f}")
        for name_, v_ in ext.items():
            norms.append(f"|{name_}|_max={v_.abs().max().item():.4f}")
        print(
            f"[expert_retrain] round {round_idx}: retraining a fresh expert for "
            f"{expert_retrain_epochs} epochs against the CURRENT trigger ({', '.join(norms)})..."
        )

        if expert_retrain_agg_method != agg_method:
            print(
                f"WARNING: expert_retrain_agg_method={expert_retrain_agg_method!r} differs "
                f"from agg_method={agg_method!r}, but expert retraining (mini_train below) is "
                "a plain centralized run with no federated aggregator of its own -- this value "
                "is NOT used, retraining is unaffected by it. See the comment just above."
            )
        if seed is not None:
            print(
                f"[expert_retrain] round {round_idx}: reseeding RNGs to seed={seed} "
                "(matching the original train_expert step) before this round's dataset/model."
            )
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        poisoner = pick_poisoner(
            POISONER_BY_TRIGGER_TYPE[trigger_type],
            dataset_flag,
            target_label,
            delta=trigger_artifact(),
        )
        poison_train, _, test, poison_test, _ = get_matching_datasets(
            dataset_flag,
            poisoner,
            clean_label,
            train_pct=train_pct,
            big=big_ims,
        )

        retrain_model = load_model(expert_model_flag, n_classes)
        retrain_batch_size, retrain_epochs, retrain_opt, retrain_sched = get_train_info(
            retrain_model.parameters(),
            "sgd",
            batch_size=None,
            epochs=expert_retrain_epochs,
            optim_kwargs=expert_retrain_optim_kwargs,
            scheduler_kwargs=expert_retrain_scheduler_kwargs,
        )
        retrain_marker.unlink(missing_ok=True)
        shutil.rmtree(retrain_dir, ignore_errors=True)
        # No per-epoch test_data: mini_train would only print it (record=False, result unused).
        mini_train(
            model=retrain_model,
            train_data=poison_train,
            batch_size=retrain_batch_size,
            opt=retrain_opt,
            scheduler=retrain_sched,
            epochs=retrain_epochs,
            callback=lambda m, o, e, i: checkpoint_callback(
                m,
                o,
                e,
                i,
                expert_retrain_checkpoint_iters,
                str(retrain_dir),
            ),
        )
        retrain_marker.write_text(f"{round_idx}\n")

        retrain_cta = clf_eval(retrain_model, test)[0]
        retrain_asr = clf_eval(retrain_model, poison_test.poison_dataset)[0]
        print(f"[expert_retrain] round {round_idx}: retrained expert CTA={retrain_cta:.4f} ASR={retrain_asr:.4f}")
        tracker.log(global_step, retrain_expert_cta=retrain_cta, retrain_expert_asr=retrain_asr)

        round_input_pths, round_opt_pths = retrain_input_pths, retrain_opt_pths
        extract_iterations = max(remaining_iterations, pool_size) if pool_size != 1 else remaining_iterations
        if checkpoint_sampling == "uniform":
            new_starts, new_opt_starts = extract_experts(
                expert_config,
                round_input_pths,
                extract_iterations,
                expert_opt_path=round_opt_pths,
            )
        else:
            new_starts, new_opt_starts = extract_experts_biased(
                expert_config,
                round_input_pths,
                extract_iterations,
                alpha_ckpt,
                expert_opt_path=round_opt_pths,
            )
        expert_starts = expert_starts[:it] + new_starts
        expert_opt_starts = expert_opt_starts[:it] + new_opt_starts

        if pool_size != 1:
            print(
                f"[expert_retrain] round {round_idx}: refreshing the {pool_size}-checkpoint "
                "pool from the retrained trajectory..."
            )
            expert_pool, pool_size, pool_paths = build_expert_pool(
                new_starts,
                new_opt_starts,
                pool_size,
            )
            if n_checkpoints_per_step > pool_size:
                print(
                    f"WARNING: n_checkpoints_per_step={n_checkpoints_per_step} > refreshed "
                    f"pool_size={pool_size} (round {round_idx} produced fewer distinct "
                    "checkpoints than requested, likely because remaining_iterations= "
                    f"{remaining_iterations} left too few (iteration, trajectory-length) draws "
                    f"to pool from) -- clamping n_checkpoints_per_step to {pool_size} for the "
                    "rest of this run."
                )
                n_checkpoints_per_step = pool_size
        print(f"[expert_retrain] round {round_idx}: done, trajectory refreshed from it={it}.")

    losses = []
    align_active_window = []
    mag_active_window = []
    ANTI_COLLAPSE_WINDOW = 50
    global_step = 0
    start_it = 0
    expert_retrain_round = 0

    def _cpu(t):
        return None if t is None else t.detach().cpu().clone()

    def _save_resume_state(next_it):
        """Snapshot after outer iteration next_it-1 (atomic overwrite), plus that iteration's trigger."""
        if resume_state_path is None:
            return
        state = {
            "version": 1,
            "args": json.loads(json.dumps(args)),  # plain dicts: toml's inline tables don't pickle
            "next_it": next_it,
            "expert_retrain_round": expert_retrain_round,
            "global_step": global_step,
            "labels_syn": _cpu(labels_syn),
            "optimizer_labels": optimizer_labels.state_dict(),
            "trigger": {"delta_add": _cpu(delta_add), "warp_grid": _cpu(warp_grid), "photo": _cpu(photo)},
            "ext": {k: _cpu(v) for k, v in ext.items()},
            "ext_init": None if ext_init is None else {k: _cpu(v) for k, v in ext_init.items()},
            "optimizer_delta": optimizer_delta.state_dict(),
            "trigger_init": {
                "delta_add": _cpu(delta_add_init), "warp_grid": _cpu(warp_grid_init), "photo": _cpu(photo_init),
            },
            "prev_delta_sign": _cpu(prev_delta_sign),
            "align_active_window": align_active_window,
            "mag_active_window": mag_active_window,
            "losses": losses,
            "history": history,
            "tracker_history": tracker._history,
            "lambdas": {
                "lambda_bd": lambda_bd, "lambda_gradmatch": lambda_gradmatch,
                "lambda_lpips": lambda_lpips, "lambda_consistency": lambda_consistency,
            },
            "pool_size": pool_size,
            "n_checkpoints_per_step": n_checkpoints_per_step,
            "pool_paths": pool_paths,
            "expert_starts": expert_starts,
            "expert_opt_starts": expert_opt_starts,
            "rng": _rng_state(),
        }
        resume_state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = resume_state_path.with_name(resume_state_path.name + ".tmp")
        torch.save(state, tmp)
        os.replace(tmp, resume_state_path)
        torch.save(trigger_artifact(), resume_state_path.parent / f"trigger_it{next_it:03d}.pt")
        print(f"[resume] state saved after outer iteration {next_it}/{config['iterations']} -> {resume_state_path}")

    if resume_state is not None:
        st = resume_state
        with torch.no_grad():
            labels_syn.copy_(st["labels_syn"])
            for name, v in (("delta_add", delta_add), ("warp_grid", warp_grid), ("photo", photo)):
                if v is not None:
                    v.copy_(st["trigger"][name].to(v.device))
        # load_state_dict also restores the saved hyperparameters (lr, ...): keep the config's.
        for opt, saved in ((optimizer_labels, st["optimizer_labels"]), (optimizer_delta, st["optimizer_delta"])):
            hyper = [{k: v for k, v in g.items() if k != "params"} for g in opt.param_groups]
            opt.load_state_dict(saved)
            for g, h in zip(opt.param_groups, hyper):
                g.update(h)

        def _dev(t):
            return None if t is None else t.to(device)

        delta_add_init = _dev(st["trigger_init"]["delta_add"])
        warp_grid_init = _dev(st["trigger_init"]["warp_grid"])
        photo_init = _dev(st["trigger_init"]["photo"])
        for k_, v_ in ext.items():
            with torch.no_grad():
                v_.copy_(st["ext"][k_].to(v_.device))
        ext_init = None if st.get("ext_init") is None else {k_: _dev(v_) for k_, v_ in st["ext_init"].items()}
        delta_init = delta_add_init if has_additive else warp_grid_init
        prev_delta_sign = _dev(st["prev_delta_sign"])
        align_active_window = st["align_active_window"]
        mag_active_window = st["mag_active_window"]
        losses = st["losses"]
        if history is not None and st["history"] is not None:
            history = st["history"]
        tracker._history = st["tracker_history"]
        lambda_bd = st["lambdas"]["lambda_bd"]
        lambda_gradmatch = st["lambdas"]["lambda_gradmatch"]
        lambda_lpips = st["lambdas"]["lambda_lpips"]
        lambda_consistency = st["lambdas"]["lambda_consistency"]
        global_step = st["global_step"]
        start_it = st["next_it"]
        expert_retrain_round = st["expert_retrain_round"]
        expert_starts, expert_opt_starts = st["expert_starts"], st["expert_opt_starts"]
        pool_paths = st["pool_paths"]

        # The next iteration needs the pool unless it starts with a retraining round of its own.
        retrains_first = bool(expert_retrain_interval) and start_it > 0 and start_it % expert_retrain_interval == 0
        if pool_size != 1 and start_it < config["iterations"] and not retrains_first:
            on_disk = expert_retrain_round == 0 or _retrained_round_on_disk() == expert_retrain_round
            if on_disk and all(os.path.exists(p) and os.path.exists(o) for p, o in pool_paths):
                print(f"[resume] reloading the {len(pool_paths)}-checkpoint pool...")
                expert_pool = load_expert_pool(pool_paths)
            elif expert_retrain_round > 0:
                print(
                    f"[resume] round {expert_retrain_round}'s retrained trajectory is not on this "
                    "machine -- retraining it against the restored trigger."
                )
                _retrain_expert_with_trigger(expert_retrain_round, start_it, config["iterations"] - start_it)
            else:
                missing = [p for p, o in pool_paths if not (os.path.exists(p) and os.path.exists(o))]
                raise FileNotFoundError(f"[resume] initial expert checkpoints missing, e.g. {missing[:3]}")
        _set_rng_state(st["rng"])

    with make_pbar(
        total=config["iterations"] * len(mtt_dataset),
        initial=min(start_it, config["iterations"]) * len(mtt_dataset),
    ) as pbar:
        for it in range(start_it, config["iterations"]):
            if expert_retrain_interval and it > 0 and it % expert_retrain_interval == 0:
                expert_retrain_round += 1
                _retrain_expert_with_trigger(
                    expert_retrain_round,
                    it,
                    config["iterations"] - it,
                )

            for batches in zip(*loaders):
                checkpoints_k = _load_experts_for_step(it)
                optimizer_delta.zero_grad()

                do_balance = lambda_balance_interval > 0 and global_step % lambda_balance_interval == 0
                L_bd_live_total = torch.tensor(0.0, device=device)
                L_lpips_live_total = torch.tensor(0.0, device=device)

                mtt_term_list = []
                gradmatch_list = []
                match_list = []
                matching_term_list = []
                L_bd_mean_list = []
                L_joint_mean_list = []
                L_lpips_mean_list = []

                margin_list = []
                margin_raw_mean_list = []
                consistency_list = []

                for k, (checkpoint, state_dict) in enumerate(checkpoints_k):
                    expert_model = expert_models[k]
                    student_model = student_models[k]
                    expert_model.load_state_dict(checkpoint)
                    student_model.load_state_dict(checkpoint)

                    expert_start = [p.clone() for p in expert_model.parameters()]

                    expert_params = list(expert_model.parameters())
                    student_params = list(student_model.parameters())

                    student_grad_buf = [[] for _ in student_params]
                    expert_grad_buf = [[] for _ in expert_params]

                    clean_grad_chunks = [[] for _ in expert_params]
                    poison_grad_chunks = [[] for _ in expert_params]

                    L_bd_sum = torch.tensor(0.0, device=device)
                    n_bd_valid = 0

                    L_joint_sum = torch.tensor(0.0, device=device)
                    n_joint_valid = 0
                    L_lpips_sum = torch.tensor(0.0, device=device)
                    n_lpips_valid = 0

                    margin_sum = torch.tensor(0.0, device=device)
                    margin_raw_sum = torch.tensor(0.0, device=device)

                    for cid, batch in enumerate(batches):
                        if cid < num_honests:
                            x, y, _, _, _, is_p = batch
                            keep = ~is_p
                            if not keep.any():
                                continue
                            x, y = x[keep].to(device), y[keep].to(device)

                            expert_model.zero_grad()
                            loss = clf_loss(expert_model(x), y)
                            loss.backward()

                            for i, p in enumerate(expert_params):
                                if p.grad is not None:
                                    g = p.grad.detach().clone()
                                    expert_grad_buf[i].append(g)
                                    student_grad_buf[i].append(g)
                                    if track_gradmatch:
                                        clean_grad_chunks[i].append(g)

                        # POISONED CLIENTS
                        else:
                            x_t, y_t, x_d, _, idx, is_poisoned = batch

                            x_t, y_t = x_t.to(device), y_t.to(device)
                            x_d = x_d.to(device)
                            is_poisoned_dev = is_poisoned.to(device)
                            y_d = labels_syn[idx].to(device)

                            x_t_adv = x_t
                            if is_poisoned.any():
                                idx_poisoned = idx[is_poisoned].tolist()
                                x_raw_adv = _raw_poison_batch(idx_poisoned)
                                x_trig = apply_trigger(x_raw_adv, delta_add, warp_grid, photo, ext)
                                x_t_adv = x_t.clone()

                                x_t_adv[is_poisoned_dev] = x_trig

                            expert_model.zero_grad()
                            loss_e = clf_loss(expert_model(x_t_adv), y_t)
                            grads_e = torch.autograd.grad(
                                loss_e,
                                expert_params,
                                create_graph=True,
                                retain_graph=True,
                            )

                            for i, g in enumerate(grads_e):
                                expert_grad_buf[i].append(g)

                            if track_joint:
                                accumulate_trigger_grad(loss_e, lambda_joint / n_checkpoints_per_step)
                                L_joint_sum = L_joint_sum + loss_e.detach()
                                n_joint_valid += 1

                            if track_gradmatch:
                                clean_mask = ~is_poisoned_dev
                                if clean_mask.any():
                                    loss_clean_local = clf_loss(expert_model(x_t[clean_mask]), y_t[clean_mask])
                                    clean_grads_local = torch.autograd.grad(
                                        loss_clean_local,
                                        expert_params,
                                        allow_unused=True,
                                    )
                                    for i, g in enumerate(clean_grads_local):
                                        if g is not None:
                                            clean_grad_chunks[i].append(g.detach())

                            if is_poisoned.any():
                                logits_bd = expert_model(x_trig)
                                L_bd_cid = clf_loss(logits_bd, y_t[is_poisoned_dev])

                                if track_margin:
                                    L_margin_cid, margin_raw_cid = margin_floor_penalty(
                                        logits_bd,
                                        target_label,
                                        margin_min,
                                    )
                                    margin_sum = margin_sum + L_margin_cid
                                    margin_raw_sum = margin_raw_sum + margin_raw_cid.detach()

                                expert_bd_grads = accumulate_trigger_grad(
                                    L_bd_cid,
                                    lambda_bd / n_checkpoints_per_step,
                                    extra_targets=expert_params if track_poison_grad else (),
                                    create_graph=track_poison_grad,
                                )
                                if track_poison_grad:
                                    for i, g in enumerate(expert_bd_grads):
                                        if g is not None:
                                            poison_grad_chunks[i].append(g)
                                L_bd_sum = L_bd_sum + L_bd_cid.detach()
                                n_bd_valid += 1
                                if do_balance:
                                    L_bd_live_total = L_bd_live_total + lambda_bd * L_bd_cid

                                if lpips_model is not None:
                                    L_lpips_cid = lpips_penalty(
                                        x_raw_adv, lpips_model, delta=delta_add,
                                        control_grid=warp_grid, warp_s=warp_s, photo=photo, ext=ext,
                                    )
                                    accumulate_trigger_grad(
                                        L_lpips_cid, lambda_lpips / n_checkpoints_per_step,
                                    )
                                    L_lpips_sum = L_lpips_sum + L_lpips_cid.detach()
                                    n_lpips_valid += 1
                                    if do_balance:
                                        L_lpips_live_total = L_lpips_live_total + lambda_lpips * L_lpips_cid

                            # Student
                            loss_s = clf_loss(student_model(x_d), softmax(y_d))
                            grads_s = torch.autograd.grad(loss_s, student_params, create_graph=True)

                            for i, g in enumerate(grads_s):
                                student_grad_buf[i].append(g)

                    have_poison_grad = all(len(c) > 0 for c in poison_grad_chunks)
                    poison_grad_mean = (
                        [torch.stack(c, dim=0).mean(dim=0) for c in poison_grad_chunks] if have_poison_grad else None
                    )

                    consistency_k = (
                        poison_consistency(poison_grad_chunks)
                        if track_consistency and have_poison_grad
                        else torch.tensor(0.0, device=device)
                    )
                    consistency_list.append(consistency_k)

                    if track_gradmatch and poison_grad_mean is not None and all(len(c) > 0 for c in clean_grad_chunks):
                        clean_grad_mean = [torch.stack(c, dim=0).mean(dim=0) for c in clean_grad_chunks]
                        penalty_fn = grad_cosine_penalty if gradmatch_metric == "cosine" else grad_mismatch_penalty
                        L_gradmatch_k = penalty_fn(
                            clean_grad_mean,
                            poison_grad_mean,
                            eps=gradmatch_eps,
                        )
                    else:
                        L_gradmatch_k = torch.tensor(0.0, device=device)

                    agg_student_grads = agg(student_params, student_grad_buf, agg_method, f=num_poisoned, pre_agg=pre_agg)

                    agg_expert_grads = agg(expert_params, expert_grad_buf, agg_method, f=num_poisoned, pre_agg=pre_agg)

                    if track_match and poison_grad_mean is not None:
                        L_match_k = grad_cosine_penalty(agg_expert_grads, poison_grad_mean)
                    else:
                        L_match_k = torch.tensor(0.0, device=device)

                    param_loss = torch.tensor(0.0, device=device)
                    param_dist = torch.tensor(0.0, device=device)

                    if attack in ["backdoor", "untargeted"]:
                        for init_p, student, grad, grad_e, state in zip(
                            expert_start,
                            student_params,
                            agg_student_grads,
                            agg_expert_grads,
                            state_dict["state"].values(),
                        ):
                            student_update = sgd_step(student, grad, state, state_dict["param_groups"][0])

                            state_expert_copy = {
                                kk: (v.clone() if torch.is_tensor(v) else v) for kk, v in state.items()
                            }
                            expert_next_param = sgd_step(
                                init_p, grad_e, state_expert_copy, state_dict["param_groups"][0]
                            )

                            param_loss += total_mse_distance(student_update, expert_next_param)
                            param_dist += total_mse_distance(init_p, expert_next_param)

                        mtt_denom = param_dist.detach() if detach_param_dist else param_dist
                        mtt_term_k = param_loss / mtt_denom

                    mtt_term_list.append(mtt_term_k)
                    gradmatch_list.append(L_gradmatch_k)
                    match_list.append(L_match_k)
                    matching_term_list.append((param_loss / param_dist).item() if param_dist.item() != 0 else 0.0)
                    L_bd_mean_list.append((L_bd_sum / n_bd_valid).item() if n_bd_valid > 0 else 0.0)
                    L_joint_mean_list.append((L_joint_sum / n_joint_valid).item() if n_joint_valid > 0 else 0.0)
                    L_lpips_mean_list.append((L_lpips_sum / n_lpips_valid).item() if n_lpips_valid > 0 else 0.0)
                    margin_list.append(margin_sum / n_bd_valid if n_bd_valid > 0 else torch.tensor(0.0, device=device))
                    margin_raw_mean_list.append((margin_raw_sum / n_bd_valid).item() if n_bd_valid > 0 else 0.0)

                manual_only_grads = [
                    v.grad.detach().clone() if v.grad is not None else torch.zeros_like(v) for v in trigger_vars
                ]

                reg_term = lam * torch.linalg.vector_norm(softmax(labels_syn) - labels_init, ord=1, dim=1).mean()

                mtt_term_avg = torch.stack(mtt_term_list).mean()
                L_gradmatch_avg = torch.stack(gradmatch_list).mean()
                L_match_avg = torch.stack(match_list).mean()
                margin_avg = torch.stack(margin_list).mean()
                consistency_avg = torch.stack(consistency_list).mean()

                grand_loss = gamma_stealth * (mtt_term_avg + reg_term)

                if has_additive:  # pixel-delta penalties: undefined for the warp control grid
                    L_pen = trigger_penalty(delta, mu)
                    L_tv = tv_loss(delta)
                    grand_loss = grand_loss + (lambda_penalty * L_pen + lambda_delta * delta.norm() + lambda_tv * L_tv)

                    L_align, _ = directional_floor_penalty(delta, mu, align_kappa)
                    L_mag, _ = magnitude_floor_penalty(delta, delta_min)
                    if trigger_constraint == "penalty":
                        grand_loss = grand_loss + (lambda_align * L_align + lambda_mag * L_mag)

                grand_loss = grand_loss + lambda_gradmatch * L_gradmatch_avg
                grand_loss = grand_loss + lambda_match * L_match_avg

                grand_loss = grand_loss + lambda_margin * margin_avg

                grand_loss = grand_loss + lambda_consistency * consistency_avg

                if do_balance:

                    def _grad_norm(term):
                        """||dterm/d(trigger)||_2 over every trigger variable at once (the two
                        components' gradients concatenated under trigger_type='hybrid')."""
                        grads = [
                            g
                            for g in torch.autograd.grad(
                                term,
                                trigger_vars,
                                retain_graph=True,
                                allow_unused=True,
                            )
                            if g is not None
                        ]
                        if not grads:
                            return 0.0
                        return torch.cat([g.reshape(-1) for g in grads]).norm().item()

                    g_main = _grad_norm(gamma_stealth * mtt_term_avg)
                    balance_targets = {
                        "L_bd": (L_bd_live_total, "lambda_bd"),
                        "L_gradmatch": (lambda_gradmatch * L_gradmatch_avg, "lambda_gradmatch"),
                        "L_lpips": (L_lpips_live_total, "lambda_lpips"),
                    }

                    if lambda_consistency != 0:
                        balance_targets["L_consistency"] = (lambda_consistency * consistency_avg, "lambda_consistency")
                    for name, (term, lambda_attr) in balance_targets.items():
                        ratio_i = lambda_balance_ratios.get(name)
                        if ratio_i is None:
                            continue
                        g_i = _grad_norm(term)
                        if g_i <= 0:
                            continue
                        current = locals()[lambda_attr]
                        if current == 0:
                            continue
                        scale = math.sqrt(max(ratio_i * g_main / g_i, 0.0))
                        new_val = min(max(current * scale, lambda_balance_min), lambda_balance_max)
                        if lambda_attr == "lambda_bd":
                            lambda_bd = new_val
                        elif lambda_attr == "lambda_gradmatch":
                            lambda_gradmatch = new_val
                        elif lambda_attr == "lambda_lpips":
                            lambda_lpips = new_val
                        elif lambda_attr == "lambda_consistency":
                            lambda_consistency = new_val

                optimizer_labels.zero_grad()
                grand_loss.backward()
                global_step += 1

                # What grand_loss.backward() (the MTT matching term and the terms folded into it)
                # added on top of the manually accumulated L_bd/L_joint/L_lpips gradients, per
                # trigger variable: [0] is the primary one (delta), [-1] the warp control grid.
                mtt_trigger_grad_norms = [
                    (v.grad.detach() - g_manual).norm().item() if v.grad is not None else 0.0
                    for v, g_manual in zip(trigger_vars, manual_only_grads)
                ]
                mtt_delta_grad_norm = mtt_trigger_grad_norms[0]
                mtt_warp_grad_norm = mtt_trigger_grad_norms[warp_var_idx] if has_warp else 0.0

                optimizer_labels.step()
                optimizer_delta.step()

                with torch.no_grad():
                    if has_additive:
                        if trigger_constraint == "projection":
                            delta.copy_(project_trigger_constraints(delta, mu, epsilon, align_kappa, delta_min))
                        else:
                            delta.clamp_(-epsilon, epsilon)
                    if has_warp:
                        warp_grid.clamp_(-warp_s, warp_s)
                    if has_photo:
                        photo[0].clamp_(-photo_contrast, photo_contrast)
                        photo[1].clamp_(-photo_brightness, photo_brightness)
                    if "kernel" in ext:
                        ext["kernel"].clamp_(-kernel_budget, kernel_budget)
                    if "color_mix" in ext:
                        ext["color_mix"].clamp_(-color_mix_budget, color_mix_budget)
                    if "photo_local" in ext:
                        ext["photo_local"][0].clamp_(-photo_local_contrast, photo_local_contrast)
                        ext["photo_local"][1].clamp_(-photo_local_brightness, photo_local_brightness)

                for m in expert_models:
                    m.eval()

                L_bd_mean = float(np.mean(L_bd_mean_list))
                L_joint_mean = float(np.mean(L_joint_mean_list))
                L_lpips_mean = float(np.mean(L_lpips_mean_list))
                matching_term = float(np.mean(matching_term_list))

                with torch.no_grad():
                    delta_l2 = delta.norm().item()
                    delta_linf = delta.abs().max().item()
                    warp_l2 = warp_grid.norm().item() if has_warp else 0.0
                    warp_linf = warp_grid.abs().max().item() if has_warp else 0.0
                    if has_additive:
                        L_align_post, cos_target_post = directional_floor_penalty(delta, mu, align_kappa)
                        cos_target_post, L_align_post = cos_target_post.item(), L_align_post.item()
                        cos_source_post = cosine_to(delta, mu_source).item()
                        L_mag_post, _ = magnitude_floor_penalty(delta, delta_min)
                        L_mag_post = L_mag_post.item()
                    else:
                        L_align_post = cos_target_post = cos_source_post = L_mag_post = 0.0

                    delta_sign = delta.sign()
                    if prev_delta_sign is None:
                        delta_sign_flip_rate = 0.0
                    else:
                        delta_sign_flip_rate = (delta_sign != prev_delta_sign).float().mean().item()
                    prev_delta_sign = delta_sign.clone()

                    just_captured = delta_init is None
                    if just_captured:
                        delta_add_init = delta_add.detach().clone() if has_additive else None
                        warp_grid_init = warp_grid.detach().clone() if has_warp else None
                        photo_init = photo.detach().clone() if has_photo else None
                        ext_init = {k_: v_.detach().clone() for k_, v_ in ext.items()} if ext else None
                        delta_init = delta_add_init if has_additive else warp_grid_init
                    cos_to_init = cosine_to(delta, delta_init).item()
                    delta_drift_l2 = (delta - delta_init).norm().item()
                    if just_captured and delta_init.norm() > 0:
                        assert abs(cos_to_init - 1.0) < 1e-5 and delta_drift_l2 < 1e-6, (
                            f"delta_init was just captured but cos_to_init={cos_to_init}, "
                            f"delta_drift_l2={delta_drift_l2} -- expected exactly 1.0/0.0."
                        )

                    x_trig_eval = apply_trigger(asr_eval_raw, delta_add, warp_grid, photo, ext)
                    x_trig_eval_frozen = apply_trigger(asr_eval_raw, delta_add_init, warp_grid_init, photo_init, ext_init)
                    asr_mean_list = [
                        (m(x_trig_eval).argmax(dim=1) == target_label).float().mean().item() for m in expert_models
                    ]
                    asr_frozen_mean_list = [
                        (m(x_trig_eval_frozen).argmax(dim=1) == target_label).float().mean().item()
                        for m in expert_models
                    ]
                    asr_mean = float(np.mean(asr_mean_list))
                    asr_frozen_mean = float(np.mean(asr_frozen_mean_list))

                align_active_window.append(L_align_post > 0)
                mag_active_window.append(L_mag_post > 0)
                if len(align_active_window) > ANTI_COLLAPSE_WINDOW:
                    align_active_window.pop(0)
                if len(mag_active_window) > ANTI_COLLAPSE_WINDOW:
                    mag_active_window.pop(0)
                align_active_rate = sum(align_active_window) / len(align_active_window)
                mag_active_rate = sum(mag_active_window) / len(mag_active_window)

                losses.append(grand_loss.item())
                step_metrics = {
                    "grand_loss": grand_loss.item(),
                    "matching_term": matching_term,
                    "L_bd_mean": L_bd_mean,
                    "L_joint_mean": L_joint_mean,
                    "L_lpips_mean": L_lpips_mean,
                    "expert_asr": asr_mean,
                    "expert_asr_frozen": asr_frozen_mean,
                    "cos_delta_to_init": cos_to_init,
                    "delta_drift_l2": delta_drift_l2,
                    "delta_l2": delta_l2,
                    "delta_linf": delta_linf,
                    "warp_l2": warp_l2,
                    "warp_linf": warp_linf,
                    "cos_target": cos_target_post,
                    "cos_source": cos_source_post,
                    "L_align": L_align_post,
                    "L_mag": L_mag_post,
                    "align_active_rate": align_active_rate,
                    "mag_active_rate": mag_active_rate,
                    "reg_term": reg_term.item(),
                    "mtt_delta_grad_norm": mtt_delta_grad_norm,
                    "mtt_warp_grad_norm": mtt_warp_grad_norm,
                    "L_gradmatch": L_gradmatch_avg.item(),
                    "L_match": L_match_avg.item(),
                    "matching_term_std": float(np.std(matching_term_list)),
                    "L_bd_mean_std": float(np.std(L_bd_mean_list)),
                    "L_joint_mean_std": float(np.std(L_joint_mean_list)),
                    "expert_asr_std": float(np.std(asr_mean_list)),
                    "L_margin": margin_avg.item(),
                    "margin_mean": float(np.mean(margin_raw_mean_list)),
                    "L_consistency": consistency_avg.item(),
                    "poison_consistency": consistency_avg.item(),
                    "delta_sign_flip_rate": delta_sign_flip_rate,
                    "lambda_bd_current": lambda_bd,
                    "lambda_gradmatch_current": lambda_gradmatch,
                    "lambda_lpips_current": lambda_lpips,
                    "lambda_consistency_current": lambda_consistency,
                }
                if has_photo:
                    step_metrics["photo_contrast_abs"] = photo[0].abs().max().item()
                    step_metrics["photo_brightness_abs"] = photo[1].abs().max().item()
                for name_, v_ in ext.items():
                    step_metrics[f"{name_}_abs"] = v_.abs().max().item()
                if history is not None:
                    history.append({"it": it, **step_metrics})
                tracker.log(global_step, **step_metrics)

                pbar.update(batch_size)
                postfix = {
                    "g_loss": f"{np.mean(losses[-20:]):.4g}",
                    "match": f"{matching_term:.4g}" if attack in ["backdoor"] else "N/A",
                    "L_bd": f"{L_bd_mean:.4g}",
                    "L_joint": f"{L_joint_mean:.4g}",
                    "expert_asr": f"{asr_mean:.4g}",
                    "asr_frozen": f"{asr_frozen_mean:.4g}",
                    "cos_init": f"{cos_to_init:.4g}",
                    "delta_linf": f"{delta_linf:.4g}",
                    "delta_l2": f"{delta_l2:.4g}",
                    "cos_tgt": f"{cos_target_post:.4g}",
                    "L_align": f"{L_align_post:.4g}",
                    "L_mag": f"{L_mag_post:.4g}",
                    "align_rate": f"{align_active_rate:.2f}",
                    "mag_rate": f"{mag_active_rate:.2f}",
                    "mtt_grad": f"{mtt_delta_grad_norm:.4g}",
                    "gradmatch": f"{L_gradmatch_avg.item():.4g}",
                    "agg_match": f"{L_match_avg.item():.4g}",
                }
                if is_hybrid:  # the warp half, alongside delta_linf/mtt_grad for the additive one
                    postfix["warp_linf"] = f"{warp_linf:.4g}"
                    postfix["mtt_warp_grad"] = f"{mtt_warp_grad_norm:.4g}"
                if has_photo:
                    postfix["photo_c"] = f"{photo[0].abs().max().item():.3g}"
                    postfix["photo_b"] = f"{photo[1].abs().max().item():.3g}"
                for name_, v_ in ext.items():
                    postfix[name_] = f"{v_.abs().max().item():.3g}"
                pbar.set_postfix(**postfix)

            _save_resume_state(it + 1)

    # Save results
    print("Saving results...")
    y_true = torch.stack([mtt_dataset[i][3].detach() for i in range(len(mtt_dataset.distill))])
    np.save(output_dir + "labels.npy", labels_syn.detach().numpy())
    np.save(output_dir + "true.npy", y_true)
    np.save(output_dir + "losses.npy", losses)

    tracker.finalize()

    if metrics_log_path:
        Path(metrics_log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_log_path, "w") as f:
            json.dump(history, f, indent=2)

    run_tag = f"{num_poisoned}vs{num_honests}"
    Path(output_dir_trigger).mkdir(parents=True, exist_ok=True)
    trig_path = (
        Path(output_dir_trigger) / f"opt_trig_direct_joint_{init}_{expert_model_flag}_{dataset_flag}_{run_tag}.pt"
    )
    torch.save(trigger_artifact(), trig_path)
    print(
        f"Saved trigger to {trig_path} (trigger_type={trigger_type}, deploy it with the "
        f"{POISONER_BY_TRIGGER_TYPE[trigger_type]!r} poisoner flag)"
    )


if __name__ == "__main__":
    experiment_name, module_name = sys.argv[1], sys.argv[2]
    run(experiment_name, module_name)
