import random

import numpy as np
import torch
import torch.nn.functional as F

from modules.base_utils.datasets import MTTDataset
from modules.federated_generate_labels.utils import DEFAULT_EXPERT_CONFIG


def build_expert_pool(expert_starts, expert_opt_starts, pool_size):
    """Preloads `pool_size` distinct (params, optimizer-state) checkpoint pairs into RAM, drawn from uniformly per step."""
    all_pairs = list(dict.fromkeys(zip(expert_starts, expert_opt_starts)))
    if pool_size > len(all_pairs):
        print(
            f"WARNING: pool_size={pool_size} > {len(all_pairs)} distinct available expert "
            f"checkpoints for this run -- using all {len(all_pairs)} instead."
        )
        pool_size = len(all_pairs)
    pool_paths = random.sample(all_pairs, pool_size)
    pool = [
        (
            {k: v.float().cpu() for k, v in torch.load(p_path, map_location="cpu").items()},
            torch.load(o_path, map_location="cpu"),
        )
        for p_path, o_path in pool_paths
    ]
    return pool, pool_size


class TriggerMTTDataset(MTTDataset):
    """Wraps `MTTDataset`, additionally returning a boolean `is_poisoned` flag per example."""

    def __getitem__(self, i: int):
        is_poisoned = i >= len(self.distill)
        train_x, train_oh, distill_x, distill_oh, idx = super().__getitem__(i)
        return train_x, train_oh, distill_x, distill_oh, idx, is_poisoned

    @classmethod
    def from_mtt_dataset(cls, mtt_dataset):
        return cls(
            mtt_dataset.train, mtt_dataset.distill, mtt_dataset.poison_inds,
            mtt_dataset.transform, mtt_dataset.n_classes,
        )


def _sample_trajectory_index(n_traj, alpha_ckpt):
    """Single draw biased toward early checkpoints: P(k) proportional to exp(-alpha_ckpt*k)."""
    ks = torch.arange(0, n_traj, dtype=torch.float)
    probs = torch.exp(-alpha_ckpt * ks)
    probs = probs / probs.sum()
    return int(torch.multinomial(probs, 1).item())


def extract_experts_biased(expert_config, expert_path, iterations, alpha_ckpt, expert_opt_path=None):
    """Like `federated_generate_labels.utils.extract_experts`, but draws the trajectory index via `_sample_trajectory_index` instead of uniformly."""
    config = {**DEFAULT_EXPERT_CONFIG, **expert_config}
    n_traj = config['max'] - config['min']
    expert_starts, expert_opt_starts = [], []

    for _ in range(iterations):
        for s in config['trajectories']:
            expert = np.random.randint(config['experts'])
            trajectory = config['min'] + _sample_trajectory_index(n_traj, alpha_ckpt) + 1
            expert_starts.append(expert_path.format(expert, trajectory, str(s)))
            if expert_path:
                expert_opt_starts.append(expert_opt_path.format(expert, trajectory, str(s)))
    return expert_starts, expert_opt_starts


# Anti-collapse regularizers: guard against expert_asr -> 0 while the matching term keeps
# improving (a clean expert trivially satisfies the alignment objective).

def cosine_to(delta, mu, eps=1e-8):
    """cos(delta, mu), flattened, mu treated as a fixed (detached) target direction."""
    return F.cosine_similarity(
        delta.reshape(1, -1), mu.reshape(1, -1).detach(), eps=eps,
    ).squeeze(0)


def directional_floor_penalty(delta, mu_target, align_kappa, eps=1e-8):
    """L_align = relu(align_kappa - cos(delta, mu_target)) -- a floor, active below align_kappa."""
    cos = cosine_to(delta, mu_target, eps=eps)
    return F.relu(align_kappa - cos), cos


def magnitude_floor_penalty(delta, delta_min):
    """L_mag = relu(delta_min - ||delta||_2) -- prevents the directional floor from being satisfied via delta -> 0."""
    norm = delta.norm()
    return F.relu(delta_min - norm), norm


def _project_onto_cone(delta, mu_target, align_kappa, eps=1e-8):
    """Euclidean projection of delta onto the cone {x : cos(x, mu_target) >= align_kappa}."""
    u = mu_target / (mu_target.norm() + eps)
    h = (delta * u).sum()
    perp = delta - h * u
    r = perp.norm()
    delta_norm = delta.norm()
    if delta_norm < eps:
        return delta
    cos_phi = h / delta_norm
    if cos_phi.item() >= align_kappa:
        return delta

    alpha = torch.acos(
        torch.clamp(torch.tensor(align_kappa, device=delta.device, dtype=delta.dtype), -1.0, 1.0)
    )
    cos_a, sin_a = torch.cos(alpha), torch.sin(alpha)
    t = h * cos_a + r * sin_a
    if t.item() <= 0:
        return torch.zeros_like(delta)
    perp_unit = perp / (r + eps)
    return (t * cos_a) * u + (t * sin_a) * perp_unit


def _project_onto_magnitude_floor(delta, delta_min, mu_target, eps=1e-8):
    """Pushes delta radially out to ||delta||_2 == delta_min if it fell short."""
    n = delta.norm()
    if n.item() >= delta_min:
        return delta
    if n.item() < eps:
        return delta_min * mu_target / (mu_target.norm() + eps)
    return delta * (delta_min / n)


def grad_mismatch_penalty(clean_grads, poison_grads, eps=1e-8):
    """||grad(L_c) - grad(L_p)(delta)||^2 / ||grad(L_c)||^2, summed across parameter tensors."""
    diff_sq = sum((gc - gp).pow(2).sum() for gc, gp in zip(clean_grads, poison_grads))
    clean_sq = sum(gc.pow(2).sum() for gc in clean_grads)
    return diff_sq / (clean_sq + eps)


def grad_cosine_penalty(clean_grads, poison_grads, eps=1e-8):
    """1 - cos(grad(L_c), grad(L_p)(delta)); scale-invariant alternative to `grad_mismatch_penalty`."""
    dot = sum((gc * gp).sum() for gc, gp in zip(clean_grads, poison_grads))
    clean_norm = torch.sqrt(sum(gc.pow(2).sum() for gc in clean_grads))
    poison_norm = torch.sqrt(sum(gp.pow(2).sum() for gp in poison_grads))
    cos = dot / (clean_norm * poison_norm + eps)
    return 1 - cos


def margin_floor_penalty(logits_trig, target_label, margin_min):
    """Floor on the triggered rows' classification margin toward the target: relu(margin_min - margin)."""
    top2 = logits_trig.topk(2, dim=1)
    tgt = logits_trig[:, target_label]
    runner = torch.where(top2.indices[:, 0] == target_label,
                          top2.values[:, 1], top2.values[:, 0])
    margin = tgt - runner
    return F.relu(margin_min - margin).mean(), margin.mean()


def poison_consistency(poison_grad_chunks, eps=1e-8):
    """1 - mean cos between each poisoned client's gradient chunk and their mean; 0.0 when num_poisoned<=1."""
    n_chunks = len(poison_grad_chunks[0])
    G = torch.stack([
        torch.cat([poison_grad_chunks[i][j].reshape(-1) for i in range(len(poison_grad_chunks))])
        for j in range(n_chunks)
    ])
    g_bar = G.mean(0)
    return 1.0 - F.cosine_similarity(G, g_bar.unsqueeze(0).expand_as(G), dim=1, eps=eps).mean()


def lpips_penalty(delta, x_raw_clean, lpips_model, is_warp=False, warp_s=1.0):
    """Perceptual (LPIPS) distance between the clean raw image(s) and their triggered counterpart,
    differentiable w.r.t. delta (the pixel delta under trigger_type='additive', or the warp
    control grid under trigger_type='warp' -- composed via apply_warp instead of addition)."""
    if is_warp:
        from modules.optimizing_trigger.utils import apply_warp

        x_trig_raw = apply_warp(x_raw_clean, delta, warp_s)
    elif x_raw_clean.dim() == 3:
        x_trig_raw = (x_raw_clean + delta).clamp(0, 1)
    else:
        x_trig_raw = (x_raw_clean + delta.unsqueeze(0)).clamp(0, 1)
    x_clean_scaled = x_raw_clean * 2 - 1
    x_trig_scaled = x_trig_raw * 2 - 1
    return lpips_model(x_trig_scaled, x_clean_scaled).mean()


def project_trigger_constraints(delta, mu_target, epsilon, align_kappa, delta_min, n_iters=8, eps=1e-8):
    """Alternating projection onto {||delta||_inf<=epsilon} inter cone(align_kappa) inter {||delta||_2>=delta_min}; skips the cone step when align_kappa<=0."""
    for _ in range(n_iters):
        delta = delta.clamp(-epsilon, epsilon)
        if align_kappa > 0:
            delta = _project_onto_cone(delta, mu_target, align_kappa, eps=eps)
        delta = _project_onto_magnitude_floor(delta, delta_min, mu_target, eps=eps)
    return delta
