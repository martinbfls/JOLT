"""Full-gradient robust aggregation rules and NNM pre-aggregation.

The legacy rules in `mini_train_multi` / `federated_generate_labels.utils.agg` (mean, median,
trmean, krum, multikrum) aggregate each parameter tensor independently ("per-layer"). This
module adds rules that act on each worker's full flattened gradient, plus an optional
pre-aggregation step. Everything here is differentiable w.r.t. the worker gradients
(selection indices / Weiszfeld weights are computed on detached copies), so it can be used
inside generate_labels_trigger_joint's create_graph=True expert step.

  - krum_full / multikrum_full: Krum / Multi-Krum (Blanchard et al., NeurIPS 2017) on full
    gradients, with the same score as the per-layer version in aggregator/multikrum.py (sum
    of the n-f-1 smallest L2 distances; Multi-Krum averages the m = n-f-2 best by default).
  - geomed: geometric median via smoothed Weiszfeld, following geom_median
    (https://github.com/krishnap25/geom_median; Pillutla, Kakade & Harchaoui, "Robust
    Aggregation for Federated Learning", IEEE TSP 2022).
  - pre_agg="nnm": Nearest Neighbor Mixing, following byzfl (https://github.com/LPD-EPFL/byzfl;
    Allouah et al., "Fixing by Mixing", AISTATS 2023): each worker's gradient is replaced by
    the mean of its n-f nearest neighbors (itself included) before the aggregation rule runs.
    Works with every rule, per-layer ones included (mixing always uses full gradients).
"""

import torch

from modules.base_utils.aggregator.trmean import aggr_trmean
from modules.base_utils.aggregator.multikrum import aggregate as aggr_multikrum

PER_LAYER_AGG_METHODS = ("mean", "median", "trmean", "multikrum", "krum")
FULL_AGG_METHODS = ("krum_full", "multikrum_full", "geomed")
AGG_METHODS = PER_LAYER_AGG_METHODS + FULL_AGG_METHODS
PRE_AGG_METHODS = (None, "none", "nnm")


def uses_full_path(agg_method, pre_agg=None):
    """True when `aggregate` must be used instead of the legacy per-layer loop."""
    return agg_method in FULL_AGG_METHODS or pre_agg not in (None, "none")


def validate(agg_method, pre_agg=None):
    if agg_method not in AGG_METHODS:
        raise ValueError(f"Unknown agg_method {agg_method!r}, expected one of {AGG_METHODS}")
    if pre_agg not in PRE_AGG_METHODS:
        raise ValueError(f"Unknown pre_agg {pre_agg!r}, expected one of {PRE_AGG_METHODS}")


def _pairwise_distances(G):
    """(n, n) L2 distances between the rows of G, computed pair by pair (no Gram-matrix trick,
    which loses precision on near-identical gradients)."""
    return torch.cdist(G, G, compute_mode="donot_use_mm_for_euclid_dist")


def nnm(G, f):
    """Nearest Neighbor Mixing: row i becomes the mean of the n-f rows closest to it (itself included)."""
    n = G.shape[0]
    if not 0 <= f < n:
        raise ValueError(f"NNM needs 0 <= f < n, got f={f}, n={n}")
    k = n - f
    with torch.no_grad():
        idx = _pairwise_distances(G.detach()).topk(k, dim=1, largest=False).indices
        mix = torch.zeros(n, n, dtype=G.dtype, device=G.device).scatter_(1, idx, 1.0 / k)
    return mix @ G


def krum_select(G, f, m):
    """Indices of the m rows of G with the lowest Krum score, best first."""
    n = G.shape[0]
    if n - f - 1 < 1 or not 1 <= m <= n:
        raise ValueError(f"Krum needs n - f - 1 >= 1 and 1 <= m <= n, got n={n}, f={f}, m={m}")
    with torch.no_grad():
        D = _pairwise_distances(G.detach()).double()
        D.fill_diagonal_(float("inf"))
        D = torch.nan_to_num(D, nan=float("inf"))
        scores = D.topk(n - f - 1, dim=1, largest=False).values.sum(dim=1)
        return torch.argsort(scores, stable=True)[:m]


def geometric_median(G, eps=1e-8, maxiter=100, ftol=1e-6):
    """Smoothed Weiszfeld: argmin_z sum_i ||G_i - z||, started from the mean, stopped when the
    objective's relative improvement drops below ftol. Weights are computed without autograd;
    the final weighted average is taken with it, so gradients flow to G."""
    n = G.shape[0]
    with torch.no_grad():
        Gd = G.detach()
        weights = torch.full((n,), 1.0 / n, dtype=G.dtype, device=G.device)
        median = weights @ Gd
        objective = torch.linalg.vector_norm(Gd - median, dim=1).mean()
        for _ in range(maxiter):
            prev_objective = objective
            weights = 1.0 / torch.linalg.vector_norm(Gd - median, dim=1).clamp(min=eps)
            weights = weights / weights.sum()
            median = weights @ Gd
            objective = torch.linalg.vector_norm(Gd - median, dim=1).mean()
            if (prev_objective - objective).abs() <= ftol * objective:
                break
    return weights @ G


def aggregate_layer(grads, agg_method, f):
    """One per-layer rule on a (n_workers, *shape) stack -- same arithmetic as the legacy loops."""
    if agg_method == "mean":
        return grads.mean(dim=0)
    if agg_method == "median":
        return grads.median(dim=0).values
    if agg_method == "trmean":
        return aggr_trmean(grads, f=f)
    if agg_method == "multikrum":
        return aggr_multikrum(grads, f=f)
    if agg_method == "krum":
        return aggr_multikrum(grads, f=f, m=1)
    raise ValueError(f"Unknown agg_method: {agg_method}")


def aggregate(grad_buf, agg_method, f, pre_agg=None, track_selection=False):
    """Aggregates per-worker gradients.

    :param grad_buf: list over parameters of lists over workers of gradient tensors (an empty
        list for a parameter without gradient, kept as None in the output).
    :param track_selection: also return the worker indices a Krum-family rule selected. For the
        full-gradient rules this is the actual selection; for per-layer krum/multikrum it is the
        full-vector proxy mini_train_multi has always reported, on the (possibly mixed) vectors.
    :returns: (list of aggregated gradients aligned with grad_buf, selected indices or None)
    """
    validate(agg_method, pre_agg)
    used = [i for i, b in enumerate(grad_buf) if b]
    n_workers = len(grad_buf[used[0]])
    shapes = [grad_buf[i][0].shape for i in used]
    sizes = [grad_buf[i][0].numel() for i in used]
    G = torch.stack([torch.cat([grad_buf[i][w].reshape(-1) for i in used]) for w in range(n_workers)])

    if pre_agg == "nnm":
        G = nnm(G, f)

    selected = None
    if agg_method in FULL_AGG_METHODS:
        if agg_method == "geomed":
            flat = geometric_median(G)
        else:
            m = 1 if agg_method == "krum_full" else n_workers - f - 2
            sel = krum_select(G, f, m)
            flat = G[sel].mean(dim=0)
            selected = sel.tolist()
        per_param = [c.reshape(s) for c, s in zip(flat.split(sizes), shapes)]
    else:
        per_param = [
            aggregate_layer(c.reshape(n_workers, *s), agg_method, f)
            for c, s in zip(G.split(sizes, dim=1), shapes)
        ]
        if track_selection and agg_method in ("multikrum", "krum"):
            _, selected = aggr_multikrum(list(G.detach()), f=f, m=1 if agg_method == "krum" else None,
                                         return_selected=True)

    out = [None] * len(grad_buf)
    for i, g in zip(used, per_param):
        out[i] = g
    return out, selected
