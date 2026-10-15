import torch
import numpy as np
import scipy.sparse as sp
import osqp
import torch.nn.functional as F
from modules.base_utils.datasets import (
    get_n_classes, load_dataset, MappedDataset, TRANSFORM_TRAIN_XY, TRANSFORM_TEST_XY,
    pick_poisoner,
    CIFAR_TRANSFORM_NORMALIZE_MEAN,   CIFAR_TRANSFORM_NORMALIZE_STD,
    CIFAR_100_TRANSFORM_NORMALIZE_MEAN, CIFAR_100_TRANSFORM_NORMALIZE_STD,
    SVHN_TRANSFORM_NORMALIZE_MEAN,    SVHN_TRANSFORM_NORMALIZE_STD,
    TINY_IMAGENET_TRANSFORM_NORMALIZE_MEAN, TINY_IMAGENET_TRANSFORM_NORMALIZE_STD,
)
from modules.federated_generate_labels.utils import DEFAULT_EXPERT_CONFIG, DEFAULT_ATTACK_ITERATIONS
from modules.base_utils.util import needs_big_ims
from PIL import Image
from torchvision import transforms
from torch.utils.data import Subset, ConcatDataset

RAW_TRANSFORM_X = transforms.ToTensor()
RAW_TRANSFORM_Y = lambda y: y

DATASET_NORMALIZATION = {
    'cifar':         (CIFAR_TRANSFORM_NORMALIZE_MEAN,       CIFAR_TRANSFORM_NORMALIZE_STD),
    'cifar_100':     (CIFAR_100_TRANSFORM_NORMALIZE_MEAN,   CIFAR_100_TRANSFORM_NORMALIZE_STD),
    'svhn':          (SVHN_TRANSFORM_NORMALIZE_MEAN,        SVHN_TRANSFORM_NORMALIZE_STD),
    'tiny_imagenet': (TINY_IMAGENET_TRANSFORM_NORMALIZE_MEAN, TINY_IMAGENET_TRANSFORM_NORMALIZE_STD),
}

def get_norm_tensors(dataset_flag: str, device):
    mean, std = DATASET_NORMALIZATION[dataset_flag]
    mean = torch.tensor(mean, device=device).view(3, 1, 1)
    std  = torch.tensor(std,  device=device).view(3, 1, 1)
    return mean, std


def get_transforms(dataset_flag, train=True, big=False):
    key = dataset_flag + ('_big' if big else '')
    if train:
        return TRANSFORM_TRAIN_XY[key]
    else:
        return TRANSFORM_TEST_XY[key]


def sample_checkpoints(K, S, alpha=0.01, device="cpu"):
    ks = torch.arange(0, K, device=device, dtype=torch.float)
    probs = torch.exp(-alpha * ks)
    probs = probs / probs.sum()
    idx = torch.multinomial(probs, S, replacement=False)
    idx = idx.tolist()
    if K - 1 not in idx:
        idx.append(K - 1)
    idx.sort()
    return idx


def init_delta(mu_shape, strength=6.0, freq=16, horizontal=True, device="cuda", init='stripe'):
    if init == 'stripe':
        C, H, W = mu_shape
        sin_1d = torch.sin(torch.linspace(0, freq * torch.pi, W, device=device))
        mask = sin_1d.view(1, 1, W).expand(C, H, W)
        if horizontal:
            mask = mask.transpose(1, 2)
        delta = strength * mask.clone()
        delta.requires_grad_(True)
    elif init == 'random':
        delta = strength * torch.rand(mu_shape, device=device)
        delta.requires_grad_(True)
    else:
        raise ValueError(f"Unknown init type: {init}")
    return delta


def init_warp_field(k, s, device="cuda"):
    """Control grid (1, 2, k, k) of a WaNet-style warp, in pixels (dx, dy); zero = identity warp. `s` (max
    displacement in pixels) is enforced by apply_warp / the optimizer's post-step clamp, not by the init."""
    return torch.zeros(1, 2, k, k, device=device, requires_grad=True)


def compute_batch_gradients(model, loss_fn, batch, create_graph, retain_graph=False):
    model.zero_grad(set_to_none=True)
    x, y = batch
    logits = model(x)
    loss = loss_fn(logits, y)
    grads = torch.autograd.grad(
        loss,
        model.parameters(),
        create_graph=create_graph,
        retain_graph=retain_graph,
    )
    return grads, logits


def trigger_penalty(delta, mu, eps=1e-8):
    delta_f = delta.view(1, -1)
    mu_f = mu.view(1, -1).detach()
    cos = F.cosine_similarity(delta_f, mu_f).mean()
    return cos + 1.0


def trigger_penalty_hinge(delta, mu_target, mu_source, kappa, eps=1e-8):
    delta_f = delta.view(1, -1)
    diff_f = (mu_target - mu_source).view(1, -1).detach()
    cos = F.cosine_similarity(delta_f, diff_f, eps=eps).mean()
    return F.relu(cos - kappa)


def tv_loss(delta):
    '''Anisotropic total variation of a (C, H, W) trigger perturbation.'''
    dh = (delta[:, 1:, :] - delta[:, :-1, :]).abs().sum()
    dw = (delta[:, :, 1:] - delta[:, :, :-1]).abs().sum()
    return dh + dw

def extract_experts(expert_config, expert_path):
    config = {**DEFAULT_EXPERT_CONFIG, **expert_config}
    expert_starts = []
    for expert in range(config['experts']):
        for epoch in range(config['min'], config['max']):
            for s in config['trajectories']:
                expert_starts.append(expert_path.format(str(expert), str(epoch + 1), str(s)))
    return expert_starts


def get_clean_dataset(dataset_flag, train=True, big=False):
    transform = get_transforms(dataset_flag, train=train, big=big)
    base_dataset = load_dataset(dataset_flag, train=train)
    return MappedDataset(base_dataset, transform)


def get_poison_dataset(dataset_flag, source_label, target_label, delta,
                       train=True, train_pct=1.0, big=False,
                       lambda_target=None, lambda_overflow="clip", seed=0,
                       include_clean=True):
    transform = get_transforms(dataset_flag, train=train, big=big)
    base_dataset = load_dataset(dataset_flag, train=train)

    if train_pct < 1.0:
        n = int(len(base_dataset) * train_pct)
        base_dataset = Subset(base_dataset, np.arange(n))

    labels = np.array([y for _, y in base_dataset])
    poison_inds = np.where(labels == source_label)[0]

    if lambda_target is not None:
        if not (0.0 < lambda_target < 1.0):
            raise ValueError(f"lambda_target must be in (0, 1), got {lambda_target}")
        n_base = len(base_dataset)
        n_s = len(poison_inds)
        n_add = round(lambda_target * n_base / (1 - lambda_target))
        rng = np.random.RandomState(seed)

        if n_add > n_s:
            beta_max = n_s / (n_base + n_s)
            if lambda_overflow == "clip":
                print(
                    f"[get_poison_dataset] lambda_overflow='clip': lambda_target="
                    f"{lambda_target:.6f} needs n_add={n_add} > n_s={n_s} available "
                    f"source-class ({dataset_flag}, source_label={source_label}, "
                    f"train={train}) examples; clipping to n_add={n_s}. Effective "
                    f"lambda={n_s / (n_base + n_s):.6f}, beta_max={beta_max:.6f}."
                )
            elif lambda_overflow == "duplicate":
                poison_inds = rng.choice(poison_inds, size=n_add, replace=True)
            else:
                raise ValueError(
                    f"Unknown lambda_overflow={lambda_overflow!r}, expected "
                    "'clip' or 'duplicate'."
                )
        elif n_add < n_s:
            poison_inds = rng.choice(poison_inds, size=n_add, replace=False)
        # n_add == n_s: use poison_inds as-is.

    poisoner = pick_poisoner('optimized', dataset_flag, target_label, delta)

    poison_subset = Subset(base_dataset, poison_inds)
    poison_dataset = MappedDataset(poison_subset, poisoner)
    poison_dataset = MappedDataset(poison_dataset, transform)

    if not include_clean:
        return poison_dataset

    clean_dataset = MappedDataset(base_dataset, transform)
    return ConcatDataset([clean_dataset, poison_dataset])


def preprocess_for_model(x: torch.Tensor, dataset_flag: str, model_flag: str):
    big = needs_big_ims(model_flag)
    mean, std = get_norm_tensors(dataset_flag, x.device)

    if big:
        squeeze = x.dim() == 3
        if squeeze:
            x = x.unsqueeze(0)
        x = F.interpolate(x, size=224, mode='bilinear', align_corners=False)
        if squeeze:
            x = x.squeeze(0)

    if x.dim() == 4:
        mean = mean.unsqueeze(0)
        std  = std.unsqueeze(0)
    x = (x - mean) / std
    return x


def get_raw_clean_dataset(dataset_flag, train=True):
    base_dataset = load_dataset(dataset_flag, train=train)
    def mapper(sample):
        x, y = sample
        return RAW_TRANSFORM_X(x), y  # PIL → tensor [0,1]
    return MappedDataset(base_dataset, mapper)


def get_mu(dataset_flag, y_target, device, model_flag=None):
    dataset = get_raw_clean_dataset(dataset_flag, train=True)
    xs = [x for x, y in dataset if y == y_target]
    if not xs:
        raise ValueError(f"No samples found for class {y_target}")
    mu = torch.stack(xs).to(device).mean(dim=0)
    return mu


def move_to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device)
    elif isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(b, device) for b in batch)
    return batch


def raw_to_preprocess(x_raw: torch.Tensor, dataset_flag: str, model_flag: str = None):
    if model_flag:
        return preprocess_for_model(x_raw, dataset_flag, model_flag)
    mean, std = get_norm_tensors(dataset_flag, x_raw.device)
    if x_raw.dim() == 4:
        mean = mean.unsqueeze(0)
        std  = std.unsqueeze(0)
    return (x_raw - mean) / std


def raw_to_trigger_preprocess(x_raw: torch.Tensor, delta: torch.Tensor,
                               dataset_flag: str, model_flag: str = None):
    if x_raw.dim() == 3:
        x_trig = (x_raw + delta).clamp(0, 1)
    else:
        x_trig = (x_raw + delta.unsqueeze(0)).clamp(0, 1)
    return raw_to_preprocess(x_trig, dataset_flag, model_flag)


def apply_warp(x_raw: torch.Tensor, control_grid: torch.Tensor, s: float):
    """Image-agnostic warp of x_raw ((C, H, W) or (B, C, H, W), in [0, 1]). The (1, 2, k, k) control grid is a
    displacement field in PIXELS (dx, dy), clamped to [-s, s] here (single source of truth for the budget),
    bicubically upsampled to (H, W), converted to normalized coordinates and added to the identity sampling grid."""
    squeeze = x_raw.dim() == 3
    if squeeze:
        x_raw = x_raw.unsqueeze(0)
    B, C, H, W = x_raw.shape
    cg = control_grid.clamp(-s, s)
    ups = F.interpolate(cg, size=(H, W), mode="bicubic", align_corners=True)
    ups = ups.permute(0, 2, 3, 1)  # (1, H, W, 2) = (dx_px, dy_px)
    scale = torch.tensor([2 / (W - 1), 2 / (H - 1)], device=x_raw.device, dtype=x_raw.dtype).view(1, 1, 1, 2)
    identity_theta = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], device=x_raw.device, dtype=x_raw.dtype)
    base = F.affine_grid(identity_theta, (1, C, H, W), align_corners=True)
    grid = (base + ups * scale).clamp(-1, 1)
    warped = F.grid_sample(x_raw, grid.expand(B, -1, -1, -1), mode="bilinear", align_corners=True,
                           padding_mode="border")
    return warped.squeeze(0) if squeeze else warped


def raw_to_warp_preprocess(x_raw: torch.Tensor, control_grid: torch.Tensor,
                           dataset_flag: str, model_flag: str = None, s: float = 1.0):
    return raw_to_preprocess(apply_warp(x_raw, control_grid, s), dataset_flag, model_flag)
