"""The trigger itself: how each component is initialized and how they compose into a triggered
image, in raw [0, 1] space.

This module deliberately imports NOTHING from the project -- torch only. It is the single source
of truth shared by three callers that would otherwise form an import cycle: the optimizer
(generate_labels_trigger_joint.run_module), the LPIPS penalty, and the deploy-time poisoners in
modules.base_utils.datasets, which modules.optimizing_trigger.utils itself imports. Keeping this
file project-free is what lets the poisoners import it directly instead of reaching into
`utils` from inside a method, and what lets scripts/check_trigger_composition.py verify the
composition without a dataset, a model or a checkpoint. Please keep it that way.

Everything here is re-exported from modules.optimizing_trigger.utils, so existing import sites
are unaffected. What stays in `utils` is the part that is NOT dataset-agnostic: the per-dataset
normalization and the raw_to_*_preprocess wrappers built on it.
"""

import torch
import torch.nn.functional as F


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


def init_color_params(device="cuda"):
    """Photometric component of the trigger: a 3x4 [A | b] block, zero-init (the identity map, so
    an unoptimized color component is an exact no-op). The deployed map is

        x -> clamp((I + clamp(A, -a_max, a_max)) @ x + clamp(b, -b_max, b_max), 0, 1)

    i.e. 12 parameters for a global, spatially uniform color shift. It is invisible at small
    budgets, it is invariant to crops and flips -- so it is a feature the victim's label flips can
    reproduce from clean data, which is the binding constraint of this threat model -- and it costs
    nothing to compute."""
    return torch.zeros(3, 4, device=device, requires_grad=True)


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


def apply_color(x_raw: torch.Tensor, color_params: torch.Tensor,
                a_max: float = 0.05, b_max: float = 0.02):
    """Global affine color map of x_raw ((C, H, W) or (B, C, H, W), in [0, 1]). The budgets are
    clamped HERE (single source of truth for them, as apply_warp is for warp_s) as well as after
    every optimizer step, so the deployed map always respects them."""
    squeeze = x_raw.dim() == 3
    if squeeze:
        x_raw = x_raw.unsqueeze(0)
    A = color_params[:, :3].clamp(-a_max, a_max)
    b = color_params[:, 3].clamp(-b_max, b_max)
    M = torch.eye(3, device=x_raw.device, dtype=x_raw.dtype) + A.to(x_raw.dtype)
    out = torch.einsum("ij,bjhw->bihw", M, x_raw) + b.to(x_raw.dtype).view(1, 3, 1, 1)
    out = out.clamp(0, 1)
    return out.squeeze(0) if squeeze else out


def contrast_mask(x_raw: torch.Tensor, ksize: int = 5, floor: float = 0.25):
    """Per-pixel weight in [floor, 1] for the additive budget: ~floor in flat regions, ~1 in
    textured ones, from the local standard deviation of x_raw over a ksize x ksize window.

    A uniform Linf budget spends as much perturbation on a flat sky (where it is maximally
    visible) as on a texture (where it hides), so at equal perceived visibility it carries less
    trigger energy. This reweighting is image-adaptive with ZERO extra parameters, differentiable
    w.r.t. x_raw (a warp placed before it still receives its gradient) and deploy-safe: the
    poisoner sees exactly the image the optimizer did, and apply_trigger_raw composes both the
    same way.

    The weight is <= 1 by construction, so ||w * delta||_inf <= ||delta||_inf <= epsilon -- the
    headline Linf budget stays a valid bound on the deployed perturbation. The mask is computed
    on luminance, not per channel, so it scales the trigger without distorting its color."""
    squeeze = x_raw.dim() == 3
    if squeeze:
        x_raw = x_raw.unsqueeze(0)
    gray = x_raw.mean(dim=1, keepdim=True)
    pad = ksize // 2
    mean = F.avg_pool2d(gray, ksize, stride=1, padding=pad, count_include_pad=False)
    mean_sq = F.avg_pool2d(gray * gray, ksize, stride=1, padding=pad, count_include_pad=False)
    sd = (mean_sq - mean * mean).clamp_min(0).sqrt()
    sd_max = sd.amax(dim=(-2, -1), keepdim=True)
    w = floor + (1.0 - floor) * sd / (sd_max + 1e-8)
    return w.squeeze(0) if squeeze else w


def tv_loss(delta):
    '''Anisotropic total variation of a (C, H, W) trigger perturbation.'''
    dh = (delta[:, 1:, :] - delta[:, :-1, :]).abs().sum()
    dw = (delta[:, :, 1:] - delta[:, :, :-1]).abs().sum()
    return dh + dw


def eot_augment(x_raw: torch.Tensor, pad: int = 4, flip: bool = True):
    """One draw of the victim's own train-time augmentation (TRANSFORM_TRAIN: RandomCrop(H,
    padding=pad) then RandomHorizontalFlip), applied to the TRIGGERED image in raw [0, 1] space --
    the order the deploy pipeline uses, where the poisoner runs on the PIL image and the transform
    after it (get_matching_datasets, modules/base_utils/datasets.py).

    The expert whose trajectory is matched was trained on triggered images under that augmentation,
    while the optimization loop feeds it un-augmented ones; one fresh draw per step is an unbiased
    estimator of the augmented objective and costs no extra forward pass. It matters most for the
    warp, which is not equivariant: a horizontal flip maps a displacement field (dx, dy) to its
    mirror (-dx, dy), so a field optimized without it spends half its signal on an orientation the
    victim sees half the time.

    The crop offset is drawn per BATCH (one slice of a padded tensor, free) and the flip per
    SAMPLE. `pad` is in RAW image pixels, i.e. it matches TRANSFORM_TRAIN for the 32x32 datasets;
    a needs_big_ims model crops at its upsampled resolution instead, so the scales differ there."""
    squeeze = x_raw.dim() == 3
    if squeeze:
        x_raw = x_raw.unsqueeze(0)
    B, _, H, W = x_raw.shape
    out = x_raw
    if flip:
        mask = torch.rand(B, device=out.device) < 0.5
        out = torch.where(mask.view(B, 1, 1, 1), out.flip(-1), out)
    if pad > 0:
        out = F.pad(out, (pad, pad, pad, pad))
        ox = int(torch.randint(0, 2 * pad + 1, (1,)).item())
        oy = int(torch.randint(0, 2 * pad + 1, (1,)).item())
        out = out[:, :, oy:oy + H, ox:ox + W]
    return out.squeeze(0) if squeeze else out


def apply_trigger_raw(x_raw: torch.Tensor, delta: torch.Tensor = None,
                      control_grid: torch.Tensor = None, s: float = 1.0,
                      color_params: torch.Tensor = None,
                      color_a_max: float = 0.05, color_b_max: float = 0.02,
                      contrast_mask_floor: float = None, contrast_mask_ksize: int = 5):
    """The composed trigger in RAW [0, 1] space -- single source of truth for the composition
    order, shared by the optimizer (generate_labels_trigger_joint), the LPIPS penalty and the
    deploy-time poisoners (modules/base_utils/datasets.py):

        x1 = apply_warp(x_raw, control_grid, s)            (control_grid is not None)
        x2 = apply_color(x1, color_params, a_max, b_max)   (color_params is not None)
        out = clamp(x2 + w(x2) * delta, 0, 1)              (delta is not None)

    where w is contrast_mask(., contrast_mask_ksize, contrast_mask_floor) when
    contrast_mask_floor is given and 1 otherwise.

    Every component may be None: `control_grid=None` gives the pure additive trigger
    (trigger_type='additive'), `delta=None` the pure warp ('warp', apply_warp alone -- it needs
    no clamp, grid_sample interpolating an in-range image stays in range), both the hybrid one
    ('hybrid'). The warp comes FIRST so the additive pattern lands unaltered on top of the warped
    image: ||.||_inf of the additive component in the final image is then at most ||delta||_inf,
    and each component zeroed out reduces the composed trigger exactly to the others.

    The photometric map is pointwise, so it commutes with the warp's bilinear interpolation up to
    the clamp; it is placed between the two so that the additive pattern still lands last and the
    Linf budget keeps bounding it. The contrast mask reads x2 -- the image the deploy-time
    poisoner also has in hand at that point -- so optimizer and poisoner compute the same weights.
    Both are no-ops at their defaults (zero color_params is the identity map, no mask), which is
    what keeps an older saved trigger deploying bit-for-bit as before."""
    x_trig = x_raw if control_grid is None else apply_warp(x_raw, control_grid, s)
    if color_params is not None:
        x_trig = apply_color(x_trig, color_params, color_a_max, color_b_max)
    if delta is not None:
        d = delta if x_trig.dim() == 3 else delta.unsqueeze(0)
        if contrast_mask_floor is not None:
            d = d * contrast_mask(x_trig, contrast_mask_ksize, contrast_mask_floor)
        x_trig = (x_trig + d).clamp(0, 1)
    return x_trig
