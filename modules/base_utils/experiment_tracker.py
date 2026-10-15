"""
Generic experiment tracking for optimization modules: local loss-curve plots +
optional Weights & Biases mirroring.

Every `run_module.run(experiment_name, module_name, **kwargs)` resolves its config from
`experiments/<experiment_name>/config.toml` (see `modules.base_utils.util.extract_toml`).
`ExperimentTracker` reuses that same resolution so plots/metrics land NEXT TO the config
file that produced the run, independent of wherever `output_dir` (often shared/scratch
storage) happens to point.

Usage in a module's `run()`, without touching any optimization math:

    from modules.base_utils.experiment_tracker import ExperimentTracker

    with ExperimentTracker(experiment_name, module_name, args, slurm_id=slurm_id) as tracker:
        for step in range(n_steps):
            ...
            tracker.log(step, grand_loss=grand_loss.item(), param_loss=param_loss.item())

`tracker.finalize()` (called automatically on context-manager exit, including on
exceptions, so crashed runs still leave diagnostics) writes:
  - `experiments/<experiment_name>/logs/<module_name>_metrics.json` (or the module's own
    `metrics_log_path`, if set, for backward compatibility with existing readers)
  - `experiments/<experiment_name>/plots/<module_name>/<metric>.png` + `overview.png`
  - `experiments/<experiment_name>/plots/<module_name>/run_manifest.json`

W&B is entirely optional: if not installed, disabled in config, or `wandb.init` fails
(e.g. no network on a cluster node), the tracker silently falls back to local-only
logging -- it never raises and never slows down the hot loop.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Callable, Optional

import numpy as np


def generate_full_path(path):
    """Mirrors modules.base_utils.util.generate_full_path, duplicated here so this module
    stays importable without pulling in util.py's torch/cudnn dependencies."""
    return os.path.join(os.getcwd(), path)


def slurmify_path(path, slurm_id):
    """Mirrors modules.base_utils.util.slurmify_path (see generate_full_path's note)."""
    if path is None:
        return path
    return path if slurm_id is None else path.format(slurm_id)


try:
    import matplotlib.pyplot as plt
    from modules.base_utils import plot_style

    plot_style.apply()
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


def compose_callbacks(*callbacks: Optional[Callable]) -> Optional[Callable]:
    """Chains callables sharing a call signature (e.g. mini_train's `callback` slot),
    such as `train_expert`'s checkpoint_callback and a tracker's epoch logger."""
    fns = [f for f in callbacks if f is not None]
    if not fns:
        return None
    if len(fns) == 1:
        return fns[0]

    def _combined(*args, **kwargs):
        for f in fns:
            f(*args, **kwargs)

    return _combined


class ExperimentTracker:
    """`x_label` names the step axis of the plots ("Epoch", "Optimization step", ...).
    `use_config_metrics_path=False` ignores the config's `metrics_log_path` (for modules that
    write their own file there) and keeps the tracker's JSON under logs/."""

    def __init__(self, experiment_name, module_name, args, slurm_id=None, x_label="Step",
                 use_config_metrics_path=True):
        self.experiment_name = experiment_name
        self.module_name = module_name
        self.slurm_id = slurm_id

        config_path = generate_full_path("experiments/" + experiment_name + "/config.toml")
        self.run_dir = Path(config_path).parent
        self.config_path = config_path

        suffix = f"_{slurm_id}" if slurm_id is not None else ""
        default_metrics_path = self.run_dir / "logs" / f"{module_name}{suffix}_metrics.json"
        metrics_log_path = (
            args.get("metrics_log_path") if isinstance(args, dict) and use_config_metrics_path else None
        )
        self.metrics_log_path = Path(
            slurmify_path(metrics_log_path, slurm_id) if metrics_log_path else default_metrics_path
        )

        self.plots_dir = self.run_dir / "plots" / f"{module_name}{suffix}"

        self._history: dict[str, list[tuple[int, float]]] = {}
        self._finalized = False
        self.x_label = x_label
        self._plot_groups = []

        self._wandb = None
        self._wandb_run = None
        self._init_wandb(args if isinstance(args, dict) else {})

    def _init_wandb(self, args):
        wandb_cfg = args.get("wandb", {}) if isinstance(args, dict) else {}
        if not wandb_cfg or not wandb_cfg.get("enabled", False):
            return
        try:
            import wandb
        except ImportError:
            warnings.warn(
                "ExperimentTracker: wandb.enabled=true in config but the `wandb` package "
                "is not installed -- continuing with local-only logging."
            )
            return
        try:
            self._wandb_run = wandb.init(
                project=wandb_cfg.get("project", "flip"),
                entity=wandb_cfg.get("entity"),
                name=wandb_cfg.get("run_name", f"{self.experiment_name}/{self.module_name}"),
                group=wandb_cfg.get("group"),
                tags=wandb_cfg.get("tags"),
                mode=wandb_cfg.get("mode", "online"),
                config={
                    "experiment_name": self.experiment_name,
                    "module_name": self.module_name,
                    "config_path": str(self.config_path),
                    "slurm_id": self.slurm_id,
                    **{k: v for k, v in args.items() if _is_wandb_config_safe(v)},
                },
                reinit=True,
            )
            self._wandb = wandb
        except Exception as e:
            warnings.warn(f"ExperimentTracker: wandb.init failed ({e}) -- disabling wandb.")
            self._wandb = None
            self._wandb_run = None

    def log(self, step, **metrics):
        for key, value in metrics.items():
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            self._history.setdefault(key, []).append((int(step), value))

        if self._wandb is not None:
            try:
                self._wandb.log(metrics, step=int(step))
            except Exception as e:
                warnings.warn(f"ExperimentTracker: wandb.log failed ({e}) -- disabling wandb.")
                self._wandb = None

    def add_plot_group(self, name, keys, ylabel=None, title=None):
        """Also plots `keys` together on one axis (written as `<name>.{png,pdf}`; `name` must not be
        a metric key, whose own plot would share that file)."""
        self._plot_groups.append((name, list(keys), ylabel, title))

    def epoch_callback(self, test_names=("cta", "asr")):
        """Returns a callback matching mini_train/mini_train_multi's `epoch_callback=` slot
        (`fn(epoch, loss, acc, lr, test_metrics=None)`, called once per epoch -- cheap, no
        per-batch overhead). `test_metrics` is the epoch's [(acc, loss), ...] over the
        loop's test_data, logged as `test_names[i]` (accuracy) and `<name>_loss`."""

        def _callback(epoch, loss, acc, lr, test_metrics=None):
            metrics = {"loss": loss, "acc": acc, "lr": lr}
            for name, (test_acc, test_loss) in zip(test_names, test_metrics or []):
                metrics[name] = test_acc
                metrics[f"{name}_loss"] = test_loss
            self.log(epoch, **metrics)

        return _callback

    def _write_metrics_json(self):
        self.metrics_log_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            key: [{"step": s, "value": v} for s, v in points]
            for key, points in self._history.items()
        }
        with open(self.metrics_log_path, "w") as f:
            json.dump(serializable, f, indent=2)

    def _series(self, key):
        steps, values = zip(*self._history[key])
        return np.asarray(steps), np.asarray(values, dtype=float) * plot_style.scale(key)

    def _draw(self, ax, key, color, label=None):
        steps, values = self._series(key)
        window = plot_style.auto_window(len(values))
        if window > 1:
            ax.plot(steps, values, color=color, alpha=0.2, linewidth=1)
            values = plot_style.smooth(values, window)
        ax.plot(steps, values, color=color, label=label)

    def _write_plots(self):
        if not _HAS_MPL or not self._history:
            return []
        written = []
        title = f"{self.module_name}"
        keys = [k for k, v in self._history.items() if v and len({val for _, val in v}) > 1]

        for key in keys:
            fig, ax = plt.subplots(figsize=plot_style.SMALL_FIGSIZE)
            self._draw(ax, key, plot_style.COLORS.get(key, "tab:blue"))
            plot_style.style_axes(ax, self.x_label, plot_style.label(key), title)
            written += plot_style.save(fig, self.plots_dir / key)

        for name, group_keys, ylabel, group_title in self._plot_groups:
            present = [k for k in group_keys if self._history.get(k)]
            if not present:
                continue
            fig, ax = plt.subplots(figsize=plot_style.FIGSIZE)
            for i, key in enumerate(present):
                self._draw(ax, key, plot_style.COLORS.get(key, f"C{i}"), label=plot_style.legend_label(key))
            plot_style.style_axes(ax, self.x_label, ylabel or plot_style.label(present[0]), group_title or title)
            ax.legend(frameon=True, fontsize=10, loc="best")
            written += plot_style.save(fig, self.plots_dir / name)

        if keys:
            ncols = min(3, len(keys))
            nrows = (len(keys) + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
            for i, key in enumerate(keys):
                ax = axes[i // ncols][i % ncols]
                self._draw(ax, key, plot_style.COLORS.get(key, "tab:blue"))
                plot_style.style_axes(ax, self.x_label, plot_style.label(key))
            for i in range(len(keys), nrows * ncols):
                axes[i // ncols][i % ncols].axis("off")
            fig.suptitle(f"{self.experiment_name} / {self.module_name}")
            written += plot_style.save(fig, self.plots_dir / "overview", formats=("png",))

        return written

    def _write_manifest(self, plot_paths):
        manifest = {
            "experiment_name": self.experiment_name,
            "module_name": self.module_name,
            "config_path": str(self.config_path),
            "slurm_id": self.slurm_id,
            "metrics_log_path": str(self.metrics_log_path),
            "plots": [str(p) for p in plot_paths],
            "wandb": {
                "enabled": self._wandb is not None,
                "project": getattr(self._wandb_run, "project", None) if self._wandb_run else None,
                "run_id": getattr(self._wandb_run, "id", None) if self._wandb_run else None,
                "run_url": getattr(self._wandb_run, "url", None) if self._wandb_run else None,
            },
        }
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        with open(self.plots_dir / "run_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    def finalize(self):
        if self._finalized:
            return
        self._finalized = True

        self._write_metrics_json()
        plot_paths = self._write_plots()
        self._write_manifest(plot_paths)

        if self._wandb is not None:
            try:
                for path in (p for p in plot_paths if p.suffix == ".png"):
                    self._wandb.log({f"plots/{path.stem}": self._wandb.Image(str(path))})
                self._wandb.finish()
            except Exception as e:
                warnings.warn(f"ExperimentTracker: wandb finalize failed ({e}).")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize()
        return False


def _is_wandb_config_safe(value):
    return isinstance(value, (int, float, str, bool)) or value is None
