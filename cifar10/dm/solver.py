# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
# Distance Marching samplers: gradient descent and sphere tracing on the
# learned distance field (optionally with unadjusted Langevin noise).

import datetime
import os
from typing import Callable, Optional, Sequence, Tuple, Union

import torch
from torch import nn, Tensor
from torch.nn.parallel import DistributedDataParallel


def build_sigma_schedule(schedule_dict: dict) -> Callable[[float], float]:
    """Build a piecewise-linear sigma function from a {progress_ratio: sigma} dict.

    Rules:
      - Between specified knots: linear interpolation.
      - Before the smallest key: constant equal to the value at the smallest key.
      - After  the largest  key: constant equal to the value at the largest  key.
    """
    keys = sorted(float(k) for k in schedule_dict.keys())
    vals = [float(schedule_dict[k]) for k in keys]

    def sigma_fn(progress: float) -> float:
        if progress <= keys[0]:
            return vals[0]
        if progress >= keys[-1]:
            return vals[-1]
        for i in range(len(keys) - 1):
            if keys[i] <= progress <= keys[i + 1]:
                t = (progress - keys[i]) / (keys[i + 1] - keys[i])
                return vals[i] + t * (vals[i + 1] - vals[i])
        return vals[-1]

    return sigma_fn


def get_base_model(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap DistributedDataParallel / EMA wrappers (in any order) to the base model."""
    while True:
        if isinstance(model, DistributedDataParallel):
            model = model.module
            continue
        # EMA wrapper: has .model and either .decay or .shadow_params
        if hasattr(model, "model") and (
            hasattr(model, "decay") or hasattr(model, "shadow_params")
        ):
            model = model.model
            continue
        break
    return model


class DistanceMarchingSolver(nn.Module):
    """Deterministic samplers driven by a Distance Marching model.

    The model receives ``(x, timesteps, extra, return_dist=True)`` and returns
    ``(grad, dist)``: the direction pointing away from the data manifold and the
    predicted distance to it. ``timesteps`` is a placeholder (the model has no
    time input).

    Args:
        velocity_model: the Distance Marching model (possibly EMA/DDP wrapped).
        output_dir: directory for the per-sample statistics file. If None, uses
            ``$OUTPUT_DIR``, ``./output_dir`` or the current directory.
        is_main_process: only the main process prints and saves statistics.
    """

    def __init__(
        self,
        velocity_model: Union[nn.Module, Callable],
        output_dir: Optional[str] = None,
        is_main_process: bool = True,
    ):
        super().__init__()
        self.velocity_model = velocity_model
        self.output_dir = output_dir
        self.is_main_process = is_main_process
        self.stats_saved = False

    def sample_loop(
        self,
        func: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]],
        y0: Tensor,
        timesteps: Tensor,
        *,
        step_size: Optional[float] = None,
        method: str = "gd",
        criteria: str = "fixed_steps",
        threshold: float = 1e-5,
        ula_schedule: Optional[Callable[[float], float]] = None,
    ) -> Tensor:
        """Run ``len(timesteps) - 1`` marching updates starting from ``y0``.

        Args:
            func: maps ``(x, timesteps) -> (grad, dist)``.
            y0: initial state, shape [batch_size, ...].
            timesteps: 1-D tensor; only its length (number of steps + 1) is used.
            step_size: step size eta.
            method: 'gd' (x <- x - eta * grad), 'st' (x <- x - eta * dist * grad),
                'gd_ula' / 'st_ula' (same plus annealed Langevin noise).
            criteria: 'fixed_steps', 'grad_norm', 'dist_value' or 'step_norm'.
            threshold: threshold for the adaptive criteria.
            ula_schedule: sigma schedule for the *_ula methods.

        Returns:
            Tensor of shape [len(timesteps), batch_size, ...] with the state
            after every step (index 0 is ``y0``).
        """
        assert step_size is not None, "step_size must be specified for distance marching"

        solution = torch.empty(
            len(timesteps), *y0.shape, dtype=y0.dtype, device=y0.device
        )
        solution[0] = y0

        x_current = y0.clone()
        batch_size = y0.shape[0]
        num_steps = len(timesteps) - 1

        # Which samples are still being updated (only changes for adaptive criteria).
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=y0.device)
        step_counter = torch.zeros(batch_size, dtype=torch.long, device=y0.device)

        # Placeholder time value: required by the model signature but unused.
        placeholder_t = torch.full((batch_size,), timesteps[0].item(), device=y0.device)

        last_dist = None

        for step in range(num_steps):
            grad, dist = func(x=x_current, timesteps=placeholder_t)
            x_current = x_current.detach()
            last_dist = dist

            if method == "gd":
                update = -step_size * grad
            elif method == "gd_ula":
                update = -step_size * grad
                progress = step / num_steps
                sigma = ula_schedule(progress)
                if sigma > 0:
                    noise = torch.randn_like(x_current)
                    update = update + sigma * (2 * step_size) ** 0.5 * noise
            elif method == "st_ula":
                dist_expanded = dist.reshape(batch_size, *([1] * (grad.ndim - 1)))
                update = -step_size * grad * dist_expanded
                progress = step / num_steps
                sigma = ula_schedule(progress)
                if sigma > 0:
                    noise = torch.randn_like(x_current)
                    noise_scale = sigma * torch.sqrt(
                        2 * step_size * torch.abs(dist_expanded)
                    )
                    update = update + noise_scale * noise
            elif method == "st":
                dist_expanded = dist.reshape(batch_size, *([1] * (grad.ndim - 1)))
                update = -step_size * grad * dist_expanded
            else:
                raise ValueError(f"Unknown method: {method}")

            if criteria == "grad_norm":
                grad_norm = grad.reshape(batch_size, -1).norm(dim=-1)
                active_mask = grad_norm > threshold
            elif criteria == "dist_value":
                active_mask = dist.squeeze(-1) > threshold
            elif criteria == "step_norm":
                active_mask = update.reshape(batch_size, -1).norm(dim=-1) > threshold
            elif criteria == "fixed_steps":
                pass
            else:
                raise ValueError(f"Unknown criteria: {criteria}")

            mask_expanded = active_mask.view(batch_size, *([1] * (x_current.ndim - 1)))
            x_current = torch.where(mask_expanded, x_current + update, x_current)
            step_counter = step_counter + active_mask.long()
            solution[step + 1] = x_current

            if criteria != "fixed_steps" and not active_mask.any():
                if step + 2 < len(timesteps):
                    for remaining_idx in range(step + 2, len(timesteps)):
                        solution[remaining_idx] = x_current
                if not self.stats_saved and self.is_main_process:
                    self._print_and_save_statistics(
                        step_counter, criteria, reached_limit=False, final_dist=last_dist
                    )
                break
        else:
            if not self.stats_saved and self.is_main_process:
                self._print_and_save_statistics(
                    step_counter,
                    criteria,
                    reached_limit=(criteria != "fixed_steps"),
                    final_dist=last_dist,
                )

        return solution

    def _print_and_save_statistics(
        self,
        step_counter: Tensor,
        criteria: str,
        reached_limit: bool,
        final_dist: Optional[Tensor] = None,
    ):
        steps_np = step_counter.cpu().numpy()
        batch_size = len(steps_np)

        stats_msg = (
            f"Distance Marching Statistics (criteria={criteria}):\n"
            f"  Effective steps per sample:\n"
            f"    Max:  {int(steps_np.max())}\n"
            f"    Min:  {int(steps_np.min())}\n"
            f"    Mean: {float(steps_np.mean()):.2f}\n"
        )

        dist_per_sample = None
        if final_dist is not None:
            dist_np = final_dist.detach().cpu().numpy()
            dist_flat = dist_np.flatten()
            stats_msg += (
                f"  Final distance values:\n"
                f"    Max:  {float(dist_flat.max()):.6f}\n"
                f"    Min:  {float(dist_flat.min()):.6f}\n"
                f"    Mean: {float(dist_flat.mean()):.6f}\n"
            )
            if dist_np.ndim > 1:
                dist_per_sample = dist_np.reshape(batch_size, -1).mean(axis=1)
            else:
                dist_per_sample = dist_np

        if reached_limit:
            stats_msg += (
                "  WARNING: Reached step limit without convergence!\n"
                "  Some samples may not have fully converged.\n"
            )

        print(stats_msg)

        detailed_msg = "\n  Detailed per-sample information:\n" + "  " + "-" * 56 + "\n"
        if dist_per_sample is not None:
            detailed_msg += f"  {'Sample':<8} {'Steps':<10} {'Final Distance':<20}\n"
            detailed_msg += "  " + "-" * 56 + "\n"
            for i in range(batch_size):
                detailed_msg += (
                    f"  {i:<8} {int(steps_np[i]):<10} {float(dist_per_sample[i]):<20.8f}\n"
                )
        else:
            detailed_msg += f"  {'Sample':<8} {'Steps':<10}\n"
            detailed_msg += "  " + "-" * 56 + "\n"
            for i in range(batch_size):
                detailed_msg += f"  {i:<8} {int(steps_np[i]):<10}\n"
        detailed_msg += "  " + "-" * 56 + "\n"
        print(detailed_msg)

        stats_filename = "distance_marching_stats.txt"
        if self.output_dir and os.path.exists(self.output_dir):
            stats_path = os.path.join(self.output_dir, stats_filename)
        else:
            stats_path = None
            for dir_path in [os.getenv("OUTPUT_DIR"), "output_dir", "."]:
                if dir_path and os.path.exists(dir_path):
                    stats_path = os.path.join(dir_path, stats_filename)
                    break
            if not stats_path:
                stats_path = stats_filename

        try:
            with open(stats_path, "a") as f:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"\n{'='*60}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(stats_msg)
                f.write(detailed_msg)
                f.write(f"{'='*60}\n")
            print(f"\nStatistics saved to: {os.path.abspath(stats_path)}\n")
            self.stats_saved = True
        except Exception as e:
            print(f"Warning: Could not save statistics to file: {e}")

    def sample(
        self,
        x_init: Tensor,
        step_size: Optional[float],
        method: str = "gd",
        criteria: str = "fixed_steps",
        threshold: float = 1e-5,
        time_grid: Tensor = torch.tensor([0.0, 1.0]),
        return_intermediates: bool = False,
        ula_schedule: Optional[Callable[[float], float]] = None,
        **model_extras,
    ) -> Union[Tensor, Sequence[Tensor]]:
        """March ``x_init`` toward the data manifold.

        Args:
            x_init: initial samples (e.g. Gaussian noise), shape [batch_size, ...].
            step_size: step size eta.
            method: 'gd', 'st', 'gd_ula' or 'st_ula'.
            criteria: stopping criteria, see :meth:`sample_loop`.
            threshold: threshold for adaptive criteria.
            time_grid: only ``len(time_grid) - 1`` (the number of steps) is used.
            return_intermediates: return the whole trajectory instead of the last state.
            ula_schedule: sigma schedule for the *_ula methods (see ``build_sigma_schedule``).
            **model_extras: forwarded to the model as ``extra`` (e.g. ``label``).
        """
        assert method in ["gd", "st", "gd_ula", "st_ula"], f"Unknown method {method}"
        if method in ["gd_ula", "st_ula"]:
            assert ula_schedule is not None, f"{method} requires ula_schedule"
        assert criteria in [
            "fixed_steps",
            "grad_norm",
            "dist_value",
            "step_norm",
        ], f"Unknown criteria {criteria}"

        time_grid = time_grid.to(x_init.device)

        def grad_and_dist_func(x, timesteps):
            return self.velocity_model(
                x=x, timesteps=timesteps, extra=model_extras, return_dist=True
            )

        base_model = get_base_model(self.velocity_model)
        # With a separate distance head no autograd is needed; the original
        # (gradient-of-scalar-field) variant needs autograd through the model.
        set_grad_enabled = not base_model.separate_t

        with torch.set_grad_enabled(set_grad_enabled):
            sol = self.sample_loop(
                func=grad_and_dist_func,
                y0=x_init,
                timesteps=time_grid,
                step_size=step_size,
                method=method,
                criteria=criteria,
                threshold=threshold,
                ula_schedule=ula_schedule,
            )

        if return_intermediates:
            return sol
        else:
            return sol[-1]
