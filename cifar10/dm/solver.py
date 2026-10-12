# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
# Distance Marching samplers: sphere tracing and gradient descent on the
# learned distance field, run for a fixed number of steps.

from typing import Callable, Union

import torch
from torch import nn, Tensor


class DistanceMarchingSolver(nn.Module):
    """Deterministic samplers driven by a Distance Marching model.

    The model receives ``(x, timesteps, extra, return_dist=True)`` and returns
    ``(grad, dist)``: the direction pointing away from the data manifold and the
    predicted distance to it. ``timesteps`` is a placeholder (the model has no
    time input).

    Args:
        velocity_model: the Distance Marching model (possibly EMA/DDP wrapped).
    """

    def __init__(self, velocity_model: Union[nn.Module, Callable]):
        super().__init__()
        self.velocity_model = velocity_model

    @torch.no_grad()
    def sample(
        self,
        x_init: Tensor,
        step_size: float,
        method: str = "st",
        num_steps: int = 250,
        return_intermediates: bool = False,
        **model_extras,
    ) -> Tensor:
        """March ``x_init`` toward the data manifold.

        Args:
            x_init: initial samples (e.g. Gaussian noise), shape [batch_size, ...].
            step_size: step size eta.
            method: 'st' (x <- x - eta * dist * grad) or 'gd' (x <- x - eta * grad).
            num_steps: number of marching steps.
            return_intermediates: return the states after every step, shape
                [num_steps + 1, batch_size, ...] (index 0 is ``x_init``),
                instead of the final state.
            **model_extras: forwarded to the model as ``extra``.

        Returns:
            The final state, or the whole trajectory if ``return_intermediates``.
        """
        assert method in ["st", "gd"], f"Unknown method {method}"
        assert step_size is not None, "step_size must be specified for distance marching"

        batch_size = x_init.shape[0]
        # Placeholder time value: required by the model signature but unused.
        placeholder_t = torch.zeros((batch_size,), device=x_init.device)

        x_current = x_init.clone()
        trajectory = [x_init] if return_intermediates else None

        for _ in range(num_steps):
            grad, dist = self.velocity_model(
                x=x_current, timesteps=placeholder_t, extra=model_extras, return_dist=True
            )
            if method == "gd":
                update = -step_size * grad
            else:  # "st"
                dist_expanded = dist.reshape(batch_size, *([1] * (grad.ndim - 1)))
                update = -step_size * grad * dist_expanded
            x_current = x_current + update
            if return_intermediates:
                trajectory.append(x_current)

        if return_intermediates:
            return torch.stack(trajectory)
        return x_current
