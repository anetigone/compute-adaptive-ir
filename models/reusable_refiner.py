"""Cached-observation, shared-update image restoration research model.

The gain head is an untrained future extension and is frozen by default.
Restoration forwards never call it. A restoration checkpoint alone does
not provide a usable dynamic stopping policy.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .blocks import DEFAULT_BLOCK_COUNTS, make_block_groups, make_up, pad_rgb_to_multiple, validate_architecture, validate_steps


CostTable = Mapping[int, Sequence[float]]
CostProvider = Callable[[int, int, int, int], Sequence[float]]


class SharedUpdate(nn.Module):
    def __init__(self, width: int, block_counts: Sequence[int]) -> None:
        super().__init__()
        self.groups = make_block_groups(width, block_counts)
        self.down0 = nn.Conv2d(width, width * 2, 2, stride=2)
        self.down1 = nn.Conv2d(width * 2, width * 4, 2, stride=2)
        self.up1 = make_up(width * 4)
        self.up0 = make_up(width * 2)
        self.delta = nn.Conv2d(width, width, 3, padding=1)
        self.alpha_logit = nn.Parameter(torch.tensor(math.log(0.1 / 0.9)))

    def forward(self, h: Tensor, features: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        f0, f1, f2 = features
        b0 = self.groups[0](h + f0)
        b1 = self.groups[1](self.down0(b0) + f1)
        b2 = self.groups[2](self.down1(b1) + f2)
        u1 = self.groups[3](self.up1(b2) + b1)
        u0 = self.groups[4](self.up0(u1) + b0)
        return h + self.alpha_logit.sigmoid() * self.delta(u0)


@dataclass
class DynamicResult:
    image: Tensor
    steps: int


class ReusableRefiner(nn.Module):
    supports_trajectory = True

    def __init__(self, width: int = 24, block_counts: Sequence[int] = DEFAULT_BLOCK_COUNTS, train_steps: int = 4) -> None:
        super().__init__()
        self.width = width
        self.block_counts = validate_architecture(width, block_counts)
        self.train_steps = validate_steps(train_steps)
        self.default_steps = self.train_steps
        self.stem0 = nn.Conv2d(3, width, 3, padding=1)
        self.stem1 = nn.Conv2d(width, width * 2, 2, stride=2)
        self.stem2 = nn.Conv2d(width * 2, width * 4, 2, stride=2)
        self.update = SharedUpdate(width, self.block_counts)
        self.readout = nn.Conv2d(width, 3, 3, padding=1)
        nn.init.normal_(self.readout.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.readout.bias)
        self.gain_head = nn.Sequential(nn.Linear(width * 4 + 3, 64), nn.GELU(), nn.Linear(64, 3))
        self.gain_head.requires_grad_(False)

    def _resolve_steps(self, steps: int | None) -> int:
        return validate_steps(self.default_steps if steps is None else steps)

    def _prepare(self, y: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor], tuple[int, int]]:
        y, shape = pad_rgb_to_multiple(y)
        f0 = self.stem0(y)
        f1 = self.stem1(f0)
        f2 = self.stem2(f1)
        return y, (f0, f1, f2), shape

    def _decode(self, y: Tensor, h: Tensor, shape: tuple[int, int]) -> Tensor:
        return (y + self.readout(h))[..., :shape[0], :shape[1]]

    def forward_fixed(self, y: Tensor, steps: int | None = None) -> Tensor:
        """Run shared updates, reconstructing only the final requested exit."""
        steps = self._resolve_steps(steps)
        y, features, shape = self._prepare(y)
        h = features[0]
        for _ in range(steps):
            h = self.update(h, features)
        return self._decode(y, h, shape)

    def forward(self, y: Tensor, steps: int | None = None) -> Tensor:
        return self.forward_fixed(y, steps)

    def forward_trajectory(self, y: Tensor, steps: int | None = None) -> list[Tensor]:
        """Keep the complete graph and return all exits without output clipping."""
        steps = self._resolve_steps(steps)
        y, features, shape = self._prepare(y)
        h = features[0]
        outputs = []
        for _ in range(steps):
            h = self.update(h, features)
            outputs.append(self._decode(y, h, shape))
        return outputs

    def policy_features(self, h: Tensor, input_gap: Tensor, previous_gap: Tensor, shape: tuple[int, int], step: int) -> tuple[Tensor, Tensor]:
        h_float = h.float()
        gap = h_float.mean(dim=(-2, -1))
        rms = h_float.square().mean(dim=(-2, -1)).clamp_min(1e-12).sqrt()
        scalars = gap.new_tensor([math.log(shape[0]), math.log(shape[1]), step / self.train_steps]).expand(h.shape[0], -1)
        return torch.cat([gap, rms, input_gap, gap - previous_gap, scalars], dim=1), gap

    @staticmethod
    def _costs(provider: CostTable | CostProvider, step: int, available: int, shape: tuple[int, int]) -> list[float]:
        if callable(provider):
            row = provider(step, available, shape[0], shape[1])
        else:
            if step not in provider:
                raise ValueError(f"missing incremental cost row for current step {step}")
            row = provider[step]
        if len(row) < available:
            raise ValueError(f"step {step} requires costs for {available} future horizons")
        costs = [float(c) for c in row[:available]]
        if any(not math.isfinite(c) or c <= 0 for c in costs):
            raise ValueError("incremental costs must be positive finite milliseconds")
        return costs

    @torch.no_grad()
    def forward_dynamic(self, y: Tensor, max_steps: int | None = None, lambda_db_per_ms: float = 0.0, extra_cost_ms: CostTable | CostProvider | None = None) -> DynamicResult:
        """Experimental real early exit; head training/calibration is not supplied.

        ``extra_cost_ms`` is mandatory when decisions are needed. Pass a table
        ``{current_step: [cost_to_t_plus_1, cost_to_t_plus_2, ...]}``, already
        calibrated for this input size/device/precision, or a callable
        ``(step, available_horizons, height, width) -> costs``. Costs include
        future updates and decisions, excluding the already-paid current
        decision. The head predicts at most three horizons and replans after
        one update. Positive/negative infinity lambda values force first/last
        exits for structural checks; finite values need a trained gain head.
        """
        max_steps = self._resolve_steps(max_steps)
        if max_steps > self.train_steps:
            raise ValueError("dynamic max_steps must stay within train_steps; fixed/trajectory allow extrapolation")
        lambda_db_per_ms = float(lambda_db_per_ms)
        if math.isnan(lambda_db_per_ms):
            raise ValueError("lambda_db_per_ms cannot be NaN")
        if max_steps > 1 and extra_cost_ms is None:
            raise ValueError("dynamic decisions require a measured step/horizon cost table or provider")
        if extra_cost_ms is not None and not callable(extra_cost_ms) and not isinstance(extra_cost_ms, Mapping):
            raise TypeError("extra_cost_ms must be a step-indexed mapping or callable")
        y, features, shape = self._prepare(y)
        if y.shape[0] != 1:
            raise ValueError("dynamic execution currently supports batch=1")
        h = features[0]
        input_gap = h.float().mean(dim=(-2, -1))
        previous_gap = input_gap
        for step in range(1, max_steps + 1):
            h = self.update(h, features)
            if step == max_steps:
                break
            vector, current_gap = self.policy_features(h, input_gap, previous_gap, shape, step)
            gains = self.gain_head(vector.to(dtype=self.gain_head[0].weight.dtype)).float()[0]
            available = min(3, max_steps - step)
            costs = gains.new_tensor(self._costs(extra_cost_ms, step, available, shape))
            # Reading this scalar synchronizes GPU->CPU; include it in timing.
            should_continue = bool(((gains[:available] - lambda_db_per_ms * costs).max() > 0).item())
            previous_gap = current_gap
            if not should_continue:
                break
        return DynamicResult(self._decode(y, h, shape), step)
