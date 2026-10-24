"""Untrained architecture probe for shared, budget-adaptive image restoration.

This is a native-PyTorch design scaffold, not a trained restoration method.
NAF-style blocks follow https://arxiv.org/abs/2204.04676; the observation
cache, recurrent cell and gain head are experimental design choices.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ChannelLayerNorm(nn.Module):
    """LayerNorm over channels independently at each spatial position."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(
            x.permute(0, 2, 3, 1), (x.shape[1],), self.weight, self.bias, self.eps
        ).permute(0, 3, 1, 2)


class SimpleGate(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        left, right = x.chunk(2, dim=1)
        return left * right


class NAFBlock(nn.Module):
    """NAF-style DW/SCA branch and gated FFN, both with zero-init scales."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = ChannelLayerNorm(channels)
        self.expand1 = nn.Conv2d(channels, channels * 2, 1)
        self.depthwise = nn.Conv2d(channels * 2, channels * 2, 3, padding=1, groups=channels * 2)
        self.gate = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1))
        self.project1 = nn.Conv2d(channels, channels, 1)
        self.norm2 = ChannelLayerNorm(channels)
        self.expand2 = nn.Conv2d(channels, channels * 2, 1)
        self.project2 = nn.Conv2d(channels, channels, 1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        branch = self.gate(self.depthwise(self.expand1(self.norm1(x))))
        branch = self.project1(branch * self.sca(branch))
        middle = x + self.beta * branch
        return middle + self.gamma * self.project2(self.gate(self.expand2(self.norm2(middle))))


class SharedUpdate(nn.Module):
    def __init__(self, width: int, block_counts: Sequence[int]) -> None:
        super().__init__()
        if len(block_counts) != 5 or any(n < 1 for n in block_counts):
            raise ValueError("block_counts must contain five positive counts")
        self.groups = nn.ModuleList(
            nn.Sequential(*(NAFBlock(channels) for _ in range(count)))
            for channels, count in zip([width, width * 2, width * 4, width * 2, width], block_counts)
        )
        self.down0 = nn.Conv2d(width, width * 2, 2, stride=2)
        self.down1 = nn.Conv2d(width * 2, width * 4, 2, stride=2)
        self.up1 = nn.Sequential(nn.Conv2d(width * 4, width * 8, 1, bias=False), nn.PixelShuffle(2))
        self.up0 = nn.Sequential(nn.Conv2d(width * 2, width * 4, 1, bias=False), nn.PixelShuffle(2))
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
    def __init__(self, width: int = 24, block_counts: Sequence[int] = (1, 1, 2, 1, 1), train_steps: int = 4) -> None:
        super().__init__()
        self.width = width
        self.train_steps = train_steps
        self.stem0 = nn.Conv2d(3, width, 3, padding=1)
        self.stem1 = nn.Conv2d(width, width * 2, 2, stride=2)
        self.stem2 = nn.Conv2d(width * 2, width * 4, 2, stride=2)
        self.update = SharedUpdate(width, block_counts)
        self.readout = nn.Conv2d(width, 3, 3, padding=1)
        nn.init.normal_(self.readout.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.readout.bias)
        self.gain_head = nn.Sequential(nn.Linear(width * 4 + 3, 64), nn.GELU(), nn.Linear(64, 3))

    @staticmethod
    def _check_steps(steps: int) -> None:
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")

    def _prepare(self, y: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor], tuple[int, int]]:
        if y.ndim != 4 or y.shape[1] != 3 or min(y.shape[-2:]) < 1:
            raise ValueError("expected [N,3,H,W] with positive spatial dimensions")
        height, width = y.shape[-2:]
        pad_h, pad_w = (-height) % 4, (-width) % 4
        if pad_h or pad_w:
            mode = "reflect" if height > pad_h and width > pad_w else "replicate"
            y = F.pad(y, (0, pad_w, 0, pad_h), mode=mode)
        f0 = self.stem0(y)
        f1 = self.stem1(f0)
        f2 = self.stem2(f1)
        return y, (f0, f1, f2), (height, width)

    def _decode(self, y: Tensor, h: Tensor, shape: tuple[int, int]) -> Tensor:
        return (y + self.readout(h))[..., : shape[0], : shape[1]]

    def forward_fixed(self, y: Tensor, steps: int = 4) -> Tensor:
        """Only reconstruct at the requested final step; no policy overhead."""
        self._check_steps(steps)
        y, features, shape = self._prepare(y)
        h = features[0]
        for _ in range(steps):
            h = self.update(h, features)
        return self._decode(y, h, shape)

    def forward(self, y: Tensor, steps: int = 4) -> Tensor:
        return self.forward_fixed(y, steps)

    def forward_trajectory(self, y: Tensor, steps: int = 4) -> list[Tensor]:
        """Training/offline diagnosis path: reconstruct every step without clamping."""
        self._check_steps(steps)
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

    def forward_dynamic(self, y: Tensor, max_steps: int = 4, lambda_db_per_ms: float = 0.0, extra_cost_ms: Sequence[float] = (1.0, 2.0, 3.0)) -> DynamicResult:
        """True early exit, with signed 1/2/3-step gain predictions.

        Continue while an available terminal horizon has positive gain minus
        lambda*additional-cost. Replan after one update. Cost values must be
        calibrated for the actual device, shape and complete policy path.
        The default costs and RANDOM head are only for API/smoke testing.
        lambda=+inf forces first exit; lambda=-inf forces the last exit.
        """
        self._check_steps(max_steps)
        if max_steps > self.train_steps:
            raise ValueError("dynamic max_steps must stay within the trained horizon; use fixed/trajectory for extrapolation")
        if y.shape[0] != 1:
            raise ValueError("dynamic execution currently supports batch=1")
        if len(extra_cost_ms) != 3 or any(not math.isfinite(c) or c <= 0 for c in extra_cost_ms):
            raise ValueError("provide three positive finite incremental costs")
        y, features, shape = self._prepare(y)
        h = features[0]
        input_gap = h.float().mean(dim=(-2, -1))
        previous_gap = input_gap
        for step in range(1, max_steps + 1):
            h = self.update(h, features)
            if step == max_steps:
                break
            policy_input, current_gap = self.policy_features(h, input_gap, previous_gap, shape, step)
            gains = self.gain_head(policy_input).float()[0]
            available = min(3, max_steps - step)
            costs = gains.new_tensor(extra_cost_ms[:available])
            net_gain = gains[:available] - lambda_db_per_ms * costs
            # .item() synchronizes GPU->CPU: this is deliberately inside timing.
            should_continue = bool((net_gain.max() > 0).item())
            # No gain logging conversion here: that would add another sync/copy.
            previous_gap = current_gap
            if not should_continue:
                break
        return DynamicResult(self._decode(y, h, shape), step)
