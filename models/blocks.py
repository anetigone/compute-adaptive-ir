"""Native PyTorch NAF-style blocks shared by the experimental models.

These building blocks follow NAFNet (https://arxiv.org/abs/2204.04676).
The models in this package are compact research configurations, not a
reproduction of the official NAFNet architecture or published weights.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


DEFAULT_BLOCK_COUNTS = (1, 1, 2, 1, 1)


def validate_architecture(width: int, block_counts: Sequence[int]) -> tuple[int, ...]:
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise ValueError("width must be a positive integer")
    if len(block_counts) != 5 or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in block_counts):
        raise ValueError("block_counts must contain five positive integers")
    return tuple(block_counts)


def validate_steps(steps: int) -> int:
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    return steps


def pad_rgb_to_multiple(y: Tensor, multiple: int = 4) -> tuple[Tensor, tuple[int, int]]:
    if y.ndim != 4 or y.shape[0] < 1 or y.shape[1] != 3 or min(y.shape[-2:]) < 1:
        raise ValueError("expected an RGB tensor [N,3,H,W] with positive N, H and W")
    if not y.is_floating_point():
        raise TypeError("model inputs must be floating point RGB tensors")
    height, width = y.shape[-2:]
    pad_h, pad_w = (-height) % multiple, (-width) % multiple
    if pad_h or pad_w:
        mode = "reflect" if height > pad_h and width > pad_w else "replicate"
        y = F.pad(y, (0, pad_w, 0, pad_h), mode=mode)
    return y, (height, width)


class ChannelLayerNorm(nn.Module):
    """Normalize channels independently at each pixel, with native autograd."""

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
    """Depthwise/SCA branch and gated FFN, with zero-initialized scales."""

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
        middle = x + self.beta * self.project1(branch * self.sca(branch))
        return middle + self.gamma * self.project2(self.gate(self.expand2(self.norm2(middle))))


def make_block_groups(width: int, block_counts: Sequence[int]) -> nn.ModuleList:
    block_counts = validate_architecture(width, block_counts)
    return nn.ModuleList(
        nn.Sequential(*(NAFBlock(channels) for _ in range(count)))
        for channels, count in zip((width, width * 2, width * 4, width * 2, width), block_counts)
    )


def make_up(channels: int) -> nn.Sequential:
    """Upsample spatially by two and halve the channel count."""
    return nn.Sequential(nn.Conv2d(channels, channels * 2, 1, bias=False), nn.PixelShuffle(2))
