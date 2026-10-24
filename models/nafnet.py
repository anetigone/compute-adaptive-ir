"""Compact fixed-budget NAF-style U-Net baseline, not official NAFNet weights."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn

from .blocks import DEFAULT_BLOCK_COUNTS, make_block_groups, make_up, pad_rgb_to_multiple, validate_architecture, validate_steps


class NAFNet(nn.Module):
    """Two downsampling levels, RGB residual output and a single exit."""

    default_steps = 1
    supports_trajectory = False

    def __init__(self, width: int = 24, block_counts: Sequence[int] = DEFAULT_BLOCK_COUNTS) -> None:
        super().__init__()
        self.width = width
        self.block_counts = validate_architecture(width, block_counts)
        self.stem = nn.Conv2d(3, width, 3, padding=1)
        self.groups = make_block_groups(width, self.block_counts)
        self.down0 = nn.Conv2d(width, width * 2, 2, stride=2)
        self.down1 = nn.Conv2d(width * 2, width * 4, 2, stride=2)
        self.up1 = make_up(width * 4)
        self.up0 = make_up(width * 2)
        self.readout = nn.Conv2d(width, 3, 3, padding=1)
        nn.init.normal_(self.readout.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.readout.bias)

    def _resolve_steps(self, steps: int | None) -> int:
        steps = self.default_steps if steps is None else validate_steps(steps)
        if steps != 1:
            raise ValueError("NAFNet is a fixed-budget single-exit baseline; steps must be 1")
        return steps

    def forward(self, y: Tensor, steps: int | None = None) -> Tensor:
        self._resolve_steps(steps)
        y, (height, width) = pad_rgb_to_multiple(y)
        b0 = self.groups[0](self.stem(y))
        b1 = self.groups[1](self.down0(b0))
        b2 = self.groups[2](self.down1(b1))
        u1 = self.groups[3](self.up1(b2) + b1)
        u0 = self.groups[4](self.up0(u1) + b0)
        return (y + self.readout(u0))[..., :height, :width]

    def forward_fixed(self, y: Tensor, steps: int | None = None) -> Tensor:
        return self.forward(y, steps)

    def forward_trajectory(self, y: Tensor, steps: int | None = None) -> list[Tensor]:
        """The uniform interface returns the baseline's sole output."""
        return [self.forward(y, steps)]
