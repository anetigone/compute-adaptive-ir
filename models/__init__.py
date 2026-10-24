"""Model factory for the fixed-budget and shared-refinement experiments."""

from __future__ import annotations

from torch import nn

from .blocks import DEFAULT_BLOCK_COUNTS
from .nafnet import NAFNet
from .reusable_refiner import ReusableRefiner

__all__ = ["NAFNet", "ReusableRefiner", "build_model"]


def build_model(config: dict) -> nn.Module:
    """Build from ``name``, ``width``, ``block_counts`` and optional ``steps``.

    ``steps`` aliases ``train_steps`` for the reusable model. For NAFNet it
    must be one. The factory rejects unknown options to surface config typos.
    """
    if not isinstance(config, dict):
        raise TypeError("model config must be a dictionary")
    allowed = {"name", "width", "block_counts", "steps", "train_steps"}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"unknown model configuration keys: {sorted(unknown)}")
    name = config.get("name", "nafnet")
    if name not in {"nafnet", "reusable_refiner"}:
        raise ValueError(f"unknown model name {name!r}; choose 'nafnet' or 'reusable_refiner'")
    kwargs = {"width": config.get("width", 24), "block_counts": config.get("block_counts", DEFAULT_BLOCK_COUNTS)}
    if "steps" in config and "train_steps" in config and config["steps"] != config["train_steps"]:
        raise ValueError("steps and train_steps conflict")
    steps = config.get("steps", config.get("train_steps", 1 if name == "nafnet" else 4))
    if name == "nafnet":
        model = NAFNet(**kwargs)
        model._resolve_steps(steps)
        return model
    return ReusableRefiner(**kwargs, train_steps=steps)
