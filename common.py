"""Small shared utilities for reproducible restoration experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    # Keep precision explicit; do not mix this with the newer fp32_precision API.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(deterministic)


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; choose --device cpu")
    return device


def capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": torch.tensor(numpy_state[1].astype(np.int64)),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value["algorithm"], value["keys"].cpu().numpy().astype(np.uint32),
                         value["position"], value["has_gauss"], value["cached_gaussian"]))
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Exact resume requires the same CUDA device count")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    # Checkpoints contain only tensors and ordinary containers, no pickled modules.
    return torch.load(Path(path), map_location=device, weights_only=True)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty result table")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def image_metrics(prediction: Tensor, reference: Tensor) -> dict[str, float]:
    """RGB [0,1], per-image PSNR and valid-window Gaussian SSIM (11, 1.5)."""
    if prediction.shape != reference.shape or prediction.ndim != 4 or prediction.shape[0] != 1:
        raise ValueError("Metrics require matching single-image NCHW tensors")
    if min(prediction.shape[-2:]) < 11:
        raise ValueError("SSIM requires images at least 11x11")
    prediction, reference = prediction.float().clamp(0, 1), reference.float()
    mse = F.mse_loss(prediction, reference).item()
    # A finite 120 dB cap keeps JSON logs portable for exact-identity test cases.
    psnr = -10 * math.log10(max(mse, 1e-12))
    axis = torch.arange(11, dtype=torch.float32, device=prediction.device) - 5
    kernel = torch.exp(-axis.square() / (2 * 1.5**2))
    kernel = kernel / kernel.sum()
    window = torch.outer(kernel, kernel).expand(prediction.shape[1], 1, 11, 11).contiguous()

    def filter_image(value: Tensor) -> Tensor:
        return F.conv2d(value, window, groups=prediction.shape[1])

    mean_x, mean_y = filter_image(prediction), filter_image(reference)
    var_x = (filter_image(prediction.square()) - mean_x.square()).clamp_min(0)
    var_y = (filter_image(reference.square()) - mean_y.square()).clamp_min(0)
    covariance = filter_image(prediction * reference) - mean_x * mean_y
    ssim = ((2 * mean_x * mean_y + 0.01**2) * (2 * covariance + 0.03**2) /
            ((mean_x.square() + mean_y.square() + 0.01**2) * (var_x + var_y + 0.03**2))).mean().item()
    return {"psnr": psnr, "ssim": ssim}


def environment_info(device: torch.device) -> dict[str, Any]:
    return {
        "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cudnn": torch.backends.cudnn.version(),
        "tf32": False, "cudnn_benchmark": False,
    }
