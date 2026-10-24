"""Deterministic, source-image separated blind RGB Gaussian denoising.

No torchvision or Hugging Face dataset scripts are used. Training randomness
depends only on (seed, epoch, index); evaluation randomness uses image identity
and sigma, so batch order and the method being evaluated cannot change noise.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset


def stable_seed(*parts: Any) -> int:
    """A process-independent seed below 2**63, safe for default_collate."""
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big") & ((1 << 63) - 1)


def data_config(config: dict) -> dict:
    return dict(config.get("data", config))


def _sigmas(values: Sequence[float], name: str) -> list[float]:
    result = [float(x) for x in values]
    if not result or any(not np.isfinite(x) or x < 0 for x in result):
        raise ValueError(f"{name} must contain finite nonnegative sigmas in 0--255 units")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} cannot contain duplicate sigmas")
    return result


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.uint8).copy()


def add_noise(clean_uint8: np.ndarray, sigma: float, rng: np.random.Generator, protocol: str) -> np.ndarray:
    clean = clean_uint8.astype(np.float32)
    noise = rng.normal(0.0, sigma, clean.shape).astype(np.float32)
    if protocol == "clip_uint8":
        # Match the explicitly selected clipped/quantized protocol: truncation,
        # rather than round-to-nearest, is the numpy astype(uint8) convention.
        return np.clip(clean + noise, 0.0, 255.0).astype(np.uint8).astype(np.float32) / 255.0
    if protocol == "float_awgn":
        return (clean + noise) / 255.0
    raise ValueError(f"Unknown noise_protocol={protocol!r}; use clip_uint8 or float_awgn")


def _crop(image: np.ndarray, size: int, rng: np.random.Generator | None) -> np.ndarray:
    if size <= 0:
        return image
    height, width = image.shape[:2]
    if height < size or width < size:
        raise ValueError(f"Image {width}x{height} is smaller than requested crop {size}; no implicit resize/padding is applied")
    top = (height - size) // 2 if rng is None else int(rng.integers(height - size + 1))
    left = (width - size) // 2 if rng is None else int(rng.integers(width - size + 1))
    return image[top:top + size, left:left + size]


class DenoisingDataset(Dataset):
    def __init__(self, records: Sequence[dict], role: str, config: dict):
        if role not in {"train", "val", "test"}:
            raise ValueError("role must be train, val or test")
        self.records = sorted((dict(r) for r in records if r["role"] == role), key=lambda r: r["image_id"])
        self.role = role
        self.config = data_config(config)
        self.seed = int(self.config.get("seed", 20261023))
        self.eval_seed = int(self.config.get("eval_seed", self.seed))
        self.train_sigmas = _sigmas(self.config.get("train_sigmas", [15, 25, 50]), "train_sigmas")
        self.eval_sigmas = _sigmas(self.config.get("eval_sigmas", [15, 25, 50]), "eval_sigmas")
        self.protocol = self.config.get("noise_protocol", "clip_uint8")
        if self.protocol not in {"clip_uint8", "float_awgn"}:
            raise ValueError("noise_protocol must be clip_uint8 or float_awgn")
        self.train_patch = int(self.config.get("train_patch", 128))
        self.eval_crop = int(self.config.get("eval_crop", 256) or 0)
        self.eval_mode = self.config.get("eval_mode", "center_crop")
        if self.eval_mode not in {"center_crop", "full"}:
            raise ValueError("eval_mode must be center_crop or full")
        if self.train_patch < 1:
            raise ValueError("train_patch must be positive")
        if self.eval_crop < 0:
            raise ValueError("eval_crop must be nonnegative; zero means full image")
        crop = self.train_patch if role == "train" else self.eval_crop if self.eval_mode == "center_crop" else 0
        for record in self.records:
            if "width" in record and "height" in record and min(record["width"], record["height"]) < crop:
                raise ValueError(f"{record['image_id']} is smaller than requested {role} crop {crop}; no implicit resizing is applied")
        self.repeats = int(self.config.get("samples_per_image", 1)) if role == "train" else 1
        if self.repeats < 1:
            raise ValueError("samples_per_image must be positive")
        # Shared CPU storage also propagates set_epoch to persistent workers.
        self._epoch = torch.zeros(1, dtype=torch.int64).share_memory_()

    @property
    def epoch(self) -> int:
        return int(self._epoch.item())

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        self._epoch.fill_(epoch)

    def __len__(self) -> int:
        return len(self.records) * (self.repeats if self.role == "train" else len(self.eval_sigmas))

    def __getitem__(self, index: int) -> dict:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if self.role == "train":
            record = self.records[index % len(self.records)]
            sample_seed = stable_seed(self.seed, self.epoch, index)
            rng = np.random.default_rng(sample_seed)
            rgb = _crop(load_rgb(record["path"]), self.train_patch, rng)
            if bool(rng.integers(2)):
                rgb = rgb[:, ::-1]
            rgb = np.rot90(rgb, int(rng.integers(4)), axes=(0, 1))
            sigma = self.train_sigmas[int(rng.integers(len(self.train_sigmas)))]
            noise_seed = stable_seed(sample_seed, "noise")
        else:
            record = self.records[index // len(self.eval_sigmas)]
            sigma = self.eval_sigmas[index % len(self.eval_sigmas)]
            rgb = load_rgb(record["path"])
            if self.eval_mode == "center_crop":
                rgb = _crop(rgb, self.eval_crop, None)
            noise_seed = stable_seed(self.eval_seed, record["image_id"], sigma, "evaluation-noise")
            sample_seed = noise_seed
        noisy = add_noise(rgb, sigma, np.random.default_rng(noise_seed), self.protocol)
        clean = np.ascontiguousarray(rgb.astype(np.float32) / 255.0)
        return {
            "noisy": torch.from_numpy(np.ascontiguousarray(noisy.transpose(2, 0, 1))),
            "clean": torch.from_numpy(np.ascontiguousarray(clean.transpose(2, 0, 1))),
            "image_id": record["image_id"],
            "source_id": record["source_id"],
            "role": self.role,
            "sigma": sigma,
            "noise_seed": noise_seed,
            "sample_seed": sample_seed,
            "noise_protocol": self.protocol,
        }


def build_datasets(config: dict, manifest_path: Path) -> dict[str, DenoisingDataset]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Older runs stored records; new preparations contain only three file lists.
    records = manifest.get("records")
    if records is None:
        records = [{"path": path, "image_id": path, "source_id": path, "role": role}
                   for role in ("train", "val", "test") for path in manifest[role]]
    for record in records:
        path = Path(record["path"])
        if not path.is_absolute():
            path = manifest_path.parent / path
        record["path"] = str(path.resolve())
        if not path.is_file():
            raise FileNotFoundError(path)
    merged = dict(manifest.get("config", {}))
    merged.update(data_config(config))
    return {role: DenoisingDataset(records, role, merged) for role in ("train", "val", "test")}
