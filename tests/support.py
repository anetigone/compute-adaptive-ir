"""Small, offline PNG fixtures; no benchmark downloads are used in tests."""

from pathlib import Path

import numpy as np
from PIL import Image


def make_local_config(directory: Path) -> dict:
    source = directory / "images"
    number = 0
    for role, count in (("train", 3), ("val", 2), ("test", 2)):
        (source / role).mkdir(parents=True, exist_ok=True)
        for index in range(count):
            pixels = np.random.default_rng(900 + number).integers(0, 256, (32, 40, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(source / role / f"image_{index}.png")
            number += 1
    return {
        "seed": 193,
        "device": "cpu",
        "cpu_threads": 1,
        "output_root": str(directory / "outputs"),
        "model": {"name": "reusable_refiner", "width": 4,
                  "block_counts": [1, 1, 1, 1, 1], "steps": 2},
        "data": {"source": "folder", "local_dir": str(source),
                 "root": str(directory / "cache"), "seed": 193,
                 "train_patch": 16, "eval_crop": 16,
                 "train_sigmas": [15, 25, 50], "eval_sigmas": [15, 25, 50],
                 "noise_protocol": "clip_uint8"},
        "train": {"max_steps": 4, "warmup_steps": 1, "batch_size": 1,
                  "grad_accumulation": 2, "num_workers": 0, "lr": 2e-4,
                  "weight_decay": 0.0, "grad_clip": 1.0, "amp": False,
                  "deterministic": True, "exit_weights": [0.5, 0.5],
                  "val_every": 2, "save_every": 2, "log_every": 100,
                  "val_max_samples": 1},
        "eval": {"steps": [1, 2], "amp": False},
    }
