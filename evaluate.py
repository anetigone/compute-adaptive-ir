"""Per-image RGB restoration quality and optional real fixed-exit timing.

This evaluates restoration exits only. It does not train or evaluate a
learned dynamic stopping policy. Default timing counts are short engineering
checks, not a publication-quality latency protocol.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import platform
import statistics
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from common import (environment_info, file_sha256, image_metrics, load_checkpoint,
                    resolve_device, seed_everything, write_csv, write_json)
from config import load_config, project_path
from data import build_datasets
from models import build_model


TIMING_BOUNDARY = (
    "input already on device; output remains on device; synchronized CPU wall-clock; "
    "includes encoding, exactly the requested shared updates, final readout and Python "
    "dispatch; excludes data loading, H2D/D2H transfers, quality metrics and file I/O"
)


def _sample_limit(max_samples: int) -> int:
    if isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples < 0:
        raise ValueError("max_samples must be a nonnegative integer; zero means all samples")
    return max_samples


def _loader(dataset) -> DataLoader:
    # Even a non-shuffled DataLoader consumes a base seed when iterated.
    # A private generator keeps validation from changing training/resume RNG.
    generator = torch.Generator().manual_seed(20261023)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, generator=generator)


def _finite(value: Tensor, name: str) -> None:
    if not torch.isfinite(value).all().item():
        raise FloatingPointError(f"Nonfinite {name} during evaluation")


def _metrics(prediction: Tensor, clean: Tensor) -> dict[str, float]:
    _finite(prediction, "model output")
    values = image_metrics(prediction, clean)
    if any(not math.isfinite(value) for value in values.values()):
        raise FloatingPointError("Nonfinite image quality metric")
    return values


def validate_model(model: nn.Module, dataset, device: torch.device,
                   amp_enabled: bool, max_samples: int = 0) -> dict[str, Any]:
    """Mean per-image quality at the default final exit, without timing.

    Loader construction does not advance global RNG, and the incoming model
    training/evaluation mode is restored even if validation raises an error.
    """
    max_samples = _sample_limit(max_samples)
    device = torch.device(device)
    amp_enabled = bool(amp_enabled) and device.type == "cuda"
    training = model.training
    psnr, ssim = [], []
    model.eval()
    try:
        with torch.inference_mode():
            for batch in itertools.islice(_loader(dataset), max_samples or None):
                noisy, clean = batch["noisy"].to(device), batch["clean"].to(device)
                _finite(noisy, "input")
                _finite(clean, "reference")
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    prediction = model(noisy)
                values = _metrics(prediction, clean)
                psnr.append(values["psnr"])
                ssim.append(values["ssim"])
    finally:
        model.train(training)
    if not psnr:
        raise ValueError("Validation dataset is empty")
    return {"psnr": statistics.mean(psnr), "ssim": statistics.mean(ssim), "samples": len(psnr)}


def _exits(model: nn.Module, configured: Sequence[int] | None) -> list[int]:
    if configured is None:
        configured = [1, 2, 3, 4, 6, 8] if model.supports_trajectory else [1]
    if isinstance(configured, (str, bytes)) or not isinstance(configured, Sequence):
        raise TypeError("evaluation exits must be a sequence of positive integers")
    if not configured or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in configured):
        raise ValueError("evaluation exits must contain positive integers")
    values = sorted(set(configured))
    if not model.supports_trajectory and values != [1]:
        raise ValueError("The fixed-budget baseline only supports evaluation exit 1")
    return values


def _single(value: Any) -> Any:
    """Unpack default_collate's single-sample metadata."""
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError("Expected single-sample scalar metadata")
        return value.item()
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower, upper = math.floor(index), math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _latency_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"latency_mean_ms": None, "latency_p50_ms": None, "latency_p95_ms": None,
                "latency_measurements": 0}
    return {"latency_mean_ms": statistics.mean(values),
            "latency_p50_ms": statistics.median(values),
            "latency_p95_ms": _percentile(values, 0.95), "latency_measurements": len(values)}


def _benchmark_exit(model: nn.Module, noisy: Tensor, steps: int, device: torch.device,
                    amp_enabled: bool, warmup: int, repeats: int) -> list[float]:
    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def forward() -> Tensor:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            return model.forward_fixed(noisy, steps=steps)

    with torch.inference_mode():
        for _ in range(warmup):
            output = forward()
            _finite(output, "timing warmup output")
            del output
        synchronize()
        samples = []
        for _ in range(repeats):
            synchronize()
            start = time.perf_counter()
            output = forward()
            synchronize()
            elapsed = (time.perf_counter() - start) * 1000.0
            # Quality/finite checks deliberately remain outside the timer.
            _finite(output, "timed output")
            if not math.isfinite(elapsed) or elapsed < 0:
                raise FloatingPointError("Nonfinite/negative latency")
            samples.append(elapsed)
            del output
    return samples


def _aggregate(rows: list[dict[str, Any]], timings: dict, by_sigma: bool) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for row in rows:
        key = (row["sigma"], row["steps"]) if by_sigma else row["steps"]
        groups[key].append(row)
    result = []
    for key, members in sorted(groups.items()):
        record = {"sigma": key[0], "steps": key[1]} if by_sigma else {"steps": key}
        record.update({"samples": len(members),
                       "psnr": statistics.mean(row["psnr"] for row in members),
                       "ssim": statistics.mean(row["ssim"] for row in members),
                       "input_psnr": statistics.mean(row["input_psnr"] for row in members)})
        samples = timings.get(key, [])
        record.update(_latency_summary(samples))
        result.append(record)
    return result


def evaluate(config: dict, checkpoint_path: Path, manifest_path: Path, output_dir: Path,
             split: str = "test", exits: Sequence[int] | None = None,
             benchmark: bool = False, max_samples: int = 0) -> dict[str, Any]:
    """Evaluate checkpoint architecture and per-image quality on one manifest split."""
    if split not in {"val", "test"}:
        raise ValueError("Restoration evaluation requires the val or test role")
    max_samples = _sample_limit(max_samples)
    checkpoint_path, manifest_path = Path(checkpoint_path).resolve(), Path(manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    checkpoint = load_checkpoint(checkpoint_path)
    manifest_hash = file_sha256(manifest_path)
    if checkpoint.get("manifest_sha256") != manifest_hash:
        raise ValueError("Evaluation manifest hash differs from the training checkpoint")
    try:
        model_config = checkpoint["config"]["model"]
        state = checkpoint["model"]
        checkpoint_step = checkpoint["step"]
    except KeyError as error:
        raise ValueError(f"Checkpoint is missing required restoration training metadata: {error}") from error
    settings = config.get("eval", {})
    device = resolve_device(config.get("device", "auto"))
    torch.set_num_threads(int(config.get("cpu_threads", 4)))
    seed_everything(int(config.get("seed", 20261023)), bool(config.get("train", {}).get("deterministic", False)))
    model = build_model(model_config).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    selected_exits = _exits(model, exits if exits is not None else settings.get("exits"))
    amp_enabled = bool(settings.get("amp", config.get("train", {}).get("amp", True))) and device.type == "cuda"
    warmup, repeats = int(settings.get("warmup", 5)), int(settings.get("repeats", 3))
    if warmup < 0 or repeats < 1:
        raise ValueError("eval.warmup must be nonnegative and eval.repeats must be positive")
    datasets = build_datasets(config, manifest_path)
    if split not in datasets:
        raise ValueError(f"Unknown dataset split {split!r}; available splits: {sorted(datasets)}")
    dataset = datasets[split]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    timings_by_group, timings_by_exit = defaultdict(list), defaultdict(list)
    processed = 0
    with torch.inference_mode():
        for batch in itertools.islice(_loader(dataset), max_samples or None):
            noisy, clean = batch["noisy"].to(device), batch["clean"].to(device)
            _finite(noisy, "input")
            _finite(clean, "reference")
            input_psnr = _metrics(noisy, clean)["psnr"]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                outputs = model.forward_trajectory(noisy, steps=max(selected_exits))
            sample_rows = []
            for steps in selected_exits:
                values = _metrics(outputs[steps - 1], clean)
                row = {
                    "image_id": str(_single(batch["image_id"])),
                    "source_id": str(_single(batch["source_id"])),
                    "split": split, "sigma": float(_single(batch["sigma"])),
                    "noise_seed": int(_single(batch["noise_seed"])), "steps": steps,
                    "psnr": values["psnr"], "ssim": values["ssim"], "input_psnr": input_psnr,
                    "height": int(noisy.shape[-2]), "width": int(noisy.shape[-1]),
                    "noise_protocol": str(_single(batch.get("noise_protocol", getattr(dataset, "protocol", "unknown")))),
                    "model": model_config.get("name", "nafnet"),
                    "checkpoint": str(checkpoint_path), "checkpoint_step": int(checkpoint_step),
                    "latency_mean_ms": None, "latency_p50_ms": None, "latency_p95_ms": None,
                    "latency_measurements": 0, "timing_warmup": warmup if benchmark else 0,
                    "timing_repeats": repeats if benchmark else 0,
                    "timing_precision": "amp_float16" if amp_enabled else "float32",
                    "timing_device": str(device), "timing_boundary": TIMING_BOUNDARY if benchmark else "not_measured",
                }
                sample_rows.append(row)
            # Release the quality trajectory before timing actual fixed paths.
            del outputs
            if benchmark:
                for row in sample_rows:
                    samples = _benchmark_exit(model, noisy, row["steps"], device, amp_enabled, warmup, repeats)
                    row.update(_latency_summary(samples))
                    timings_by_group[(row["sigma"], row["steps"])].extend(samples)
                    timings_by_exit[row["steps"]].extend(samples)
            rows.extend(sample_rows)
            processed += 1
    if not rows:
        raise ValueError(f"Dataset split {split!r} is empty")
    protocol = {
        "color": "RGB", "noise_protocol": getattr(dataset, "protocol", config.get("data", {}).get("noise_protocol")),
        "spatial_mode": getattr(dataset, "eval_mode", None), "eval_crop": getattr(dataset, "eval_crop", None),
        "implicit_resize": False, "output_clip": [0.0, 1.0], "shave": 0,
        "internal_padding": "right/bottom to multiple of 4; reflection with tiny-image replicate fallback; crop back",
        "psnr": "FP32 RGB [0,1], per-image MSE; image PSNR arithmetic mean; finite 120 dB cap",
        "ssim": "FP32 RGB Gaussian 11x11, sigma=1.5, valid window; channel/spatial mean",
        "input_psnr": "same clipped prediction metric applied to noisy input",
        "sample_weighting": "equal weight per image-noise realization; no pixel-area weighting",
    }
    summary = {
        "format_version": 1, "checkpoint": str(checkpoint_path), "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_step": int(checkpoint_step), "model": model_config,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "gain_head_trained": False, "scope": "restoration exits only; no learned dynamic policy evaluation",
        "manifest": str(manifest_path), "manifest_sha256": manifest_hash,
        "manifest_hash_verified": True, "manifest_schema_version": manifest.get("schema_version"),
        "manifest_config": manifest.get("config", {}), "effective_data_config": getattr(dataset, "config", config.get("data", {})),
        "checkpoint_environment": checkpoint.get("environment"),
        "environment": {**environment_info(device), "python": sys.version, "platform": platform.platform()},
        "precision": "amp_float16" if amp_enabled else "float32", "split": split,
        "samples": processed, "available_samples": len(dataset), "max_samples": max_samples,
        "rows": len(rows), "exits": selected_exits, "training_default_exit": model.default_steps,
        "budget_extrapolation_exits": [step for step in selected_exits if step > model.default_steps],
        "preprocessing_and_metrics": protocol,
        "benchmark": {"enabled": bool(benchmark), "warmup": warmup if benchmark else 0,
                      "repeats_per_image_exit": repeats if benchmark else 0,
                      "timing_boundary": TIMING_BOUNDARY if benchmark else "not_measured",
                      "classification": "short engineering timing" if benchmark else "not_measured",
                      "limitations": "fixed ascending exit order; short repetitions; uncontrolled external load/thermal state; not a publication latency benchmark" if benchmark else None,
                      "percentile_aggregation": "percentiles from pooled raw repetitions; not averages of per-image percentiles"},
        "by_sigma_and_exit": _aggregate(rows, timings_by_group, by_sigma=True),
        "by_exit": _aggregate(rows, timings_by_exit, by_sigma=False),
        "per_image_csv": str(output_dir / "per_image.csv"),
    }
    write_csv(output_dir / "per_image.csv", rows)
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Defaults to the configuration stored in the checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--exits", type=int, nargs="+")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()
    config = load_config(args.config) if args.config else load_checkpoint(project_path(args.checkpoint))["config"]
    summary = evaluate(config, project_path(args.checkpoint), project_path(args.manifest), project_path(args.output),
                       args.split, args.exits, args.benchmark, args.max_samples)
    print(json.dumps({"samples": summary["samples"], "exits": summary["exits"],
                      "by_exit": summary["by_exit"], "summary": str(project_path(args.output) / "summary.json")},
                     ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
