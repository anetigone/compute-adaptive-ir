"""Prepare data, train, evaluate, or compare the initial restoration models."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from common import load_checkpoint, write_json
from config import PROJECT_ROOT, load_config, project_path


def _configuration(args, saved: dict[str, Any] | None = None) -> dict[str, Any]:
    config = load_config(args.config) if args.config else copy.deepcopy(saved or load_config("configs/denoise.yaml"))
    if args.device:
        config["device"] = args.device
    return config


def _model_config(config: dict[str, Any], name: str) -> dict[str, Any]:
    config = copy.deepcopy(config)
    previous_name = config["model"]["name"]
    if "steps" in config["model"] and "train_steps" in config["model"] and config["model"]["steps"] != config["model"]["train_steps"]:
        raise ValueError("model.steps and model.train_steps conflict")
    if previous_name == name:
        return config
    config["model"]["name"] = name
    config["model"].pop("train_steps", None)
    if name == "nafnet":
        config["model"]["steps"] = 1
    else:
        config["model"]["steps"] = 4
    return config


def _prepare(config, args) -> Path:
    from data import prepare_data
    if getattr(args, "max_images", None) is not None:
        config["data"]["max_images"] = args.max_images
    if getattr(args, "local_dir", None):
        config["data"].update(source="folder", local_dir=str(project_path(args.local_dir)))
    if getattr(args, "endpoint", None):
        config["data"]["hf_endpoint"] = args.endpoint
    path = prepare_data(config, PROJECT_ROOT)
    print(f"Manifest: {path}", flush=True)
    return path


def _run_dir(config: dict[str, Any], name: str, model_name: str) -> Path:
    if not name or any(character in name for character in ("/", "\\", ":")) or name in {".", ".."}:
        raise ValueError("--name must be a simple run name, without path separators")
    return project_path(config.get("output_root", "outputs")) / name / model_name


def _synthetic_folder(root: Path, size: int = 80) -> Path:
    """Offline plumbing fixture; never a default research dataset."""
    import numpy as np
    from PIL import Image
    for role, count in (("train", 6), ("val", 2), ("test", 2)):
        directory = root / role
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            yy, xx = np.mgrid[:size, :size]
            phase = {"train": 1, "val": 20, "test": 40}[role] + index
            rng = np.random.default_rng(20261023 + phase)
            clean = np.stack([(xx * (phase + 1) + yy) % 256,
                              (yy * 3 + 30 * np.sin(xx / (phase + 2))) % 256,
                              (xx * 2 + yy * (phase + 3)) % 256], axis=-1)
            clean += rng.normal(0, 2, clean.shape)
            Image.fromarray(clean.clip(0, 255).astype(np.uint8)).save(directory / f"{role}_{index:02d}.png")
    return root


def smoke(args) -> None:
    from data import prepare_data
    from evaluate import evaluate
    from train import train
    config = _configuration(args)
    smoke_root = project_path(config.get("output_root", "outputs")) / (args.name or "smoke")
    if smoke_root.exists() and any(smoke_root.iterdir()):
        raise FileExistsError(f"Smoke directory already contains results: {smoke_root}; choose a new --name")
    full_model = bool(args.full_model)
    config["data"].update(source="folder", local_dir=str(_synthetic_folder(smoke_root / "fixture", 256 if full_model else 80)),
                          root=str(smoke_root / "data"), train_patch=128 if full_model else 32,
                          eval_crop=128 if full_model else 32,
                          eval_sigmas=[25], train_sigmas=[15, 25, 50], max_images=None)
    if not full_model:
        config["model"].update(width=8, block_counts=[1, 1, 1, 1, 1])
    config["train"].update(max_steps=4, warmup_steps=1, batch_size=4 if full_model else 2, grad_accumulation=1,
                           num_workers=0, log_every=1, val_every=2, save_every=2, val_max_samples=2)
    config["eval"].update(warmup=1, repeats=1)
    manifest = prepare_data(config, PROJECT_ROOT)
    reports = {}
    for name in ("nafnet", "reusable_refiner"):
        current = _model_config(config, name)
        current["eval"]["exits"] = [1] if name == "nafnet" else [1, 2, 4, 6, 8]
        output = smoke_root / name
        checkpoint = train(current, manifest, output, stop_after=2)
        checkpoint = train(current, manifest, output, resume=checkpoint)
        reports[name] = evaluate(current, checkpoint, manifest, output / "test", split="test", benchmark=True)
    write_json(smoke_root / "smoke_summary.json", {"purpose": "offline plumbing fixture, no research claims",
               "manifest": str(manifest), "models": reports})
    print(f"Smoke test passed: {smoke_root}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "train", "eval", "compare", "smoke"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--config", help="YAML configuration; defaults to configs/denoise.yaml")
        sub.add_argument("--device", choices=("auto", "cpu", "cuda"))
        if command in {"prepare", "train", "compare"}:
            sub.add_argument("--max-images", type=int, help="Cap images per role after the fixed full split")
            sub.add_argument("--local-dir", help="Use an existing local RGB image folder")
            sub.add_argument("--endpoint", help="Explicit Hugging Face endpoint, e.g. an available mirror")
        if command in {"train", "eval", "compare"}:
            sub.add_argument("--manifest", help="Use an already prepared immutable manifest")
        if command in {"train", "compare", "smoke"}:
            sub.add_argument("--name", help="Run name below output_root")
        if command == "train":
            sub.add_argument("--model", choices=("nafnet", "reusable_refiner"))
            sub.add_argument("--resume", help="Resume a full last.pt checkpoint")
            sub.add_argument("--stop-after", type=int, help="Stop at this global optimizer step without changing the schedule")
            sub.add_argument("--max-steps", type=int, help="Set total schedule steps for a new run")
        if command == "compare":
            sub.add_argument("--max-steps", type=int)
            sub.add_argument("--benchmark", action="store_true")
        if command == "eval":
            sub.add_argument("--checkpoint", required=True)
            sub.add_argument("--split", choices=("val", "test"), default="test")
            sub.add_argument("--exits", type=int, nargs="+")
            sub.add_argument("--benchmark", action="store_true")
            sub.add_argument("--max-samples", type=int, default=0)
            sub.add_argument("--output")
        if command == "smoke":
            sub.add_argument("--full-model", action="store_true", help="Check configured architecture with batch 4 and 128px patches")
    args = parser.parse_args()
    if args.command == "smoke":
        smoke(args)
        return
    saved = None
    checkpoint = getattr(args, "checkpoint", None) or getattr(args, "resume", None)
    if checkpoint:
        saved = load_checkpoint(project_path(checkpoint))
    config = _configuration(args, saved["config"] if saved else None)
    if getattr(args, "max_steps", None) is not None:
        if args.max_steps < 1:
            raise ValueError("--max-steps must be positive")
        config["train"]["max_steps"] = args.max_steps
        config["train"]["warmup_steps"] = min(config["train"].get("warmup_steps", 0), args.max_steps - 1)
    if args.command == "prepare":
        _prepare(config, args)
        return
    if getattr(args, "manifest", None):
        manifest = project_path(args.manifest)
    elif saved:
        manifest = Path(saved["manifest_path"])
    else:
        manifest = _prepare(config, args)
    if args.command == "eval":
        from evaluate import evaluate
        output = project_path(args.output) if args.output else project_path(args.checkpoint).parent / args.split
        result = evaluate(config, project_path(args.checkpoint), manifest, output, args.split,
                          args.exits, args.benchmark, args.max_samples)
        print(json.dumps({"samples": result["samples"], "exits": result["exits"],
                          "by_exit": result["by_exit"], "summary": str(output / "summary.json")},
                         ensure_ascii=False, indent=2), flush=True)
    elif args.command == "train":
        from train import train
        if args.model:
            config = _model_config(config, args.model)
        output = (project_path(args.resume).parent if args.resume and not args.name else
                  _run_dir(config, args.name or "initial", config["model"]["name"]))
        path = train(config, manifest, output, project_path(args.resume) if args.resume else None, args.stop_after)
        print(f"Checkpoint: {path}", flush=True)
    else:
        from evaluate import evaluate
        from train import train
        result = {}
        for name in ("nafnet", "reusable_refiner"):
            current = _model_config(config, name)
            output = _run_dir(current, args.name or "initial", name)
            train(current, manifest, output)
            result[name] = evaluate(current, output / "best.pt", manifest, output / "test", benchmark=args.benchmark)
        destination = _run_dir(config, args.name or "initial", "nafnet").parent / "comparison.json"
        write_json(destination, {"manifest": str(manifest), "results": result,
                   "note": "Initial same-data comparison, not matched-latency baselines or a trained stopping policy"})
        print(f"Comparison: {destination}", flush=True)


if __name__ == "__main__":
    main()
