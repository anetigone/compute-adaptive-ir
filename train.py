"""Fixed-budget restoration training, including exact data-position resume."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from common import (capture_rng_state, environment_info, file_sha256, load_checkpoint,
                    resolve_device, restore_rng_state, save_checkpoint, seed_everything, write_json)
from config import load_config, project_path
from data import build_datasets
from models import build_model


def make_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int):
    if total_steps < 1 or not 0 <= warmup_steps < total_steps:
        raise ValueError("Require max_steps >= 1 and 0 <= warmup_steps < max_steps")

    def multiplier(index: int) -> float:
        if index < warmup_steps:
            return (index + 1) / max(1, warmup_steps)
        progress = min(1.0, (index - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def optimizer_step(optimizer, scaler, parameters, max_norm: float) -> float | None:
    """Return the unscaled norm, or None if GradScaler skipped an overflow."""
    scaler.unscale_(optimizer)
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        raise RuntimeError("No gradients were produced")
    # FP32 norm reduction can itself overflow even when every gradient is finite.
    norm = torch.linalg.vector_norm(torch.stack([
        torch.linalg.vector_norm(gradient.detach(), dtype=torch.float64) for gradient in gradients
    ])).item()
    if not math.isfinite(norm):
        if not scaler.is_enabled():
            raise FloatingPointError("Nonfinite unscaled gradient with AMP disabled")
        # unscale_ recorded inf/nan. step skips the optimizer; update lowers scale.
        # Do not clip nonfinite gradients or advance the learning-rate scheduler.
        scaler.step(optimizer)
        scaler.update()
        return None
    coefficient = min(1.0, max_norm / (norm + 1e-6))
    for gradient in gradients:
        gradient.mul_(coefficient)
    scaler.step(optimizer)
    scaler.update()
    return norm


class TrainingBatches:
    """Epoch permutation and augmentation are independent of model/global RNG."""

    def __init__(self, dataset, batch_size: int, seed: int, num_workers: int = 0,
                 pin_memory: bool = False, state: dict[str, int] | None = None):
        if batch_size < 1 or len(dataset) < 1:
            raise ValueError("Training requires a nonempty dataset and positive batch size")
        self.dataset, self.batch_size, self.seed = dataset, batch_size, seed
        self.num_workers, self.pin_memory = num_workers, pin_memory
        state = state or {"epoch": 0, "batch_offset": 0}
        self.epoch, self.batch_offset = state["epoch"], state["batch_offset"]
        self.iterator = None
        self.batches_per_epoch = math.ceil(len(dataset) / batch_size)
        if self.epoch < 0 or not 0 <= self.batch_offset < self.batches_per_epoch:
            raise ValueError("Invalid checkpoint data position")

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "batch_offset": self.batch_offset}

    def next(self) -> dict[str, Any]:
        if self.iterator is None:
            self.dataset.set_epoch(self.epoch)
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            order = torch.randperm(len(self.dataset), generator=generator).tolist()
            batches = [order[start:start + self.batch_size] for start in range(0, len(order), self.batch_size)]
            # Loader iterator construction must not consume global model RNG.
            loader_generator = torch.Generator().manual_seed(self.seed + self.epoch + 1000003)
            loader = DataLoader(self.dataset, batch_sampler=batches[self.batch_offset:],
                                num_workers=self.num_workers, pin_memory=self.pin_memory,
                                generator=loader_generator)
            self.iterator = iter(loader)
        batch = next(self.iterator)
        self.batch_offset += 1
        if self.batch_offset == self.batches_per_epoch:
            self.epoch += 1
            self.batch_offset = 0
            self.iterator = None
        return batch


def _resume_signature(config: dict[str, Any]) -> dict[str, Any]:
    training = config["train"]
    keys = ("max_steps", "warmup_steps", "batch_size", "grad_accumulation", "lr",
            "weight_decay", "grad_clip", "amp", "deterministic", "exit_weights",
            "val_every", "val_max_samples")
    return {"model": config["model"], "data": config["data"], "seed": config.get("seed", 20261023),
            "train": {key: training.get(key) for key in keys}}


def train(config: dict[str, Any], manifest_path: str | Path, output_dir: str | Path,
          resume: str | Path | None = None, stop_after: int | None = None) -> Path:
    config = copy.deepcopy(config)
    settings = config["train"]
    manifest_path, output_dir = Path(manifest_path).resolve(), Path(output_dir).resolve()
    checkpoint_path = output_dir / "last.pt"
    if resume is not None and Path(resume).resolve() != checkpoint_path:
        raise ValueError("Resume must use last.pt in its original run directory; choose a new run for different experiments")
    if checkpoint_path.exists() and resume is None:
        raise FileExistsError(f"Run already exists: {checkpoint_path}; use --resume or another --name")
    if resume is None and any((output_dir / name).exists() for name in ("run.json", "train.jsonl")):
        raise FileExistsError(f"Run directory already contains experiment records: {output_dir}; use another --name")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.get("device", "auto"))
    torch.set_num_threads(int(config.get("cpu_threads", 4)))
    seed = int(config.get("seed", 20261023))
    seed_everything(seed, bool(settings.get("deterministic", False)))
    datasets = build_datasets(config, manifest_path)
    model = build_model(config["model"]).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(settings.get("lr", 2e-4)),
                                 betas=(0.9, 0.999), weight_decay=float(settings.get("weight_decay", 0.0)))
    total_steps = int(settings.get("max_steps", 20000))
    scheduler = make_scheduler(optimizer, total_steps, int(settings.get("warmup_steps", 200)))
    amp_enabled = bool(settings.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    manifest_hash = file_sha256(manifest_path)
    step, best_psnr, data_state = 0, -math.inf, None
    amp_skipped_updates = 0
    if resume is not None:
        saved = load_checkpoint(resume)
        if saved["resume_signature"] != _resume_signature(config):
            raise ValueError("Resume configuration changed; keep model/data/seed/optimizer/schedule unchanged")
        if saved["manifest_sha256"] != manifest_hash:
            raise ValueError("Data manifest changed since checkpoint")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        step, best_psnr, data_state = saved["step"], saved["best_psnr"], saved["data_position"]
        amp_skipped_updates = saved.get("amp_skipped_updates", 0)
        restore_rng_state(saved["rng"])
    target_step = total_steps if stop_after is None else min(total_steps, int(stop_after))
    if target_step <= step:
        raise ValueError(f"Requested stop step {target_step} must exceed saved step {step}")
    accumulation = int(settings.get("grad_accumulation", 1))
    if accumulation < 1:
        raise ValueError("grad_accumulation must be positive")
    batches = TrainingBatches(datasets["train"], int(settings.get("batch_size", 4)), seed,
                              int(settings.get("num_workers", 0)), device.type == "cuda", data_state)
    val_every = max(1, int(settings.get("val_every", 500)))
    save_every = max(1, int(settings.get("save_every", val_every)))
    log_every = max(1, int(settings.get("log_every", 20)))
    description = {"config": config, "manifest_path": str(manifest_path), "manifest_sha256": manifest_hash,
                   "environment": environment_info(device), "parameters": sum(p.numel() for p in model.parameters()),
                   "trainable_parameters": sum(p.numel() for p in trainable),
                   "scope": "restoration only; gain head stays untrained/frozen"}
    write_json(output_dir / "run.json", description)
    print(json.dumps({"model": config["model"]["name"], "device": str(device), "from_step": step,
                      "to_step": target_step, "parameters": description["parameters"]}), flush=True)
    start = time.perf_counter()
    consecutive_overflows = 0
    log_path = output_dir / "train.jsonl"
    if resume is not None and log_path.exists():
        # A crash can leave logged updates newer than the last durable checkpoint.
        # Preserve those diagnostics separately before replaying the saved stream.
        original = log_path.read_text(encoding="utf-8")
        kept = []
        for line in original.splitlines():
            try:
                record = json.loads(line)
                if isinstance(record, dict) and int(record["step"]) <= step:
                    kept.append(line)
            except (ValueError, KeyError, TypeError):
                pass  # An interrupted final JSONL write is not a completed update.
        trimmed = "".join(line + "\n" for line in kept)
        if trimmed != original:
            backup = output_dir / f"train.before_resume_{time.time_ns()}.jsonl"
            backup.write_text(original, encoding="utf-8")
            log_path.write_text(trimmed, encoding="utf-8")

    def payload() -> dict[str, Any]:
        return {"format_version": 1, "config": config, "resume_signature": _resume_signature(config),
                "manifest_path": str(manifest_path), "manifest_sha256": manifest_hash,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "amp_skipped_updates": amp_skipped_updates,
                "step": step, "best_psnr": best_psnr, "data_position": batches.state_dict(),
                "rng": capture_rng_state(), "environment": environment_info(device)}

    with log_path.open("a", encoding="utf-8") as log:
        while step < target_step:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            mean_loss = 0.0
            lr_used = optimizer.param_groups[0]["lr"]
            for _ in range(accumulation):
                batch = batches.next()
                noisy = batch["noisy"].to(device, non_blocking=True)
                clean = batch["clean"].to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    outputs = model.forward_trajectory(noisy)
                    weights = settings.get("exit_weights") or [1.0] * len(outputs)
                    if len(weights) != len(outputs) or any(float(weight) < 0 for weight in weights) or sum(weights) <= 0:
                        raise ValueError("exit_weights must be nonnegative, nonzero, and match the number of training exits")
                    losses = [nn.functional.l1_loss(output.float(), clean.float()) for output in outputs]
                    loss = sum(float(weight) * value for weight, value in zip(weights, losses)) / sum(weights)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(f"Nonfinite loss at optimizer step {step + 1}")
                scaler.scale(loss / accumulation).backward()
                mean_loss += float(loss.detach()) / accumulation
            scale_before = scaler.get_scale()
            grad_norm = optimizer_step(optimizer, scaler, trainable, float(settings.get("grad_clip", 1.0)))
            if grad_norm is None:
                amp_skipped_updates += 1
                consecutive_overflows += 1
                event = {"event": "amp_overflow", "step": step, "attempted_step": step + 1,
                         "amp_scale_before": scale_before, "amp_scale_after": scaler.get_scale(),
                         "consecutive_overflows": consecutive_overflows,
                         "amp_skipped_updates": amp_skipped_updates}
                log.write(json.dumps(event) + "\n")
                log.flush()
                print(json.dumps(event), flush=True)
                if consecutive_overflows >= 16:
                    raise FloatingPointError(
                        f"16 consecutive AMP gradient overflows before step {step + 1}; "
                        "check loss/model stability. The optimizer and scheduler were not advanced."
                    )
                continue
            consecutive_overflows = 0
            scheduler.step()
            step += 1
            row: dict[str, Any] = {"step": step, "loss_l1": mean_loss, "lr": lr_used,
                                   "grad_norm": float(grad_norm), "elapsed_seconds": time.perf_counter() - start,
                                   "data_epoch": batches.epoch, "data_batch_offset": batches.batch_offset}
            if amp_enabled:
                row.update(amp_scale=scaler.get_scale(), amp_skipped_updates=amp_skipped_updates)
            improved = False
            # Checkpoint selection follows a fixed schedule. An interruption at
            # an arbitrary step must not introduce an extra "best" candidate.
            if step % val_every == 0 or step == total_steps:
                from evaluate import validate_model
                validation = validate_model(model, datasets["val"], device, amp_enabled,
                                            max_samples=int(settings.get("val_max_samples", 0)))
                row["validation"] = validation
                improved = validation["psnr"] > best_psnr
                best_psnr = max(best_psnr, validation["psnr"])
            log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            log.flush()
            if step % log_every == 0 or step == 1 or step == target_step or "validation" in row:
                print(json.dumps(row, ensure_ascii=False), flush=True)
            if improved or step % save_every == 0 or step == target_step:
                current = payload()
                save_checkpoint(checkpoint_path, current)
                if improved:
                    save_checkpoint(output_dir / "best.pt", current)
    write_json(output_dir / "training_summary.json", {"step": step, "schedule_steps": total_steps,
               "completed_schedule": step == total_steps,
               "best_validation_psnr": best_psnr if math.isfinite(best_psnr) else None,
               "elapsed_seconds_this_invocation": time.perf_counter() - start,
               "amp_skipped_updates": amp_skipped_updates,
               "last_checkpoint": str(checkpoint_path), "gain_head_trained": False})
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/denoise.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device:
        config["device"] = args.device
    train(config, project_path(args.manifest), project_path(args.output), args.resume, args.stop_after)


if __name__ == "__main__":
    main()
