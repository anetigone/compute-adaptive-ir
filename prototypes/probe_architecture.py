"""Engineering checks and CUDA measurements; no trained-quality claims.

Example (PowerShell):
& 'H:\\anaconda3\\envs\\py312\\python.exe' prototypes/probe_architecture.py
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

from reusable_refiner import ReusableRefiner


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower, upper = math.floor(index), math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def measure(model: ReusableRefiner, size: int, dtype: torch.dtype, steps: int, warmup: int, repeats: int, dynamic: bool = False, force: str = "last") -> dict:
    gc.collect()
    torch.cuda.empty_cache()
    y = torch.rand(1, 3, size, size, device="cuda")
    lambda_value = float("-inf") if force == "last" else float("inf")
    def call():
        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            return model.forward_dynamic(y, steps, lambda_value) if dynamic else model.forward_fixed(y, steps)
    with torch.inference_mode():
        for _ in range(warmup):
            out = call()
        del out
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        allocated_baseline = torch.cuda.memory_allocated()
        samples = []
        exit_steps = None
        for _ in range(repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = call()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
            if dynamic:
                exit_steps = out.steps
            del out
    row = {
        "width": model.width, "size": size, "precision": str(dtype).replace("torch.", ""),
        "mode": f"dynamic_forced_{force}" if dynamic else "fixed_final_readout_only",
        "requested_steps": steps, "actual_steps": exit_steps if dynamic else steps,
        "warmup": warmup, "repeats": repeats,
        "mean_ms": statistics.mean(samples), "p50_ms": statistics.median(samples),
        "p95_ms": percentile(samples, 0.95), "samples_ms": samples,
        "baseline_allocated_MiB": allocated_baseline / 1024**2,
        "peak_allocated_MiB": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_MiB": torch.cuda.max_memory_reserved() / 1024**2,
    }
    print(json.dumps({k: v for k, v in row.items() if k != "samples_ms"}), flush=True)
    return row


def validate(model: ReusableRefiner) -> dict:
    model.eval()
    parameter_count = sum(p.numel() for p in model.parameters())
    with torch.inference_mode():
        y = torch.rand(1, 3, 65, 67, device="cuda")
        trajectory = model.forward_trajectory(y, 4)
        fixed_first = model.forward_fixed(y, 1)
        fixed_last = model.forward_fixed(y, 4)
        early = model.forward_dynamic(y, 4, float("inf"))
        late = model.forward_dynamic(y, 4, float("-inf"))
        tiny = model.forward_fixed(torch.rand(1, 3, 1, 2, device="cuda"), 1)
        checks = {
            "shape_preserved": all(output.shape == y.shape for output in trajectory),
            "finite_trajectory": all(bool(torch.isfinite(output).all().item()) for output in trajectory),
            "trajectory_final_matches_fixed": bool(torch.allclose(trajectory[-1], fixed_last, atol=1e-6, rtol=1e-5)),
            "forced_first_exit": early.steps == 1 and bool(torch.allclose(early.image, fixed_first, atol=1e-6, rtol=1e-5)),
            "forced_last_exit": late.steps == 4 and bool(torch.allclose(late.image, fixed_last, atol=1e-6, rtol=1e-5)),
            "tiny_image_padding": tuple(tiny.shape) == (1, 3, 1, 2),
            "parameter_count_independent_of_steps": parameter_count == sum(p.numel() for p in model.parameters()),
        }
        try:
            model.forward_dynamic(y, model.train_steps + 1)
            checks["dynamic_horizon_guard"] = False
        except ValueError:
            checks["dynamic_horizon_guard"] = True
    if not all(checks.values()):
        raise AssertionError(checks)
    return checks


def component_probe(model: ReusableRefiner, size: int = 512, repeats: int = 30) -> dict:
    """Isolated component timings diagnose overhead; they are not additive.

    In particular an isolated readback from a ready scalar does not model
    waiting for preceding restoration work. Use complete dynamic timings
    for any performance conclusion.
    """
    model.eval()
    h = torch.rand(1, model.width, size, size, device="cuda", dtype=torch.float16)
    input_gap = h.float().mean(dim=(-2, -1))
    previous_gap = input_gap.clone()
    policy_input, _ = model.policy_features(h, input_gap, previous_gap, (size, size), 1)
    with torch.autocast("cuda", dtype=torch.float16):
        ready_gains = model.gain_head(policy_input).float()[0]
    ready_score = ready_gains.max() > 0
    costs = ready_gains.new_tensor([1.0, 2.0, 3.0])
    def features_only():
        return model.policy_features(h, input_gap, previous_gap, (size, size), 1)
    def mlp_only():
        with torch.autocast("cuda", dtype=torch.float16):
            return model.gain_head(policy_input)
    def readback_only():
        return bool(ready_score.item())
    def whole_policy():
        vector, _ = features_only()
        with torch.autocast("cuda", dtype=torch.float16):
            gains = model.gain_head(vector).float()[0]
        return bool(((gains - 0.1 * costs).max() > 0).item())
    rows = {}
    with torch.inference_mode():
        for name, call in (("policy_features", features_only), ("MLP", mlp_only), ("ready_scalar_readback_and_branch", readback_only), ("whole_policy_from_ready_h", whole_policy)):
            for _ in range(10):
                call()
            samples = []
            for _ in range(repeats):
                torch.cuda.synchronize()
                start = time.perf_counter()
                call()
                torch.cuda.synchronize()
                samples.append((time.perf_counter() - start) * 1000)
            rows[name] = {"mean_ms": statistics.mean(samples), "p50_ms": statistics.median(samples), "p95_ms": percentile(samples, 0.95)}
    return {"width": model.width, "size": size, "repeats": repeats, "precision": "float16 state / FP32 reductions / AMP MLP", "components": rows, "limitation": "isolated synchronized diagnostics; component times are not additive and ready-scalar readback understates waiting for preceding GPU work"}


def training_probe(model: ReusableRefiner) -> dict:
    gc.collect()
    torch.cuda.empty_cache()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    clean = torch.rand(1, 3, 128, 128, device="cuda")
    noisy = (clean + torch.randn_like(clean) * (25 / 255)).clamp(0, 1)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.float16):
        outputs = model.forward_trajectory(noisy, 4)
        # Independent synthetic forward/backward check, not a training protocol.
        loss = sum((output.float() - clean).square().mean() for output in outputs) / len(outputs)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(loss).item() or not torch.isfinite(grad_norm).item():
        raise AssertionError("Nonfinite training loss/gradient")
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    result = {
        "width": model.width, "batch": 1, "patch": 128, "unroll": 4,
        "precision": "AMP float16, FP32 loss", "optimizer": "AdamW", "optimizer_updates": 1,
        "loss": float(loss.item()), "gradient_norm_before_clipping": float(grad_norm.item()),
        "elapsed_ms_including_backward_and_first_optimizer_step": (time.perf_counter() - start) * 1000,
        "peak_allocated_MiB": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_MiB": torch.cuda.max_memory_reserved() / 1024**2,
        "purpose": "synthetic gradient/memory check only; no trained-quality result",
    }
    optimizer.zero_grad(set_to_none=True)
    del optimizer, scaler, outputs, clean, noisy, loss
    model.eval()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--widths", type=int, nargs="+", default=[24, 32])
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 512])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("probe_results.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This probe expects a CUDA device")
    torch.manual_seed(20261021)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result = {
        "status": "untrained engineering probe", "python": sys.version,
        "executable": sys.executable, "platform": platform.platform(),
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "device": torch.cuda.get_device_name(),
        "device_total_GiB": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "timing_boundary": "input already on GPU; output remains on GPU; synchronized CPU wall-clock; includes Python launch/branch and policy sync; excludes I/O/quality metrics",
        "backend": {"cudnn_benchmark": False, "cudnn_tf32": False, "matmul_tf32": False, "compile": False},
        "limitations": ["random weights and random gain head", "short engineering timings, not publication benchmark", "forced exits exercise execution paths only", "sequential method order; laptop thermals/background processes uncontrolled"],
        "models": [], "timings": [], "training_probes": [], "component_probes": [],
    }
    for width in args.widths:
        model = ReusableRefiner(width=width).cuda().eval()
        record = {
            "width": width, "block_counts": [1, 1, 2, 1, 1],
            "parameters_total": sum(p.numel() for p in model.parameters()),
            "parameters_gain_head": sum(p.numel() for p in model.gain_head.parameters()),
            "checks": validate(model),
        }
        result["models"].append(record)
        print(json.dumps(record), flush=True)
        for size in args.sizes:
            for dtype in (torch.float32, torch.float16):
                for steps in (1, 2, 4, 8):
                    result["timings"].append(measure(model, size, dtype, steps, args.warmup, args.repeats))
                for force in ("first", "last"):
                    result["timings"].append(measure(model, size, dtype, 4, args.warmup, args.repeats, dynamic=True, force=force))
        result["component_probes"].append(component_probe(model, max(args.sizes)))
        result["training_probes"].append(training_probe(model))
        del model
        gc.collect()
        torch.cuda.empty_cache()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
