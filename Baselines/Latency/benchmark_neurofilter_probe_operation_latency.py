#!/usr/bin/env python3
"""Repeated latency microbenchmark for NeuroFilter probe operations.

Measures two quantities for the single-turn and multi-turn probes:

1. GPU compute time, measured with CUDA events around only the probe kernels.
2. Decision time, measured from Python immediately before the operations until
   the Boolean threshold result is available on the CPU.

Activations and weights remain resident on the GPU.  No LLM is loaded and no
network or API access is used.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch


ROOT = Path("/path/to/code")
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "logs/neurofilter_probe_operation_latency_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=10000)
    parser.add_argument("--gpu-batches", type=int, default=100)
    parser.add_argument("--operations-per-batch", type=int, default=1000)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    for name in ("warmup", "repeats", "gpu_batches", "operations_per_batch"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    return args


def atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def percentile_summary(values_ms: np.ndarray) -> dict[str, float]:
    return {
        "mean_ms": float(np.mean(values_ms)),
        "median_ms": float(np.median(values_ms)),
        "p95_ms": float(np.quantile(values_ms, 0.95)),
        "std_ms": float(np.std(values_ms, ddof=1)),
        "min_ms": float(np.min(values_ms)),
        "max_ms": float(np.max(values_ms)),
    }


def make_single_turn_operation(
    width: int, device: torch.device
) -> Callable[[], torch.Tensor]:
    activation_bf16 = torch.randn(width, dtype=torch.bfloat16, device=device)
    weights_fp32 = torch.randn(width, dtype=torch.float32, device=device)
    bias_fp32 = torch.tensor(0.0, dtype=torch.float32, device=device)
    threshold_fp32 = torch.tensor(0.0, dtype=torch.float32, device=device)

    def operation() -> torch.Tensor:
        # The hook receives BF16 model output and the sklearn weights are FP32.
        activation_fp32 = activation_bf16.float()
        score = torch.dot(activation_fp32, weights_fp32) + bias_fp32
        return score > threshold_fp32

    return operation


def make_multi_turn_operation(
    width: int, device: torch.device
) -> Callable[[], torch.Tensor]:
    current_bf16 = torch.randn(width, dtype=torch.bfloat16, device=device)
    previous_fp32 = torch.randn(width, dtype=torch.float32, device=device)
    weights_fp32 = torch.randn(width, dtype=torch.float32, device=device)
    cumulative_fp32 = torch.tensor(0.0, dtype=torch.float32, device=device)
    threshold_fp32 = torch.tensor(0.0, dtype=torch.float32, device=device)

    def operation() -> torch.Tensor:
        # This matches the activation-velocity decision used at each later turn.
        current_fp32 = current_bf16.float()
        delta_fp32 = current_fp32 - previous_fp32
        projection = torch.dot(delta_fp32, weights_fp32)
        cumulative = cumulative_fp32 + projection
        return cumulative > threshold_fp32

    return operation


def warm_up(operation: Callable[[], torch.Tensor], count: int) -> None:
    with torch.inference_mode():
        for _ in range(count):
            operation()
    torch.cuda.synchronize()


def measure_gpu_compute(
    operation: Callable[[], torch.Tensor],
    batches: int,
    operations_per_batch: int,
) -> np.ndarray:
    """Return per-operation CUDA time for independently timed batches."""
    batch_averages_ms: list[float] = []
    with torch.inference_mode():
        for _ in range(batches):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(operations_per_batch):
                operation()
            end.record()
            end.synchronize()
            batch_averages_ms.append(
                float(start.elapsed_time(end)) / operations_per_batch
            )
    return np.asarray(batch_averages_ms, dtype=np.float64)


def measure_cpu_readable_decision(
    operation: Callable[[], torch.Tensor], repeats: int
) -> np.ndarray:
    """Return wall time until each Boolean decision is available to Python."""
    values_ms = np.empty(repeats, dtype=np.float64)
    torch.cuda.synchronize()
    with torch.inference_mode():
        for index in range(repeats):
            started_ns = time.perf_counter_ns()
            bool(operation().item())
            values_ms[index] = (time.perf_counter_ns() - started_ns) / 1_000_000.0
    return values_ms


def benchmark_one(
    name: str,
    width: int,
    operation: Callable[[], torch.Tensor],
    args: argparse.Namespace,
) -> tuple[list[dict], pd.DataFrame]:
    print(f"Warming up {name} ({width:,} values) for {args.warmup:,} iterations.")
    warm_up(operation, args.warmup)

    print(
        f"Measuring {name} GPU compute: {args.gpu_batches:,} batches x "
        f"{args.operations_per_batch:,} operations."
    )
    gpu_ms = measure_gpu_compute(
        operation, args.gpu_batches, args.operations_per_batch
    )

    print(
        f"Measuring {name} CPU-readable decisions: {args.repeats:,} repetitions."
    )
    decision_ms = measure_cpu_readable_decision(operation, args.repeats)

    summary_rows = []
    for measurement, values in (
        ("gpu_compute", gpu_ms),
        ("cpu_readable_decision", decision_ms),
    ):
        row = {
            "probe": name,
            "width": width,
            "measurement": measurement,
            "samples": int(len(values)),
            **percentile_summary(values),
        }
        summary_rows.append(row)

    raw = pd.concat(
        [
            pd.DataFrame(
                {
                    "probe": name,
                    "width": width,
                    "measurement": "gpu_compute_batch_average",
                    "sample": np.arange(len(gpu_ms)),
                    "latency_ms": gpu_ms,
                }
            ),
            pd.DataFrame(
                {
                    "probe": name,
                    "width": width,
                    "measurement": "cpu_readable_decision",
                    "sample": np.arange(len(decision_ms)),
                    "latency_ms": decision_ms,
                }
            ),
        ],
        ignore_index=True,
    )
    return summary_rows, raw


def main() -> None:
    args = parse_args()
    print(f"Device requested: {args.device}")
    print(f"Warm-up iterations: {args.warmup:,}")
    print(f"CPU-readable decision repetitions: {args.repeats:,}")
    print(
        f"GPU timing: {args.gpu_batches:,} batches x "
        f"{args.operations_per_batch:,} operations"
    )
    print("Models loaded: none")
    print("Network/API access: none")
    if args.validate_only:
        print("Validation complete. No GPU work or output was performed.")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this benchmark on a GPU node.")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("This benchmark requires a CUDA device.")
    torch.cuda.set_device(device)
    torch.manual_seed(42)

    device_index = torch.cuda.current_device()
    metadata = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device_index),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
        "warmup": args.warmup,
        "decision_repeats": args.repeats,
        "gpu_batches": args.gpu_batches,
        "operations_per_gpu_batch": args.operations_per_batch,
        "activation_dtype": "bfloat16",
        "probe_dtype": "float32",
        "models_loaded": False,
        "network_api_access": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / "manifest.json", metadata)
    print(f"GPU: {metadata['gpu_name']}")

    configurations = [
        (
            "single_turn_qwen_2_5_7b",
            3584,
            make_single_turn_operation(3584, device),
        ),
        (
            "multi_turn_qwen_2_5_32b",
            5120,
            make_multi_turn_operation(5120, device),
        ),
    ]
    all_summary: list[dict] = []
    all_raw: list[pd.DataFrame] = []
    for name, width, operation in configurations:
        summary_rows, raw = benchmark_one(name, width, operation, args)
        all_summary.extend(summary_rows)
        all_raw.append(raw)

    summary = pd.DataFrame(all_summary)
    raw = pd.concat(all_raw, ignore_index=True)
    summary.to_csv(args.output_dir / "probe_operation_latency_summary.csv", index=False)
    raw.to_csv(args.output_dir / "probe_operation_latency_raw.csv", index=False)
    atomic_write_json(
        args.output_dir / "probe_operation_latency_summary.json",
        {"metadata": metadata, "results": all_summary},
    )
    print("\nProbe operation latency summary (milliseconds)")
    print(summary.to_string(index=False))
    print(f"\nSaved results to {args.output_dir}")


if __name__ == "__main__":
    main()
