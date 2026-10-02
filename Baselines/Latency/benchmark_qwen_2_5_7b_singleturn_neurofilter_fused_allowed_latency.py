#!/usr/bin/env python3
"""Correct single-turn NeuroFilter latency benchmark for Qwen 2.5 7B.

The benchmark replays the held-out benign CMPL Insurance/AutoDAN prompts from
the original single-turn experiment.  For every prompt it compares normal Qwen
generation with the same generation while a one-shot hook applies one dense
3,584-dimensional logistic probe to the already-computed final-layer
activation.  NeuroFilter never performs another model forward.

The historical experiment did not save its fitted 7B probe coefficients.  A
fixed timing-equivalent vector is therefore used here.  Its values do not
change the operation count or latency of a dense logistic probe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import benchmark_qwen_2_5_32b_neurofilter_fused_allowed_latency as fused_reference
import benchmark_qwen_2_5_32b_neurofilter_vs_llamaguard4_latency as reference


ROOT = Path("/path/to/code")
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
PROFILES_FILE = ROOT / "benchmark/datasets/generated_profiles_scenario_2.json"
BENCHMARK_FILE = (
    Path("/path/to/code")
    / "Data/autodan/"
    "autodan_style_matched_benchmark_insurance_10_per_type_nonuniformscores.json"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "logs/"
    "qwen_2_5_7b_singleturn_neurofilter_fused_allowed_latency_v1"
)
EXPERIMENT = "qwen_2_5_7b_singleturn_neurofilter_fused_allowed_latency_v1"
PROBE_LAYER = 28
HIDDEN_SIZE = 3584
RANDOM_SEED = 42
TEST_SIZE = 0.30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument(
        "--limit-prompts",
        type=int,
        default=None,
        help="Optional smoke-test limit; the default uses all 720 benign prompts.",
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive.")
    if args.warmup_requests < 0:
        parser.error("--warmup-requests must not be negative.")
    if args.limit_prompts is not None and args.limit_prompts < 1:
        parser.error("--limit-prompts must be positive.")
    return args


def read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return reference.sha256_file(path)


def load_benign_test_prompts(limit: int | None) -> list[dict[str, Any]]:
    profiles = read_json(PROFILES_FILE)
    benchmark = read_json(BENCHMARK_FILE)
    benign_prompts = benchmark.get("benign_prompts")
    privacy_prompts = benchmark.get("privacy_violating_prompts")
    if not isinstance(profiles, list) or len(profiles) != 200:
        raise RuntimeError(f"Expected 200 profiles, found {len(profiles)}.")
    if not isinstance(benign_prompts, list) or len(benign_prompts) != 40:
        raise RuntimeError("Expected 40 benign AutoDAN templates.")
    if not isinstance(privacy_prompts, list) or len(privacy_prompts) != 40:
        raise RuntimeError("Expected 40 privacy-violating AutoDAN templates.")

    indexed_profiles = list(enumerate(profiles))
    _, test_profiles = train_test_split(
        indexed_profiles, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    # Match the original script's independent split of each prompt class.
    _, test_benign = train_test_split(
        list(enumerate(benign_prompts)),
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
    )
    _, test_privacy = train_test_split(
        list(enumerate(privacy_prompts)),
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
    )
    if len(test_profiles) != 60 or len(test_benign) != 12 or len(test_privacy) != 12:
        raise RuntimeError(
            "The fixed 70/30 split did not produce 60 profiles and 12 templates."
        )

    rows: list[dict[str, Any]] = []
    for profile_index, profile in test_profiles:
        for template_index, prompt in test_benign:
            rows.append(
                {
                    "record_id": (
                        f"benign:profile{profile_index}:template{template_index}"
                    ),
                    "profile_index": int(profile_index),
                    "template_index": int(template_index),
                    "profile": profile,
                    "user_text": f"Question: {prompt}",
                }
            )
    if len(rows) != 720 or len({row["record_id"] for row in rows}) != 720:
        raise RuntimeError(f"Expected 720 distinct benign prompts, found {len(rows)}.")
    return rows if limit is None else rows[:limit]


def load_qwen_7b() -> tuple[Any, Any, float]:
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    import time

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer, time.perf_counter() - started


def timing_equivalent_weights(device: torch.device) -> torch.Tensor:
    generator = np.random.default_rng(RANDOM_SEED)
    weights = generator.standard_normal(HIDDEN_SIZE).astype(np.float32)
    weights /= np.linalg.norm(weights)
    return torch.from_numpy(weights).to(device)


class SingleTurnProbeCapture:
    def __init__(self, weights: torch.Tensor) -> None:
        self.weights = weights
        self.current: torch.Tensor | None = None
        self.score = 0.0

    def __call__(self, _module: Any, _inputs: Any, output: Any) -> None:
        if self.current is not None:
            return
        current = fused_reference.tensor_from_hook_output(output)[0, -1, :]
        current = current.detach().float()
        if current.shape != self.weights.shape:
            raise RuntimeError(
                f"Activation shape {tuple(current.shape)} does not match probe "
                f"shape {tuple(self.weights.shape)}."
            )
        if current.device != self.weights.device:
            raise RuntimeError(
                f"Activation is on {current.device}, probe is on {self.weights.device}."
            )
        self.current = current
        # Reading the scalar is required to make the filter decision before output.
        self.score = float(torch.dot(current, self.weights).item())


def benchmark_signature(args: argparse.Namespace) -> str:
    payload = {
        "experiment": EXPERIMENT,
        "model": MODEL_ID,
        "profiles_sha256": sha256_file(PROFILES_FILE),
        "benchmark_sha256": sha256_file(BENCHMARK_FILE),
        "split_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "probe_layer": PROBE_LAYER,
        "probe_width": HIDDEN_SIZE,
        "probe_weights": "fixed_dense_timing_equivalent_seed_42",
        "max_new_tokens": args.max_new_tokens,
        "limit_prompts": args.limit_prompts,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def load_cache(path: Path, signature: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    cached: dict[str, dict[str, Any]] = {}
    for row in reference.read_jsonl(path):
        if row.get("experiment") != EXPERIMENT:
            raise RuntimeError(f"Wrong experiment in {path}.")
        if row.get("benchmark_signature") != signature:
            raise RuntimeError(f"Benchmark settings changed since {path} was created.")
        key = str(row.get("record_id", ""))
        if not key or key in cached:
            raise RuntimeError(f"Missing or duplicate record ID in {path}: {key!r}")
        cached[key] = row
    return cached


def summarize(raw_path: Path, output_dir: Path, signature: str) -> None:
    frame = pd.DataFrame(load_cache(raw_path, signature).values())
    frame["added_latency_ms"] = (
        frame["neurofilter_latency_ms"] - frame["normal_latency_ms"]
    )
    frame["overhead_percent"] = (
        100.0 * frame["added_latency_ms"] / frame["normal_latency_ms"]
    )
    frame.to_csv(output_dir / "paired_latency_measurements.csv", index=False)

    def stats(name: str, values: pd.Series) -> dict[str, Any]:
        return {
            "measurement": name,
            "requests": int(len(values)),
            "mean_ms": float(values.mean()),
            "median_ms": float(values.median()),
            "p95_ms": float(values.quantile(0.95)),
        }

    summary_rows = [
        stats("normal_qwen_end_to_end", frame["normal_latency_ms"]),
        stats("qwen_with_neurofilter_end_to_end", frame["neurofilter_latency_ms"]),
        stats("paired_neurofilter_added_latency", frame["added_latency_ms"]),
    ]
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "allowed_latency_summary.csv", index=False)
    fused_reference.atomic_write_json(
        output_dir / "allowed_latency_summary.json",
        {
            "benign_allowed_requests": int(len(frame)),
            "mean_paired_overhead_percent": float(frame["overhead_percent"].mean()),
            "median_paired_overhead_percent": float(frame["overhead_percent"].median()),
            "measurements": summary_rows,
        },
    )
    print("\nSingle-turn allowed benign latency summary")
    print(summary.to_string(index=False))
    print(
        f"Mean paired overhead: {frame['added_latency_ms'].mean():.3f} ms "
        f"({frame['overhead_percent'].mean():.4f}%)"
    )


def main() -> None:
    args = parse_args()
    prompts = load_benign_test_prompts(args.limit_prompts)
    signature = benchmark_signature(args)
    raw_path = args.output_dir / "raw_paired.jsonl"
    cache = load_cache(raw_path, signature)
    expected_ids = {row["record_id"] for row in prompts}
    unexpected = set(cache) - expected_ids
    if unexpected:
        raise RuntimeError(f"Unexpected cached rows: {sorted(unexpected)[:5]}")

    print(f"Profiles: {PROFILES_FILE}")
    print(f"AutoDAN benchmark: {BENCHMARK_FILE}")
    print(f"Model: {MODEL_ID}")
    print(f"Probe layer: {PROBE_LAYER}; dense probe width: {HIDDEN_SIZE}")
    print(f"Benign held-out prompts: {len(prompts)}")
    print(f"Existing paired measurements: {len(cache)}/{len(expected_ids)}")
    print("NeuroFilter forward passes added: 0")
    print("Network/API access: disabled")
    reference.validate_local_model_files(MODEL_ID)
    if args.validate_only:
        print("Validation complete. No model was loaded and no results were written.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fused_reference.atomic_write_json(
        args.output_dir / "manifest.json",
        {
            "experiment": EXPERIMENT,
            "benchmark_signature": signature,
            "profiles_file": str(PROFILES_FILE),
            "benchmark_file": str(BENCHMARK_FILE),
            "model": MODEL_ID,
            "probe_layer": PROBE_LAYER,
            "probe_width": HIDDEN_SIZE,
            "probe_weights": (
                "fixed dense timing-equivalent vector; the historical fitted "
                "7B coefficients were not saved"
            ),
            "prompts": len(prompts),
            "max_new_tokens": args.max_new_tokens,
            "measurement": (
                "paired normal generation versus the same generation with a "
                "one-shot probe-layer hook"
            ),
            "extra_neurofilter_model_forwards": 0,
            "network_api_access": False,
        },
    )

    model = None
    tokenizer = None
    try:
        model, tokenizer, load_seconds = load_qwen_7b()
        print(f"Loaded {MODEL_ID} in {load_seconds:.2f} seconds.")
        probe_module = fused_reference.selected_probe_module(model, PROBE_LAYER)
        module_device = next(probe_module.parameters()).device
        weights = timing_equivalent_weights(module_device)
        difference = fused_reference.validate_hook_mapping(
            model, tokenizer, probe_module, PROBE_LAYER
        )
        print(
            f"Validated hook equals hidden_states[{PROBE_LAYER}]: "
            f"max difference={difference:.8f}"
        )

        warmups = min(args.warmup_requests, len(prompts))
        for row in prompts[:warmups]:
            system_prompt = reference.build_system_prompt(row["profile"])
            fused_reference.generate_once(
                model,
                tokenizer,
                system_prompt,
                [],
                row["user_text"],
                min(8, args.max_new_tokens),
            )
            capture = SingleTurnProbeCapture(weights)
            fused_reference.generate_once(
                model,
                tokenizer,
                system_prompt,
                [],
                row["user_text"],
                min(8, args.max_new_tokens),
                probe_module,
                capture,
            )
        print(f"Completed {warmups} untimed warm-up request pairs.")

        for index, row in enumerate(prompts):
            key = row["record_id"]
            if key in cache:
                continue
            system_prompt = reference.build_system_prompt(row["profile"])
            capture = SingleTurnProbeCapture(weights)
            normal_first = index % 2 == 0
            if normal_first:
                normal = fused_reference.generate_once(
                    model,
                    tokenizer,
                    system_prompt,
                    [],
                    row["user_text"],
                    args.max_new_tokens,
                )
                guarded = fused_reference.generate_once(
                    model,
                    tokenizer,
                    system_prompt,
                    [],
                    row["user_text"],
                    args.max_new_tokens,
                    probe_module,
                    capture,
                )
            else:
                guarded = fused_reference.generate_once(
                    model,
                    tokenizer,
                    system_prompt,
                    [],
                    row["user_text"],
                    args.max_new_tokens,
                    probe_module,
                    capture,
                )
                normal = fused_reference.generate_once(
                    model,
                    tokenizer,
                    system_prompt,
                    [],
                    row["user_text"],
                    args.max_new_tokens,
                )

            if (
                normal["input_tokens"] != guarded["input_tokens"]
                or normal["output_tokens"] != guarded["output_tokens"]
                or normal["output_sha256"] != guarded["output_sha256"]
            ):
                raise RuntimeError(
                    f"Normal and NeuroFilter generation differ for {key}."
                )
            if capture.current is None:
                raise RuntimeError(f"No activation captured for {key}.")

            result = {
                "experiment": EXPERIMENT,
                "benchmark_signature": signature,
                "record_id": key,
                "profile_index": row["profile_index"],
                "template_index": row["template_index"],
                "trajectory_type": "benign",
                "normal_first": normal_first,
                "input_tokens": int(normal["input_tokens"]),
                "output_tokens": int(normal["output_tokens"]),
                "normal_latency_ms": float(normal["latency_ms"]),
                "neurofilter_latency_ms": float(guarded["latency_ms"]),
                "paired_added_latency_ms": float(
                    guarded["latency_ms"] - normal["latency_ms"]
                ),
                "probe_score": float(capture.score),
                "probe_layer": PROBE_LAYER,
                "output_sha256": normal["output_sha256"],
            }
            fused_reference.append_jsonl(raw_path, result)
            cache[key] = result
            print(
                f"Saved {key}: normal={normal['latency_ms']:.2f} ms, "
                f"NeuroFilter={guarded['latency_ms']:.2f} ms, "
                f"added={result['paired_added_latency_ms']:.2f} ms"
            )
    finally:
        del model, tokenizer
        torch.cuda.empty_cache()

    if set(cache) != expected_ids:
        raise RuntimeError(
            f"Run ended with {len(cache)}/{len(expected_ids)} paired measurements."
        )
    summarize(raw_path, args.output_dir, signature)
    (args.output_dir / ".complete").write_text("complete\n", encoding="utf-8")
    print(f"Corrected single-turn latency benchmark complete: {args.output_dir}")


if __name__ == "__main__":
    main()
