#!/usr/bin/env python3
"""Measure NeuroFilter overhead while reusing Qwen's normal activations.

This local-only benchmark replays the saved benign CMPL Insurance test
conversations.  Each request is generated twice: once normally and once with a
temporary hook on the trained probe layer.  The hook reads that layer's final
prompt-token activation, computes the activation delta and logistic-probe
decision, and then removes itself before token decoding continues.  It never
runs a separate model forward for NeuroFilter.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch

import benchmark_qwen_2_5_32b_neurofilter_vs_llamaguard4_latency as reference


ROOT = Path("/path/to/code")
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "logs/"
    "qwen_2_5_32b_neurofilter_fused_allowed_latency_v1"
)
EXPERIMENT = "qwen_2_5_32b_neurofilter_fused_allowed_latency_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument(
        "--limit-conversations",
        type=int,
        default=None,
        help="Optional smoke-test limit; the default uses all 20 benign conversations.",
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive.")
    if args.warmup_requests < 0:
        parser.error("--warmup-requests must not be negative.")
    if args.limit_conversations is not None and args.limit_conversations < 1:
        parser.error("--limit-conversations must be positive.")
    return args


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_benign_records(limit: int | None) -> list[dict[str, Any]]:
    # Validate the complete paper test set before selecting its benign half.
    records = reference.validate_and_load_trajectories(
        reference.TRAJECTORY_FILE, None
    )
    benign = [row for row in records if row["trajectory_type"] == "benign"]
    if len(benign) != 20:
        raise RuntimeError(f"Expected 20 benign conversations, found {len(benign)}.")
    for record in benign:
        if len(record["turn_rows"]) != 20:
            raise RuntimeError(
                f"Expected 20 turns for {record.get('sample_id')}, "
                f"found {len(record['turn_rows'])}."
            )
    return benign if limit is None else benign[:limit]


def benchmark_signature(args: argparse.Namespace, probe: dict[str, Any]) -> str:
    payload = {
        "experiment": EXPERIMENT,
        "trajectory_sha256": reference.sha256_file(reference.TRAJECTORY_FILE),
        "probe_sha256": probe["sha256"],
        "model": reference.QWEN_MODEL_ID,
        "probe_layer": int(probe["layer"]),
        "probe_threshold": float(probe["threshold"]),
        "max_new_tokens": int(args.max_new_tokens),
        "limit_conversations": args.limit_conversations,
        "measurement": "paired_normal_vs_same_generation_with_layer_hook",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def selected_probe_module(model: Any, layer: int) -> Any:
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    final_norm = getattr(backbone, "norm", None)
    if layers is None or final_norm is None:
        raise RuntimeError("Could not locate Qwen decoder layers and final norm.")
    if layer != len(layers):
        raise RuntimeError(
            f"Probe layer {layer} is not the final hidden state after "
            f"Qwen's {len(layers)} decoder layers."
        )
    return final_norm


def tensor_from_hook_output(output: Any) -> torch.Tensor:
    tensor = output[0] if isinstance(output, (tuple, list)) else output
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
        raise RuntimeError(
            f"Unexpected probe-layer hook output: {type(output)!r}."
        )
    return tensor


class ProbeCapture:
    def __init__(
        self,
        weights: torch.Tensor,
        threshold: float,
        previous: torch.Tensor | None,
        cumulative: float,
    ) -> None:
        self.weights = weights
        self.threshold = float(threshold)
        self.previous = previous
        self.cumulative_before = float(cumulative)
        self.current: torch.Tensor | None = None
        self.projection = 0.0
        self.cumulative = float(cumulative)
        self.flagged = False
        self.decision_latency_ms = 0.0

    def __call__(self, _module: Any, _inputs: Any, output: Any) -> None:
        if self.current is not None:
            return
        started = time.perf_counter()
        current = tensor_from_hook_output(output)[0, -1, :].detach().float()
        if current.shape != self.weights.shape:
            raise RuntimeError(
                f"Activation shape {tuple(current.shape)} does not match probe "
                f"shape {tuple(self.weights.shape)}."
            )
        if current.device != self.weights.device:
            raise RuntimeError(
                f"Activation is on {current.device}, but probe is on "
                f"{self.weights.device}."
            )
        self.current = current
        if self.previous is not None:
            previous = self.previous.to(current.device, non_blocking=True)
            projection_tensor = torch.dot(current - previous, self.weights)
            cumulative_tensor = projection_tensor + self.cumulative_before
            # A deployed filter must obtain this decision before allowing output.
            values = torch.stack((projection_tensor, cumulative_tensor)).cpu().numpy()
            self.projection = float(values[0])
            self.cumulative = float(values[1])
            self.flagged = self.cumulative > self.threshold
        self.decision_latency_ms = (time.perf_counter() - started) * 1000.0


def install_one_shot_hook(module: Any, capture: ProbeCapture) -> Any:
    handle_box: dict[str, Any] = {}

    def hook(inner_module: Any, inputs: Any, output: Any) -> None:
        capture(inner_module, inputs, output)
        handle = handle_box.get("handle")
        if handle is not None:
            handle.remove()
            handle_box["handle"] = None

    handle = module.register_forward_hook(hook)
    handle_box["handle"] = handle
    return handle_box


def generate_once(
    model: Any,
    tokenizer: Any,
    system_prompt: str,
    history: list[dict[str, str]],
    current_user: str,
    max_new_tokens: int,
    probe_module: Any | None = None,
    capture: ProbeCapture | None = None,
) -> dict[str, Any]:
    if (probe_module is None) != (capture is None):
        raise ValueError("probe_module and capture must be supplied together.")
    handle_box: dict[str, Any] | None = None
    if capture is not None:
        handle_box = install_one_shot_hook(probe_module, capture)

    devices = reference.model_cuda_devices(model)
    reference.synchronize(devices)
    started = time.perf_counter()
    try:
        input_ids, inputs = reference.qwen_inputs(
            tokenizer,
            system_prompt,
            history,
            current_user,
            reference.model_input_device(model),
        )
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        input_tokens = int(input_ids.shape[-1])
        generated_ids = generated[0, input_ids.shape[-1] :]
        output_tokens = int(generated_ids.shape[-1])
        output_text = tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()
        reference.synchronize(devices)
        latency_ms = (time.perf_counter() - started) * 1000.0
    finally:
        if handle_box is not None and handle_box.get("handle") is not None:
            handle_box["handle"].remove()

    if capture is not None and capture.current is None:
        raise RuntimeError("The probe-layer hook did not observe Qwen's prompt pass.")
    return {
        "latency_ms": float(latency_ms),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "output_sha256": hashlib.sha256(output_text.encode("utf-8")).hexdigest(),
    }


def validate_hook_mapping(
    model: Any,
    tokenizer: Any,
    probe_module: Any,
    layer: int,
) -> float:
    captured: dict[str, torch.Tensor] = {}

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        captured["activation"] = tensor_from_hook_output(output)[0, -1, :].detach()

    handle = probe_module.register_forward_hook(hook)
    input_ids, inputs = reference.qwen_inputs(
        tokenizer,
        "Safety test.",
        [],
        "Hello.",
        reference.model_input_device(model),
    )
    try:
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
    finally:
        handle.remove()
    hooked = captured.get("activation")
    if hooked is None:
        raise RuntimeError("Probe-layer validation hook did not run.")
    expected = outputs.hidden_states[layer][0, -1, :]
    max_difference = float((hooked.float() - expected.float()).abs().max().item())
    if max_difference > 1e-5:
        raise RuntimeError(
            f"Hook does not match hidden_states[{layer}]: "
            f"maximum difference={max_difference}."
        )
    del outputs, inputs, input_ids, hooked, expected
    torch.cuda.empty_cache()
    return max_difference


def encode_activation(activation: torch.Tensor) -> str:
    values = activation.detach().cpu().numpy().astype(np.float32, copy=False)
    return base64.b64encode(values.tobytes()).decode("ascii")


def decode_activation(value: str, expected_size: int, device: torch.device) -> torch.Tensor:
    array = np.frombuffer(base64.b64decode(value.encode("ascii")), dtype=np.float32)
    if array.size != expected_size:
        raise RuntimeError(
            f"Cached activation has {array.size} values; expected {expected_size}."
        )
    return torch.from_numpy(array.copy()).to(device)


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


def summarize(path: Path, output_dir: Path, signature: str) -> None:
    rows = list(load_cache(path, signature).values())
    frame = pd.DataFrame(rows).drop(columns=["activation_f32_base64"])
    frame.to_csv(output_dir / "paired_latency_measurements.csv", index=False)
    allowed = frame[~frame["flagged"].astype(bool)].copy()
    allowed["added_latency_ms"] = (
        allowed["neurofilter_latency_ms"] - allowed["normal_latency_ms"]
    )
    allowed["overhead_percent"] = (
        100.0 * allowed["added_latency_ms"] / allowed["normal_latency_ms"]
    )
    allowed.to_csv(output_dir / "allowed_paired_latency.csv", index=False)

    def stats(name: str, values: pd.Series) -> dict[str, Any]:
        return {
            "measurement": name,
            "requests": int(len(values)),
            "mean_ms": float(values.mean()),
            "median_ms": float(values.median()),
            "p95_ms": float(values.quantile(0.95)),
        }

    summary_rows = [
        stats("normal_qwen_end_to_end", allowed["normal_latency_ms"]),
        stats("qwen_with_neurofilter_end_to_end", allowed["neurofilter_latency_ms"]),
        stats("paired_neurofilter_added_latency", allowed["added_latency_ms"]),
        stats("probe_decision_inside_forward", allowed["probe_decision_latency_ms"]),
    ]
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "allowed_latency_summary.csv", index=False)
    atomic_write_json(
        output_dir / "allowed_latency_summary.json",
        {
            "benign_requests_measured": int(len(frame)),
            "allowed_requests": int(len(allowed)),
            "benign_requests_flagged": int(frame["flagged"].sum()),
            "mean_paired_overhead_percent": float(allowed["overhead_percent"].mean()),
            "median_paired_overhead_percent": float(allowed["overhead_percent"].median()),
            "measurements": summary_rows,
        },
    )
    print("\nAllowed benign latency summary")
    print(summary.to_string(index=False))
    print(
        f"Mean paired overhead: {allowed['added_latency_ms'].mean():.3f} ms "
        f"({allowed['overhead_percent'].mean():.4f}%)"
    )


def main() -> None:
    args = parse_args()
    records = load_benign_records(args.limit_conversations)
    probe = reference.load_probe(reference.PROBE_CHECKPOINT)
    signature = benchmark_signature(args, probe)
    expected_ids = {
        reference.record_id(record, int(turn["turn_index"]))
        for record in records
        for turn in record["turn_rows"]
    }
    raw_path = args.output_dir / "raw_paired.jsonl"
    cache = load_cache(raw_path, signature)
    unexpected = set(cache) - expected_ids
    if unexpected:
        raise RuntimeError(f"Unexpected cached rows: {sorted(unexpected)[:5]}")

    print(f"Saved trajectories: {reference.TRAJECTORY_FILE}")
    print(f"Probe checkpoint: {reference.PROBE_CHECKPOINT}")
    print(f"Probe layer: {probe['layer']}; threshold: {probe['threshold']:.6f}")
    print(f"Benign conversations: {len(records)}; requests: {len(expected_ids)}")
    print(f"Existing paired measurements: {len(cache)}/{len(expected_ids)}")
    print("NeuroFilter forward passes added: 0")
    print("Network/API access: disabled")
    reference.validate_local_model_files(reference.QWEN_MODEL_ID)
    if args.validate_only:
        print("Validation complete. No model was loaded and no output was changed.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        args.output_dir / "manifest.json",
        {
            "experiment": EXPERIMENT,
            "benchmark_signature": signature,
            "trajectory_file": str(reference.TRAJECTORY_FILE),
            "trajectory_sha256": reference.sha256_file(reference.TRAJECTORY_FILE),
            "probe_checkpoint": str(reference.PROBE_CHECKPOINT),
            "probe_sha256": probe["sha256"],
            "model": reference.QWEN_MODEL_ID,
            "probe_layer": int(probe["layer"]),
            "probe_threshold": float(probe["threshold"]),
            "conversations": len(records),
            "requests": len(expected_ids),
            "max_new_tokens": int(args.max_new_tokens),
            "measurement": "paired normal generation versus the same generation with a one-shot probe-layer hook",
            "extra_neurofilter_model_forwards": 0,
            "network_api_access": False,
        },
    )

    model = None
    tokenizer = None
    try:
        model, tokenizer, load_seconds = reference.load_qwen()
        print(f"Loaded {reference.QWEN_MODEL_ID} in {load_seconds:.2f} seconds.")
        probe_module = selected_probe_module(model, int(probe["layer"]))
        module_device = next(probe_module.parameters()).device
        weights = torch.as_tensor(
            probe["weights"], dtype=torch.float32, device=module_device
        )
        difference = validate_hook_mapping(
            model, tokenizer, probe_module, int(probe["layer"])
        )
        print(
            f"Validated hook equals hidden_states[{probe['layer']}]: "
            f"max difference={difference:.8f}"
        )

        # Warm-up is outside every recorded timing.
        warmups = min(args.warmup_requests, len(records[0]["turn_rows"]))
        for index in range(warmups):
            turn_row = records[0]["turn_rows"][index]
            history = reference.fixed_history(records[0]["turn_rows"], index)
            system_prompt = reference.build_system_prompt(records[0]["patient_profile"])
            generate_once(
                model,
                tokenizer,
                system_prompt,
                history,
                str(turn_row["user_text"]),
                min(8, args.max_new_tokens),
            )
            warm_capture = ProbeCapture(weights, probe["threshold"], None, 0.0)
            generate_once(
                model,
                tokenizer,
                system_prompt,
                history,
                str(turn_row["user_text"]),
                min(8, args.max_new_tokens),
                probe_module,
                warm_capture,
            )
        print(f"Completed {warmups} untimed warm-up request pairs.")

        for record in records:
            system_prompt = reference.build_system_prompt(record["patient_profile"])
            previous: torch.Tensor | None = None
            cumulative = 0.0
            for index, turn_row in enumerate(record["turn_rows"]):
                turn = int(turn_row["turn_index"])
                key = reference.record_id(record, turn)
                if key in cache:
                    cached = cache[key]
                    previous = decode_activation(
                        str(cached["activation_f32_base64"]),
                        weights.numel(),
                        module_device,
                    )
                    cumulative = float(cached["cumulative_score"])
                    continue

                history = reference.fixed_history(record["turn_rows"], index)
                current_user = str(turn_row["user_text"])
                capture = ProbeCapture(
                    weights, probe["threshold"], previous, cumulative
                )

                # Alternate order so allocator/cache effects do not consistently
                # favor either the normal or NeuroFilter path.
                normal_first = (int(record["person_index"]) + turn) % 2 == 0
                if normal_first:
                    normal = generate_once(
                        model,
                        tokenizer,
                        system_prompt,
                        history,
                        current_user,
                        args.max_new_tokens,
                    )
                    guarded = generate_once(
                        model,
                        tokenizer,
                        system_prompt,
                        history,
                        current_user,
                        args.max_new_tokens,
                        probe_module,
                        capture,
                    )
                else:
                    guarded = generate_once(
                        model,
                        tokenizer,
                        system_prompt,
                        history,
                        current_user,
                        args.max_new_tokens,
                        probe_module,
                        capture,
                    )
                    normal = generate_once(
                        model,
                        tokenizer,
                        system_prompt,
                        history,
                        current_user,
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

                previous = capture.current
                cumulative = capture.cumulative
                row = {
                    "experiment": EXPERIMENT,
                    "benchmark_signature": signature,
                    "record_id": key,
                    "sample_id": record.get("sample_id"),
                    "person_index": int(record["person_index"]),
                    "trajectory_type": "benign",
                    "turn": turn,
                    "normal_first": normal_first,
                    "input_tokens": int(normal["input_tokens"]),
                    "output_tokens": int(normal["output_tokens"]),
                    "normal_latency_ms": float(normal["latency_ms"]),
                    "neurofilter_latency_ms": float(guarded["latency_ms"]),
                    "paired_added_latency_ms": float(
                        guarded["latency_ms"] - normal["latency_ms"]
                    ),
                    "probe_decision_latency_ms": float(capture.decision_latency_ms),
                    "projection_score": float(capture.projection),
                    "cumulative_score": float(capture.cumulative),
                    "threshold": float(probe["threshold"]),
                    "flagged": bool(capture.flagged),
                    "probe_layer": int(probe["layer"]),
                    "output_sha256": normal["output_sha256"],
                    "activation_f32_base64": encode_activation(capture.current),
                }
                append_jsonl(raw_path, row)
                cache[key] = row
                print(
                    f"Saved {key}: normal={normal['latency_ms']:.2f} ms, "
                    f"NeuroFilter={guarded['latency_ms']:.2f} ms, "
                    f"added={row['paired_added_latency_ms']:.2f} ms, "
                    f"flagged={capture.flagged}"
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
    print(f"Corrected NeuroFilter latency benchmark complete: {args.output_dir}")


if __name__ == "__main__":
    main()
