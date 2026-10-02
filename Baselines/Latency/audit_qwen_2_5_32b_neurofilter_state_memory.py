#!/usr/bin/env python3
"""Measure Qwen 2.5 32B NeuroFilter runtime-state memory exactly.

This reproduces the NF4 model loading and FP32 activation/probe handling used
by benchmark_filter_response_latency.py. It makes no API calls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/path/to/code")
MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"
CHECKPOINT = (
    ROOT
    / "dataframes/trajectoryprobe_final_probe_checkpoint_"
    "Qwen2_5_32B_Instruct_insurance_cmpl80_v1.npz"
)
DEFAULT_OUTPUT = (
    ROOT
    / "logs/"
    "qwen_2_5_32b_neurofilter_state_memory_audit_v1/report.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def tensor_report(tensor: Any) -> dict[str, Any]:
    logical_bytes = int(tensor.numel() * tensor.element_size())
    storage_bytes = int(tensor.untyped_storage().nbytes())
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": int(tensor.numel()),
        "element_size_bytes": int(tensor.element_size()),
        "logical_bytes": logical_bytes,
        "storage_bytes": storage_bytes,
        "logical_KiB": logical_bytes / 1024.0,
        "storage_KiB": storage_bytes / 1024.0,
    }


def main() -> None:
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this NF4 runtime audit.")
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Missing probe checkpoint: {CHECKPOINT}")

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.eval()

    loaded_in_4bit = bool(getattr(model, "is_loaded_in_4bit", False))
    params4bit = sum(
        1 for parameter in model.parameters()
        if parameter.__class__.__name__ == "Params4bit"
    )
    if not loaded_in_4bit or params4bit == 0:
        raise RuntimeError(
            "The model did not load with bitsandbytes 4-bit parameters as required."
        )

    checkpoint = np.load(CHECKPOINT, allow_pickle=False)
    layer = int(checkpoint["best_layer_index"][0])
    checkpoint_weights = checkpoint["best_probe_weights"]
    if checkpoint_weights.shape != (5120,):
        raise RuntimeError(f"Unexpected probe shape: {checkpoint_weights.shape}")

    # Match load_multi_probe(): the deployed latency benchmark casts the saved
    # checkpoint weights to FP32 before moving them to the model device.
    runtime_weights_np = np.asarray(checkpoint_weights, dtype=np.float32)
    input_device = model.get_input_embeddings().weight.device
    runtime_weights = torch.from_numpy(runtime_weights_np).to(input_device)

    rendered = tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": "You are an insurance assistant. Protect private information.",
            },
            {"role": "user", "content": "Hello."},
        ],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer(
        rendered,
        return_tensors="pt",
        add_special_tokens=False,
    )
    inputs = {key: value.to(input_device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model.model(**inputs, use_cache=False, return_dict=True)

    # Raw final-token activation produced by the NF4/BF16 model.
    raw_activation = outputs.last_hidden_state[0, -1, :].detach()

    # Match final_activations(): the deployed latency code explicitly calls
    # .float(), so the previous-turn cache is FP32 despite NF4 model weights.
    cached_activation = raw_activation.float().contiguous()
    if cached_activation.shape != runtime_weights.shape:
        raise RuntimeError(
            f"Activation/probe mismatch: {cached_activation.shape}, "
            f"{runtime_weights.shape}"
        )

    raw_report = tensor_report(raw_activation)
    cache_report = tensor_report(cached_activation)
    weights_report = tensor_report(runtime_weights)
    tensor_state_bytes = (
        cache_report["storage_bytes"] + weights_report["storage_bytes"]
    )

    report = {
        "model_id": MODEL_ID,
        "model_loaded_in_4bit": loaded_in_4bit,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": str(torch.bfloat16),
        "number_of_Params4bit_tensors": params4bit,
        "checkpoint": str(CHECKPOINT),
        "probe_layer": layer,
        "checkpoint_probe_dtype": str(checkpoint_weights.dtype),
        "checkpoint_probe_bytes": int(checkpoint_weights.nbytes),
        "raw_model_activation": raw_report,
        "cached_previous_activation_current_latency_code": cache_report,
        "runtime_probe_weights_current_latency_code": weights_report,
        "runtime_tensor_state_total_bytes": tensor_state_bytes,
        "runtime_tensor_state_total_KiB": tensor_state_bytes / 1024.0,
        "scope": (
            "Additional NeuroFilter state only: one selected-layer previous-turn "
            "activation plus one probe-weight vector. Model weights and the "
            "ordinary transformer KV cache are excluded."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Model loaded in NF4: {loaded_in_4bit} ({params4bit} Params4bit tensors)")
    print(
        "Raw model activation: "
        f"{raw_report['dtype']}, {raw_report['storage_bytes']} bytes "
        f"({raw_report['storage_KiB']:.3f} KiB)"
    )
    print(
        "Cached previous activation used by current latency code: "
        f"{cache_report['dtype']}, {cache_report['storage_bytes']} bytes "
        f"({cache_report['storage_KiB']:.3f} KiB)"
    )
    print(
        "Runtime probe weights used by current latency code: "
        f"{weights_report['dtype']}, {weights_report['storage_bytes']} bytes "
        f"({weights_report['storage_KiB']:.3f} KiB)"
    )
    print(
        "Total additional runtime tensor state: "
        f"{tensor_state_bytes} bytes ({tensor_state_bytes / 1024.0:.3f} KiB)"
    )
    print(f"Saved report: {args.output}")


if __name__ == "__main__":
    main()
