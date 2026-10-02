from __future__ import annotations

import argparse
import asyncio
import copy
import gc
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/path/to/code")
EXPERIMENT_VERSION = "v1"
EXTENSION_CACHE_SUFFIX = "cmpl80_v1"


def _load_module(script_path: Path) -> ModuleType:
    name = f"_cmpl_count_baseline_{script_path.stem}"
    spec = importlib.util.spec_from_file_location(name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load baseline script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FailoverClient:
    def __init__(self, module: ModuleType, primary: Any) -> None:
        self._module = module
        self._primary_config = primary
        self._primary = module.OpenAI(api_key=primary.api_key, base_url=primary.endpoint)
        endpoint = os.environ.get(
            "CMPL_OPENAI_ENDPOINT_FALLBACK",
            os.environ.get("ATTACKER_OPENAI_ENDPOINT_FALLBACK"),
        )
        api_key = os.environ.get(
            "CMPL_OPENAI_API_KEY_FALLBACK",
            os.environ.get("ATTACKER_OPENAI_API_KEY_FALLBACK"),
        )
        model = os.environ.get(
            "CMPL_OPENAI_MODEL_FALLBACK",
            os.environ.get("ATTACKER_OPENAI_MODEL_FALLBACK", primary.model),
        )
        if bool(endpoint) != bool(api_key):
            raise KeyError("Fallback configuration requires both endpoint and API key.")
        self._fallback_config = None
        self._fallback = None
        if endpoint and api_key:
            normalized_model = module.normalize_model_for_endpoint(endpoint, model)
            self._fallback_config = module.EndpointConfig(endpoint, api_key, normalized_model)
            self._fallback = module.OpenAI(api_key=api_key, base_url=endpoint)
        self._using_fallback = False
        self.chat = self
        self.completions = self

    def create(self, **kwargs: Any) -> Any:
        if self._using_fallback:
            return self._call(self._fallback, self._fallback_config, kwargs)
        try:
            return self._call(self._primary, self._primary_config, kwargs)
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            retryable = type(exc).__name__ in {
                "RateLimitError",
                "APIConnectionError",
                "APITimeoutError",
            } or (isinstance(status, int) and status >= 500)
            if self._fallback is None or not retryable:
                raise
            self._using_fallback = True
            print(
                f"Primary API failed with {type(exc).__name__}; switching to fallback endpoint "
                f"{self._fallback_config.endpoint} for the remainder of this process."
            )
            return self._call(self._fallback, self._fallback_config, kwargs)

    @staticmethod
    def _call(client: Any, config: Any, kwargs: dict[str, Any]) -> Any:
        if client is None or config is None:
            raise RuntimeError("Fallback API client is not configured.")
        routed = dict(kwargs)
        routed["model"] = config.model
        return client.chat.completions.create(**routed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-count", type=int, choices=(40, 80), required=True)
    parser.add_argument("--stage", choices=("training", "live", "all"), default="all")
    parser.add_argument("--profiles-file", default=str(ROOT / "benchmark/datasets/generated_profiles_scenario_2.json"))
    parser.add_argument("--scenario-file", default=str(ROOT / "benchmark/scenarios_descriptions.yaml"))
    parser.add_argument("--scenario-index", type=int, default=2)
    parser.add_argument("--train-max-turns", type=int, default=20)
    parser.add_argument("--online-max-turns", type=int, default=20)
    parser.add_argument("--target-attribute", default="family medical history")
    parser.add_argument("--artifact-log-dir", default=str(ROOT / "logs"))
    parser.add_argument("--baseline-training-cache", default="")
    parser.add_argument("--repeated-splits", type=int, default=10)
    parser.add_argument("--validation-fraction", type=float, default=0.30)
    parser.add_argument("--cmpl-temperature", type=float, default=0.85)
    parser.add_argument("--cmpl-top-p", type=float, default=0.90)
    parser.add_argument("--cmpl-max-tokens", type=int, default=1024)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--api-max-retries", type=int, default=3)
    parser.add_argument("--api-retry-delay-sec", type=float, default=5.0)
    parser.add_argument("--force-retrain-probe", action="store_true")
    parser.add_argument("--force-regenerate-online-trajectories", action="store_true")
    parser.add_argument(
        "--live-threshold",
        choices=("checkpoint", "zero"),
        default="checkpoint",
        help="Use the saved CV threshold or threshold 0 during live testing.",
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.repeated_splits < 2:
        parser.error("--repeated-splits must be at least 2")
    if not 0.0 < args.validation_fraction < 0.5:
        parser.error("--validation-fraction must be between 0 and 0.5")
    return args


def experiment_suffix(count: int) -> str:
    return f"cmpl{count}_{EXPERIMENT_VERSION}"


def train_subject_ids(count: int) -> list[int]:
    return list(range(0, 20)) if count == 40 else list(range(0, 20)) + list(range(40, 60))


def test_subject_ids(count: int) -> list[int]:
    return list(range(20, 40)) if count == 40 else list(range(20, 40)) + list(range(60, 80))


def artifact_paths(
    module: ModuleType,
    count: int,
    artifact_log_dir: str | Path,
    live_threshold: str = "checkpoint",
) -> dict[str, Path]:
    suffix = experiment_suffix(count)
    slug = module.MODEL_ID.split("/")[-1].replace(".", "_").replace("-", "_")
    test_tag = "20_39" if count == 40 else "20_39_60_79"
    live_tag = "_threshold0" if live_threshold == "zero" else ""
    log_dir = Path(artifact_log_dir)
    return {
        "activation_dir": ROOT / f"temp_stateful_deltas_{slug}_insurance_kfold_live_cmpl_train",
        "extension_cache": log_dir / f"trajectoryprobe_live_cmpl_training_cache_{slug}_insurance_{EXTENSION_CACHE_SUFFIX}.jsonl",
        "extension_trace": log_dir / f"trajectoryprobe_live_cmpl_training_trace_{slug}_insurance_{EXTENSION_CACHE_SUFFIX}.jsonl",
        "checkpoint": ROOT / "dataframes" / f"trajectoryprobe_final_probe_checkpoint_{slug}_insurance_{suffix}.npz",
        "heldout_scores": ROOT / "dataframes" / f"trajectoryprobe_repeated_split_heldout_scores_{slug}_insurance_{suffix}.csv",
        "thresholds": ROOT / "dataframes" / f"trajectoryprobe_repeated_split_thresholds_{slug}_insurance_{suffix}.csv",
        "split_summary": ROOT / "dataframes" / f"trajectoryprobe_repeated_split_summary_{slug}_insurance_{suffix}.csv",
        "online_output": log_dir / f"cmpl_online_guarded_{slug}_subjects_{test_tag}_{suffix}{live_tag}.jsonl",
        "turn_csv": ROOT / "dataframes" / f"cmpl_online_guarded_turns_{slug}_subjects_{test_tag}_{suffix}{live_tag}.csv",
    }


def _read_jsonl(module: ModuleType, path: Path) -> list[dict[str, Any]]:
    return module.read_jsonl(str(path)) if path.exists() else []


def _seed_extension_cache(
    module: ModuleType,
    paths: dict[str, Path],
    baseline_training_cache: str,
) -> None:
    paths["extension_cache"].parent.mkdir(parents=True, exist_ok=True)
    existing = _read_jsonl(module, paths["extension_cache"])
    baseline_path = Path(baseline_training_cache) if baseline_training_cache else Path(module.LIVE_TRAINING_CACHE_PATH)
    baseline_records = _read_jsonl(module, baseline_path)
    expected_generators: dict[str, str] = {}
    for trajectory_type in ("attack", "benign"):
        generators = {
            str(record.get("generator"))
            for record in baseline_records
            if int(record.get("person_index", -1)) in range(0, 20)
            and str(record.get("trajectory_type")) == trajectory_type
        }
        if len(generators) != 1:
            raise RuntimeError(
                f"Expected exactly one {trajectory_type} generator in baseline cache "
                f"{baseline_path}; found {sorted(generators)}"
            )
        expected_generators[trajectory_type] = generators.pop()
    incompatible_existing = [
        record for record in existing
        if record.get("generator") != expected_generators.get(str(record.get("trajectory_type")))
    ]
    if incompatible_existing:
        raise RuntimeError(
            f"The isolated extension cache contains {len(incompatible_existing)} incompatible API-target record(s): "
            f"{paths['extension_cache']}"
        )
    existing_keys = module.completed_record_keys(existing)
    copied = 0
    for record in baseline_records:
        key = (int(record.get("person_index", -1)), str(record.get("trajectory_type")))
        expected_generator = expected_generators.get(key[1])
        if key[0] not in range(0, 20) or key in existing_keys or record.get("generator") != expected_generator:
            continue
        module.append_jsonl(str(paths["extension_cache"]), record)
        existing_keys.add(key)
        copied += 1
    if copied:
        print(f"Copied {copied} baseline record(s) into the isolated extension cache; baseline remained read-only.")


def _generation_namespace(args: argparse.Namespace, paths: dict[str, Path], start: int, end: int) -> argparse.Namespace:
    ns = copy.copy(args)
    ns.train_subject_start = start
    ns.train_subject_end = end
    ns.cmpl_training_cache = str(paths["extension_cache"])
    ns.cmpl_training_trace_log = str(paths["extension_trace"])
    ns.force_regenerate_training_trajectories = False
    ns.reuse_cmpl_training_trajectories = False
    return ns


def ensure_training_trajectories(
    module: ModuleType,
    args: argparse.Namespace,
    paths: dict[str, Path],
    model: Any,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    _seed_extension_cache(module, paths, args.baseline_training_cache)
    # Use the baseline model script's own tested API/failover implementation.
    # The newer CMPL scripts pass both primary and fallback configurations.
    module.generate_live_training_records(_generation_namespace(args, paths, 0, 19), model, tokenizer)
    if args.trajectory_count == 80:
        module.generate_live_training_records(_generation_namespace(args, paths, 40, 59), model, tokenizer)

    required_ids = set(train_subject_ids(args.trajectory_count))
    records = [
        record
        for record in _read_jsonl(module, paths["extension_cache"])
        if int(record.get("person_index", -1)) in required_ids
    ]
    keys = module.completed_record_keys(records)
    required_keys = {(idx, kind) for idx in required_ids for kind in ("attack", "benign")}
    missing = sorted(required_keys - keys)
    if missing:
        raise RuntimeError(f"Training cache is incomplete after generation: {missing[:20]}")
    key_counts = Counter(
        (int(record["person_index"]), str(record["trajectory_type"]))
        for record in records
    )
    duplicates = sorted(key for key, value in key_counts.items() if value != 1)
    if duplicates:
        raise RuntimeError(f"Training cache has duplicate subject/class records: {duplicates[:20]}")
    attack_count = sum(record["trajectory_type"] == "attack" for record in records)
    benign_count = sum(record["trajectory_type"] == "benign" for record in records)
    expected_per_class = len(required_ids)
    if attack_count != expected_per_class or benign_count != expected_per_class:
        raise RuntimeError(
            "Training cache has the wrong class counts: "
            f"attack={attack_count}, benign={benign_count}, expected={expected_per_class} each"
        )
    print(
        f"Validated final-probe training set: {attack_count} attack + "
        f"{benign_count} benign conversations."
    )
    return sorted(records, key=lambda row: (int(row["person_index"]), str(row["trajectory_type"])))


def build_samples(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for record in records:
        samples.append(
            {
                "person_index": int(record["person_index"]),
                "type": str(record["trajectory_type"]),
                "shots": list(record["user_turns"]),
                "profile_text": json.dumps(record["patient_profile"], indent=2),
            }
        )
    return samples


def cached_deltas(
    module: ModuleType,
    model: Any,
    tokenizer: Any,
    samples: list[dict[str, Any]],
    trajectory_type: str,
    activation_dir: Path,
    desc: str,
) -> np.ndarray:
    arrays: list[np.ndarray] = []
    activation_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        key = module.stable_sample_key(sample["shots"], sample["profile_text"], trajectory_type)
        path = activation_dir / f"{trajectory_type}_{key}.npy"
        if path.exists():
            try:
                array = np.load(path, allow_pickle=False)
            except Exception as exc:
                raise RuntimeError(f"Existing activation cache is unreadable and will not be overwritten: {path}") from exc
        else:
            array = module.get_or_generate_deltas(
                model,
                tokenizer,
                [(sample["shots"], sample["profile_text"])],
                "shared_cache",
                trajectory_type,
                str(activation_dir),
                desc,
            )
        expected = len(sample["shots"]) - 1
        if array.ndim != 3 or array.shape[0] != expected:
            raise RuntimeError(f"Unexpected activation cache shape for {path}: {array.shape}; expected {expected} turns")
        arrays.append(array)
    if not arrays:
        return np.array([])
    return np.concatenate(arrays, axis=0)


def analyze_samples(
    module: ModuleType,
    samples: list[dict[str, Any]],
    trajectory_type: str,
    activation_dir: Path,
    weights: np.ndarray,
    layer: int,
) -> list[dict[str, Any]]:
    items = [(sample["shots"], sample["profile_text"]) for sample in samples]
    return module.analyze_cumulative_from_cache(items, trajectory_type, str(activation_dir), weights, layer)


def repeated_group_splits(subjects: list[int], repeats: int, fraction: float, seed: int) -> list[tuple[set[int], set[int]]]:
    splits: list[tuple[set[int], set[int]]] = []
    count = max(1, int(round(len(subjects) * fraction)))
    ordered = np.array(sorted(subjects), dtype=int)
    for repeat in range(repeats):
        rng = np.random.default_rng(seed + repeat)
        validation = set(int(value) for value in rng.choice(ordered, size=count, replace=False))
        training = set(int(value) for value in ordered if int(value) not in validation)
        splits.append((training, validation))
    return splits


def threshold_search(module: ModuleType, heldout: pd.DataFrame, paths: dict[str, Path]) -> tuple[pd.DataFrame, float]:
    scores = heldout["Score"].to_numpy(dtype=float)
    lo = float(np.quantile(scores, module.THRESHOLD_RANGE_LO_Q))
    hi = float(np.quantile(scores, module.THRESHOLD_RANGE_HI_Q))
    pad = max(abs(hi - lo) * module.THRESHOLD_RANGE_PAD_FRAC, 1e-6)
    grid = np.unique(np.concatenate([np.linspace(lo - pad, hi + pad, module.THRESHOLD_GRID_POINTS), [0.0]]))
    rows: list[dict[str, Any]] = []
    for threshold in grid:
        repeat_metrics: list[float] = []
        row: dict[str, Any] = {"Threshold": float(threshold)}
        for repeat, frame in heldout.groupby("Repeat"):
            turn_acc: list[float] = []
            for _, turn_frame in frame.groupby("Turn"):
                truth = turn_frame["Type"].eq("attack").to_numpy()
                pred = turn_frame["Score"].gt(threshold).to_numpy()
                turn_acc.append(float(np.mean(truth == pred)))
            metric = float(np.mean(turn_acc))
            repeat_metrics.append(metric)
            row[f"Repeat{int(repeat)}"] = metric
        row["Mean"] = float(np.mean(repeat_metrics))
        row["Std"] = float(np.std(repeat_metrics))
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(["Mean", "Threshold"], ascending=[False, True])
    best_mean = float(table["Mean"].max())
    candidates = table[np.isclose(table["Mean"], best_mean)].copy()
    candidates["AbsThreshold"] = candidates["Threshold"].abs()
    threshold = float(candidates.sort_values(["AbsThreshold", "Threshold"]).iloc[0]["Threshold"])
    table.to_csv(paths["thresholds"], index=False)
    return table, threshold


def summarize_repeats(heldout: pd.DataFrame, threshold: float, paths: dict[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for repeat, frame in heldout.groupby("Repeat"):
        truth = frame["Type"].eq("attack")
        pred = frame["Score"].gt(threshold)
        attack = truth
        benign = ~truth
        rows.append(
            {
                "Repeat": int(repeat),
                "Threshold": threshold,
                "Accuracy": float(np.mean(truth == pred)),
                "AttackTPR": float(np.mean(pred[attack])),
                "BenignTNR": float(np.mean(~pred[benign])),
                "FalsePositiveRate": float(np.mean(pred[benign])),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(paths["split_summary"], index=False)
    return summary


def train_repeated_probe(
    module: ModuleType,
    args: argparse.Namespace,
    paths: dict[str, Path],
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
) -> tuple[float, int, np.ndarray, float]:
    samples = build_samples(records)
    subjects = sorted({sample["person_index"] for sample in samples})
    split_results: list[pd.DataFrame] = []
    layers: list[int] = []
    for repeat, (train_ids, validation_ids) in enumerate(
        repeated_group_splits(subjects, args.repeated_splits, args.validation_fraction, module.RANDOM_SEED),
        start=1,
    ):
        train = [sample for sample in samples if sample["person_index"] in train_ids]
        validation = [sample for sample in samples if sample["person_index"] in validation_ids]
        attack_train = [sample for sample in train if sample["type"] == "attack"]
        benign_train = [sample for sample in train if sample["type"] == "benign"]
        attack_x = cached_deltas(module, model, tokenizer, attack_train, "attack", paths["activation_dir"], f"Repeat {repeat} attack")
        benign_x = cached_deltas(module, model, tokenizer, benign_train, "benign", paths["activation_dir"], f"Repeat {repeat} benign")
        x = np.concatenate([attack_x, benign_x])
        y = np.concatenate([np.ones(len(attack_x)), np.zeros(len(benign_x))])
        layer, weights, accuracies = module.train_differential_probe(x, y)
        layers.append(int(layer))
        print(f"Repeat {repeat}: selected layer={layer}, train_acc={accuracies[layer]:.4f}")
        attack_val = [sample for sample in validation if sample["type"] == "attack"]
        benign_val = [sample for sample in validation if sample["type"] == "benign"]
        cached_deltas(module, model, tokenizer, attack_val, "attack", paths["activation_dir"], f"Repeat {repeat} attack validation")
        cached_deltas(module, model, tokenizer, benign_val, "benign", paths["activation_dir"], f"Repeat {repeat} benign validation")
        rows = analyze_samples(module, attack_val, "attack", paths["activation_dir"], weights, layer)
        rows += analyze_samples(module, benign_val, "benign", paths["activation_dir"], weights, layer)
        frame = pd.DataFrame(rows)
        frame["Repeat"] = repeat
        frame["BestLayer"] = layer
        split_results.append(frame)

    heldout = pd.concat(split_results, ignore_index=True)
    heldout.to_csv(paths["heldout_scores"], index=False)
    _, threshold = threshold_search(module, heldout, paths)
    summarize_repeats(heldout, threshold, paths)

    counts = Counter(layers)
    max_count = max(counts.values())
    selected_layer = max(layer for layer, count in counts.items() if count == max_count)
    attack_all = [sample for sample in samples if sample["type"] == "attack"]
    benign_all = [sample for sample in samples if sample["type"] == "benign"]
    expected_subjects = train_subject_ids(args.trajectory_count)
    if subjects != expected_subjects:
        raise RuntimeError(
            f"Final-probe subject mismatch: expected {expected_subjects}, found {subjects}"
        )
    if len(attack_all) != len(expected_subjects) or len(benign_all) != len(expected_subjects):
        raise RuntimeError(
            "Final probe must use one attack and one benign conversation per training subject; "
            f"found attack={len(attack_all)}, benign={len(benign_all)}"
        )
    print(
        f"Fitting final probe on all {len(attack_all)} attack + "
        f"{len(benign_all)} benign training conversations."
    )
    attack_x = cached_deltas(module, model, tokenizer, attack_all, "attack", paths["activation_dir"], "Final attack")
    benign_x = cached_deltas(module, model, tokenizer, benign_all, "benign", paths["activation_dir"], "Final benign")
    x = np.concatenate([attack_x, benign_x])
    y = np.concatenate([np.ones(len(attack_x)), np.zeros(len(benign_x))])
    probe = module.LogisticRegression(max_iter=1000, random_state=module.RANDOM_SEED, class_weight="balanced")
    probe.fit(x[:, selected_layer, :], y)
    weights = np.asarray(probe.coef_[0], dtype=float)
    accuracy = float(probe.score(x[:, selected_layer, :], y))
    np.savez(
        paths["checkpoint"],
        best_threshold=np.array([threshold]),
        best_layer_index=np.array([selected_layer]),
        best_probe_weights=weights,
        final_probe_acc=np.array([accuracy]),
        experiment_suffix=np.array([experiment_suffix(args.trajectory_count)]),
        repeated_splits=np.array([args.repeated_splits]),
        train_subject_ids=np.array(subjects),
        training_pipeline=np.array([module.TRAINING_PIPELINE_ID]),
    )
    print(f"Saved repeated-split checkpoint: {paths['checkpoint']}")
    return threshold, selected_layer, weights, accuracy


def load_checkpoint(
    module: ModuleType,
    paths: dict[str, Path],
    count: int,
) -> tuple[float, int, np.ndarray, float] | None:
    if not paths["checkpoint"].exists():
        return None
    checkpoint = np.load(paths["checkpoint"], allow_pickle=False)
    expected = experiment_suffix(count)
    actual = str(checkpoint["experiment_suffix"][0]) if "experiment_suffix" in checkpoint else ""
    if actual != expected:
        raise RuntimeError(f"Checkpoint namespace mismatch: expected {expected}, found {actual}")
    pipeline = str(checkpoint["training_pipeline"][0]) if "training_pipeline" in checkpoint else ""
    if pipeline != module.TRAINING_PIPELINE_ID:
        raise RuntimeError(f"Checkpoint training pipeline is incompatible: {pipeline!r}")
    expected_subjects = train_subject_ids(count)
    actual_subjects = (
        [int(value) for value in checkpoint["train_subject_ids"].tolist()]
        if "train_subject_ids" in checkpoint
        else []
    )
    if actual_subjects != expected_subjects:
        raise RuntimeError(
            f"Checkpoint training-subject mismatch: expected {expected_subjects}, found {actual_subjects}"
        )
    return (
        float(checkpoint["best_threshold"][0]),
        int(checkpoint["best_layer_index"][0]),
        np.asarray(checkpoint["best_probe_weights"], dtype=float),
        float(checkpoint["final_probe_acc"][0]),
    )


async def live_test(
    module: ModuleType,
    args: argparse.Namespace,
    paths: dict[str, Path],
    model: Any,
    tokenizer: Any,
    checkpoint: tuple[float, int, np.ndarray, float],
) -> None:
    checkpoint_threshold, layer, weights, _ = checkpoint
    threshold = 0.0 if args.live_threshold == "zero" else checkpoint_threshold
    print(
        f"Live threshold: {threshold:.6f} "
        f"({'threshold 0' if args.live_threshold == 'zero' else 'saved CV threshold'})"
    )
    profiles = module.load_profile_objects(args.profiles_file)
    wanted = set(test_subject_ids(args.trajectory_count))
    selected = [profile for profile in profiles if int(profile["patient_profile"]["person_index"]) in wanted]
    if {int(profile["patient_profile"]["person_index"]) for profile in selected} != wanted:
        raise RuntimeError("Test profiles are incomplete.")
    scenario = module.load_scenario_text(args.scenario_file, args.scenario_index)
    primary = module.env_config_with_fallback("CMPL_OPENAI", "ATTACKER_OPENAI", default_model=module.DEFAULT_CMPL_MODEL)
    client = FailoverClient(module, primary)
    if args.force_regenerate_online_trajectories and paths["online_output"].exists():
        paths["online_output"].unlink()
        print(f"Deleted only the isolated online output: {paths['online_output']}")
    records = _read_jsonl(module, paths["online_output"])
    expected_suffix = experiment_suffix(args.trajectory_count)
    incompatible = [
        record for record in records
        if record.get("experiment_suffix") != expected_suffix
        or record.get("training_pipeline") != module.TRAINING_PIPELINE_ID
    ]
    if incompatible:
        raise RuntimeError("Online output contains records from another experiment namespace.")
    keys = module.completed_record_keys(records)

    for profile in selected:
        person = int(profile["patient_profile"]["person_index"])
        for kind in ("attack", "benign"):
            if (person, kind) in keys:
                continue
            last_error: Exception | None = None
            for attempt in range(1, args.api_max_retries + 1):
                try:
                    if kind == "attack":
                        result = module.run_api_generated_guarded_attack_conversation(
                            profile=profile,
                            scenario=scenario,
                            model=model,
                            tokenizer=tokenizer,
                            client=client,
                            model_name=primary.model,
                            target_attribute=args.target_attribute,
                            threshold=threshold,
                            best_layer_index=layer,
                            best_probe_weights=weights,
                            max_turns=args.online_max_turns,
                            temperature=args.cmpl_temperature,
                            top_p=args.cmpl_top_p,
                            max_tokens=args.cmpl_max_tokens,
                        )
                    else:
                        result = module.run_api_generated_guarded_benign_conversation(
                            model=model,
                            tokenizer=tokenizer,
                            client=client,
                            model_name=primary.model,
                            profile=profile,
                            scenario=scenario,
                            threshold=threshold,
                            best_layer_index=layer,
                            best_probe_weights=weights,
                            max_turns=args.online_max_turns,
                            temperature=args.cmpl_temperature,
                            top_p=args.cmpl_top_p,
                            max_tokens=args.cmpl_max_tokens,
                        )
                    record = module.build_online_record(profile, kind, result, threshold, layer)
                    record["experiment_suffix"] = expected_suffix
                    module.append_jsonl(str(paths["online_output"]), record)
                    records.append(record)
                    keys.add((person, kind))
                    break
                except Exception as exc:
                    last_error = exc
                    print(f"person_index={person} {kind} attempt {attempt}/{args.api_max_retries} failed: {exc}")
                    if attempt < args.api_max_retries:
                        await asyncio.sleep(args.api_retry_delay_sec)
            if (person, kind) not in keys:
                raise RuntimeError(f"{kind} run failed for person_index={person}") from last_error

    frame = module.flatten_turn_rows(records)
    frame.to_csv(paths["turn_csv"], index=False)
    print(f"Saved isolated live-test records: {paths['online_output']}")
    print(f"Saved isolated per-turn table: {paths['turn_csv']}")


def validate_configuration(module: ModuleType, args: argparse.Namespace, paths: dict[str, Path]) -> None:
    print(f"Model: {module.MODEL_ID}")
    print(f"Experiment suffix: {experiment_suffix(args.trajectory_count)}")
    print(f"Training subjects: {train_subject_ids(args.trajectory_count)}")
    print(f"Test subjects: {test_subject_ids(args.trajectory_count)}")
    print(f"Live threshold mode: {args.live_threshold}")
    for name, path in paths.items():
        print(f"{name}: {path}")
    activation_dir = paths["activation_dir"].resolve()
    allowed_cache_root = Path("/path/to").resolve()
    if not activation_dir.is_relative_to(allowed_cache_root):
        raise RuntimeError(
            f"Activation cache must be under {allowed_cache_root}, found {activation_dir}"
        )
    baseline_outputs = {
        Path(module.PROBE_CHECKPOINT_PATH).resolve(),
        Path(module.LIVE_TRAINING_CACHE_PATH).resolve(),
    }
    writable = {path.resolve() for name, path in paths.items() if name != "activation_dir"}
    if baseline_outputs & writable:
        raise RuntimeError("New artifact paths overlap baseline paths.")


def run_count_experiment(baseline_script: str | Path) -> None:
    args = parse_args()
    module = _load_module(Path(baseline_script).resolve())
    paths = artifact_paths(
        module,
        args.trajectory_count,
        args.artifact_log_dir,
        args.live_threshold,
    )
    module.TEMP_DIR = str(paths["activation_dir"])
    validate_configuration(module, args, paths)
    if args.validate_only:
        print("Validation complete; no files were changed and no experiment was run.")
        return

    checkpoint = (
        None
        if args.force_retrain_probe
        else load_checkpoint(module, paths, args.trajectory_count)
    )
    model = tokenizer = None
    if args.stage in {"training", "all"}:
        if checkpoint is None:
            model, tokenizer = module.load_model_and_tokenizer()
            records = ensure_training_trajectories(module, args, paths, model, tokenizer)
            checkpoint = train_repeated_probe(module, args, paths, model, tokenizer, records)
        else:
            print(f"Using existing checkpoint: {paths['checkpoint']}")
        if args.stage == "training":
            if model is not None:
                module.unload_model_and_tokenizer(model, tokenizer)
            return

    if args.stage in {"live", "all"}:
        if checkpoint is None:
            checkpoint = load_checkpoint(module, paths, args.trajectory_count)
        if checkpoint is None:
            raise RuntimeError(f"No checkpoint exists for {experiment_suffix(args.trajectory_count)}; run --stage training first.")
        if model is None:
            model, tokenizer = module.load_model_and_tokenizer()
        asyncio.run(live_test(module, args, paths, model, tokenizer, checkpoint))
        module.unload_model_and_tokenizer(model, tokenizer)
        gc.collect()
