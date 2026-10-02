from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, ShuffleSplit


ROOT = Path("/path/to/code")
NOTEBOOKS = ROOT / "notebooks"
TRAINING_SUBJECTS = list(range(0, 20)) + list(range(40, 60))
EXPECTED_GENERATORS = {
    "attack": "live_cmpl_api_attacker_local_target",
    "benign": "live_cmpl_benign_api_user_local_target",
}
EXPERIMENT_ID = "cmpl80_offline_repeated_train_test_50_50_v1"


@dataclass(frozen=True)
class ModelConfig:
    key: str
    baseline_script: Path
    training_cache: Path
    activation_dir: Path


@dataclass(frozen=True)
class Sample:
    person_index: int
    kind: str
    path: Path
    turns: int


MODEL_CONFIGS = {
    "qwen_2_5_32b": ModelConfig(
        key="qwen_2_5_32b",
        baseline_script=NOTEBOOKS
        / "trajectoryprobe_cmpl_insurance_multiturn_w_acc_Qwen_2_5_32B_kfoldcrossval_cmpl_train_cmpltest.py",
        training_cache=ROOT
        / "logs/qwen_2_5_32b_40_to_80_v1/trajectoryprobe_live_cmpl_training_cache_Qwen2_5_32B_Instruct_insurance_cmpl80_v1.jsonl",
        activation_dir=ROOT
        / "temp_stateful_deltas_Qwen2_5_32B_Instruct_insurance_kfold_live_cmpl_train",
    ),
    "llama_3_3_70b": ModelConfig(
        key="llama_3_3_70b",
        baseline_script=NOTEBOOKS
        / "trajectoryprobe_cmpl_insurance_multiturn_w_acc_llama_3_3_70B_kfoldcrossval_cmpl_train_cmpltest.py",
        training_cache=ROOT
        / "logs/llama_3_3_70b_40_to_80_v1/trajectoryprobe_live_cmpl_training_cache_Llama_3_3_70B_Instruct_insurance_cmpl80_v1.jsonl",
        activation_dir=ROOT
        / "temp_stateful_deltas_Llama_3_3_70B_Instruct_insurance_kfold_live_cmpl_train",
    ),
    "gpt_oss_20b": ModelConfig(
        key="gpt_oss_20b",
        baseline_script=NOTEBOOKS
        / "trajectoryprobe_cmpl_insurance_multiturn_w_acc_gpt_oss_20B_kfoldcrossval_layerthreshold_cmpl_train_cmpltest.py",
        training_cache=ROOT
        / "logs/gpt_oss_20b_40_to_80_v1/trajectoryprobe_live_cmpl_training_cache_gpt_oss_20b_insurance_cmpl80_v1.jsonl",
        activation_dir=ROOT
        / "temp_stateful_deltas_gpt_oss_20b_insurance_kfold_live_cmpl_train",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline repeated 50:50 train/test evaluation on cached CMPL trajectories."
    )
    parser.add_argument("--model", choices=sorted(MODEL_CONFIGS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")
    if args.inner_folds < 2:
        parser.error("--inner-folds must be at least 2")
    return args


def load_module(config: ModelConfig) -> ModuleType:
    name = f"_offline_repeated_{config.key}"
    spec = importlib.util.spec_from_file_location(name, config.baseline_script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {config.baseline_script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_samples(module: ModuleType, config: ModelConfig) -> list[Sample]:
    records = read_jsonl(config.training_cache)
    if len(records) != 80:
        raise RuntimeError(
            f"Expected 80 cached trajectories, found {len(records)} in {config.training_cache}"
        )
    by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for record in records:
        person = int(record["person_index"])
        kind = str(record["trajectory_type"])
        key = (person, kind)
        if key in by_key:
            raise RuntimeError(f"Duplicate cached trajectory: {key}")
        if record.get("generator") != EXPECTED_GENERATORS.get(kind):
            raise RuntimeError(f"Incompatible cached trajectory generator for {key}")
        by_key[key] = record

    expected = {
        (person, kind)
        for person in TRAINING_SUBJECTS
        for kind in ("attack", "benign")
    }
    if set(by_key) != expected:
        missing = sorted(expected - set(by_key))
        extra = sorted(set(by_key) - expected)
        raise RuntimeError(f"Cache subject mismatch; missing={missing[:5]}, extra={extra[:5]}")

    samples: list[Sample] = []
    shapes: set[tuple[int, ...]] = set()
    for person, kind in sorted(expected):
        record = by_key[(person, kind)]
        profile_text = json.dumps(record["patient_profile"], indent=2)
        shots = list(record["user_turns"])
        key = module.stable_sample_key(shots, profile_text, kind)
        path = config.activation_dir / f"{kind}_{key}.npy"
        if not path.exists():
            raise RuntimeError(f"Missing saved activation file: {path}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        expected_turns = len(shots) - 1
        if array.ndim != 3 or array.shape[0] != expected_turns:
            raise RuntimeError(
                f"Invalid activation shape for {path}: {array.shape}; expected first dimension {expected_turns}"
            )
        shapes.add(tuple(array.shape[1:]))
        samples.append(
            Sample(
                person_index=person,
                kind=kind,
                path=path,
                turns=expected_turns,
            )
        )
    if len(shapes) != 1:
        raise RuntimeError(f"Activation layer/width shapes differ: {sorted(shapes)}")
    layers, hidden = next(iter(shapes))
    print(
        f"Validated offline dataset for {config.key}: 40 subjects, "
        f"80 trajectories (40 attack, 40 benign), layers={layers}, hidden={hidden}"
    )
    return samples


def samples_for_subjects(samples: list[Sample], subject_ids: set[int]) -> list[Sample]:
    return [sample for sample in samples if sample.person_index in subject_ids]


def load_training_matrix(
    samples: list[Sample],
    layer: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    arrays: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for sample in samples:
        array = np.load(sample.path, mmap_mode="r", allow_pickle=False)
        values = np.asarray(array if layer is None else array[:, layer, :])
        arrays.append(values)
        labels.append(
            np.full(values.shape[0], 1 if sample.kind == "attack" else 0, dtype=int)
        )
    return np.concatenate(arrays), np.concatenate(labels)


def trajectory_score_rows(
    samples: list[Sample],
    layer: int,
    weights: np.ndarray,
    repeat: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        array = np.load(sample.path, mmap_mode="r", allow_pickle=False)
        increments = np.asarray(array[:, layer, :]) @ weights
        scores = np.cumsum(increments)
        for turn, score in enumerate(scores, start=2):
            rows.append(
                {
                    "Repeat": repeat,
                    "PersonIndex": sample.person_index,
                    "Type": sample.kind,
                    "Turn": turn,
                    "Score": float(score),
                }
            )
    return rows


def choose_threshold(module: ModuleType, heldout: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    scores = heldout["Score"].to_numpy(dtype=float)
    lo = float(np.quantile(scores, module.THRESHOLD_RANGE_LO_Q))
    hi = float(np.quantile(scores, module.THRESHOLD_RANGE_HI_Q))
    if np.isclose(lo, hi):
        lo, hi = lo - 1.0, hi + 1.0
    pad = max((hi - lo) * module.THRESHOLD_RANGE_PAD_FRAC, 1e-6)
    thresholds = np.unique(
        np.concatenate(
            [
                np.linspace(
                    lo - pad,
                    hi + pad,
                    module.THRESHOLD_GRID_POINTS,
                ),
                np.array([0.0]),
            ]
        )
    )
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        turn_accuracies: list[float] = []
        for _, frame in heldout.groupby("Turn"):
            truth = frame["Type"].eq("attack").to_numpy()
            prediction = frame["Score"].gt(float(threshold)).to_numpy()
            turn_accuracies.append(float(np.mean(truth == prediction)))
        rows.append(
            {
                "Threshold": float(threshold),
                "MacroTurnAccuracy": float(np.mean(turn_accuracies)),
            }
        )
    table = pd.DataFrame(rows).sort_values(
        ["MacroTurnAccuracy", "Threshold"], ascending=[False, True]
    )
    best = float(table["MacroTurnAccuracy"].max())
    candidates = table[np.isclose(table["MacroTurnAccuracy"], best)].copy()
    candidates["AbsThreshold"] = candidates["Threshold"].abs()
    threshold = float(
        candidates.sort_values(["AbsThreshold", "Threshold"]).iloc[0]["Threshold"]
    )
    return threshold, table


def train_one_repeat(
    module: ModuleType,
    samples: list[Sample],
    train_ids: set[int],
    test_ids: set[int],
    repeat: int,
    seed: int,
    inner_folds: int,
    repeat_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    checkpoint_path = repeat_dir / "probe.npz"
    scores_path = repeat_dir / "outer_test_scores.csv"
    complete_path = repeat_dir / ".complete"
    repeat_dir.mkdir(parents=True, exist_ok=True)

    if complete_path.exists() and checkpoint_path.exists() and scores_path.exists():
        checkpoint = np.load(checkpoint_path, allow_pickle=False)
        if checkpoint["train_subject_ids"].astype(int).tolist() != sorted(train_ids):
            raise RuntimeError(f"Saved train split mismatch in {checkpoint_path}")
        if checkpoint["test_subject_ids"].astype(int).tolist() != sorted(test_ids):
            raise RuntimeError(f"Saved test split mismatch in {checkpoint_path}")
        print(f"Repeat {repeat}: already complete; reusing it.")
        metadata = {
            "Repeat": repeat,
            "Layer": int(checkpoint["best_layer_index"][0]),
            "CVThreshold": float(checkpoint["best_threshold"][0]),
            "TrainSubjects": " ".join(map(str, sorted(train_ids))),
            "TestSubjects": " ".join(map(str, sorted(test_ids))),
        }
        return metadata, pd.read_csv(scores_path)

    training_samples = samples_for_subjects(samples, train_ids)
    test_samples = samples_for_subjects(samples, test_ids)
    x_train, y_train = load_training_matrix(training_samples)
    layer, weights, accuracies = module.train_differential_probe(x_train, y_train)
    del x_train, y_train
    print(
        f"Repeat {repeat}: outer train=40 trajectories, outer test=40 trajectories, "
        f"selected layer={layer}, train_accuracy={accuracies[layer]:.4f}"
    )

    ordered_train = np.asarray(sorted(train_ids), dtype=int)
    inner = KFold(n_splits=inner_folds, shuffle=True, random_state=seed + repeat)
    inner_rows: list[dict[str, Any]] = []
    for fold, (inner_train_indices, inner_validation_indices) in enumerate(
        inner.split(ordered_train), start=1
    ):
        inner_train_ids = set(int(value) for value in ordered_train[inner_train_indices])
        inner_validation_ids = set(
            int(value) for value in ordered_train[inner_validation_indices]
        )
        inner_train_samples = samples_for_subjects(samples, inner_train_ids)
        inner_validation_samples = samples_for_subjects(
            samples, inner_validation_ids
        )
        x_inner, y_inner = load_training_matrix(inner_train_samples, layer=layer)
        probe = LogisticRegression(
            max_iter=1000,
            random_state=module.RANDOM_SEED,
            class_weight="balanced",
        )
        probe.fit(x_inner, y_inner)
        del x_inner, y_inner
        fold_rows = trajectory_score_rows(
            inner_validation_samples,
            layer,
            np.asarray(probe.coef_[0], dtype=float),
            repeat,
        )
        for row in fold_rows:
            row["InnerFold"] = fold
        inner_rows.extend(fold_rows)

    inner_frame = pd.DataFrame(inner_rows)
    inner_frame.to_csv(repeat_dir / "inner_cv_scores.csv", index=False)
    threshold, threshold_table = choose_threshold(module, inner_frame)
    threshold_table.to_csv(repeat_dir / "inner_cv_thresholds.csv", index=False)

    np.savez(
        checkpoint_path,
        experiment_id=np.array([EXPERIMENT_ID]),
        repeat=np.array([repeat], dtype=int),
        best_threshold=np.array([threshold], dtype=float),
        best_layer_index=np.array([layer], dtype=int),
        best_probe_weights=np.asarray(weights, dtype=float),
        final_probe_acc=np.array([accuracies[layer]], dtype=float),
        train_subject_ids=np.array(sorted(train_ids), dtype=int),
        test_subject_ids=np.array(sorted(test_ids), dtype=int),
    )

    scores = pd.DataFrame(
        trajectory_score_rows(test_samples, layer, np.asarray(weights), repeat)
    )
    scores["CVThreshold"] = threshold
    scores.to_csv(scores_path, index=False)
    complete_path.write_text("complete\n", encoding="utf-8")
    metadata = {
        "Repeat": repeat,
        "Layer": int(layer),
        "CVThreshold": float(threshold),
        "TrainSubjects": " ".join(map(str, sorted(train_ids))),
        "TestSubjects": " ".join(map(str, sorted(test_ids))),
    }
    return metadata, scores


def add_metrics(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    repeated_rows: list[dict[str, Any]] = []
    per_turn_rows: list[dict[str, Any]] = []
    for repeat, frame in scores.groupby("Repeat"):
        threshold = float(frame["CVThreshold"].iloc[0])
        truth = frame["Type"].eq("attack")
        for name, value in (("CV threshold", threshold), ("Threshold 0", 0.0)):
            prediction = frame["Score"].gt(value)
            attack = truth
            benign = ~truth
            repeated_rows.append(
                {
                    "Repeat": int(repeat),
                    "Method": name,
                    "Threshold": value,
                    "Accuracy": float(np.mean(prediction == truth)),
                    "AttackAccuracy": float(np.mean(prediction[attack])),
                    "BenignAccuracy": float(np.mean(~prediction[benign])),
                }
            )
            trajectories = frame[["PersonIndex", "Type"]].drop_duplicates()
            turn_one_truth = trajectories["Type"].eq("attack")
            turn_one_prediction = np.full(len(trajectories), 0.0 > value)
            per_turn_rows.append(
                {
                    "Repeat": int(repeat),
                    "Method": name,
                    "Turn": 1,
                    "Accuracy": float(np.mean(turn_one_prediction == turn_one_truth)),
                    "AttackAccuracy": float(
                        np.mean(turn_one_prediction[turn_one_truth.to_numpy()])
                    ),
                    "BenignAccuracy": float(
                        np.mean(~turn_one_prediction[(~turn_one_truth).to_numpy()])
                    ),
                }
            )
            for turn, turn_frame in frame.groupby("Turn"):
                turn_truth = turn_frame["Type"].eq("attack")
                turn_prediction = turn_frame["Score"].gt(value)
                turn_attack = turn_truth
                turn_benign = ~turn_truth
                per_turn_rows.append(
                    {
                        "Repeat": int(repeat),
                        "Method": name,
                        "Turn": int(turn),
                        "Accuracy": float(np.mean(turn_prediction == turn_truth)),
                        "AttackAccuracy": float(np.mean(turn_prediction[turn_attack])),
                        "BenignAccuracy": float(np.mean(~turn_prediction[turn_benign])),
                    }
                )
    return pd.DataFrame(repeated_rows), pd.DataFrame(per_turn_rows)


def aggregate_per_turn(per_turn: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (method, turn), frame in per_turn.groupby(["Method", "Turn"]):
        for metric in ("Accuracy", "AttackAccuracy", "BenignAccuracy"):
            values = frame[metric].to_numpy(dtype=float)
            mean = float(np.mean(values))
            sem = float(np.std(values, ddof=1) / np.sqrt(len(values)))
            rows.append(
                {
                    "Method": method,
                    "Turn": int(turn),
                    "Metric": metric,
                    "Mean": mean,
                    "CI95Low": max(0.0, mean - 1.96 * sem),
                    "CI95High": min(1.0, mean + 1.96 * sem),
                }
            )
    return pd.DataFrame(rows)


def plot_results(aggregate: pd.DataFrame, output_dir: Path) -> None:
    # Turn 1 is the shared greeting/reference state, so its activation delta and
    # projection score are exactly zero. Include the corresponding initial
    # classification point when older summaries begin at turn 2.
    if 1 not in aggregate["Turn"].unique():
        initial_rows = []
        initial_values = {
            "Accuracy": 0.5,
            "AttackAccuracy": 0.0,
            "BenignAccuracy": 1.0,
        }
        for method in ("CV threshold", "Threshold 0"):
            for metric, value in initial_values.items():
                initial_rows.append(
                    {
                        "Method": method,
                        "Turn": 1,
                        "Metric": metric,
                        "Mean": value,
                        "CI95Low": value,
                        "CI95High": value,
                    }
                )
        aggregate = pd.concat([pd.DataFrame(initial_rows), aggregate], ignore_index=True)

    font_size = 11.5
    tick_size = 10.5
    legend_size = 10.5
    line_width = 2.4
    marker_size = 4.5
    labels = {
        "Accuracy": "Overall accuracy",
        "AttackAccuracy": "Attack accuracy",
        "BenignAccuracy": "Benign accuracy",
    }
    colors = {"CV threshold": "#1f77b4", "Threshold 0": "#d62728"}
    figure, axes = plt.subplots(1, 3, figsize=(10.4, 3.35), sharex=True, sharey=True)
    for axis, metric in zip(axes, labels):
        for method in ("CV threshold", "Threshold 0"):
            frame = aggregate[
                (aggregate["Metric"] == metric) & (aggregate["Method"] == method)
            ].sort_values("Turn")
            x = frame["Turn"].to_numpy(dtype=int)
            mean = frame["Mean"].to_numpy(dtype=float)
            low = frame["CI95Low"].to_numpy(dtype=float)
            high = frame["CI95High"].to_numpy(dtype=float)
            axis.plot(
                x,
                mean,
                marker="o",
                markersize=marker_size,
                linewidth=line_width,
                color=colors[method],
                label=method,
            )
            axis.fill_between(x, low, high, color=colors[method], alpha=0.16)
        axis.set_xlabel("Conversation turn", fontsize=font_size)
        axis.set_ylabel(labels[metric], fontsize=font_size)
        axis.set_ylim(-0.02, 1.02)
        axis.set_xticks([1, 5, 10, 15, 20])
        axis.tick_params(labelsize=tick_size)
    axes[-1].legend(frameon=False, loc="lower right", fontsize=legend_size)
    figure.tight_layout(pad=0.35, w_pad=0.65)
    figure.savefig(output_dir / "offline_repeated_50_50_per_turn.pdf", bbox_inches="tight")
    figure.savefig(
        output_dir / "offline_repeated_50_50_per_turn.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    metric_labels = {
        "Accuracy": "Accuracy",
        "AttackAccuracy": "Attack detection",
        "BenignAccuracy": "Benign accuracy",
    }
    metric_colors = {
        "Accuracy": "#222222",
        "AttackAccuracy": "#e66101",
        "BenignAccuracy": "#1f77b4",
    }
    output_names = {
        "CV threshold": "offline_repeated_50_50_cv_threshold_per_turn",
        "Threshold 0": "offline_repeated_50_50_threshold_0_per_turn",
    }
    for method, output_name in output_names.items():
        figure, axis = plt.subplots(figsize=(4.6, 3.35))
        for metric, label in metric_labels.items():
            frame = aggregate[
                (aggregate["Metric"] == metric) & (aggregate["Method"] == method)
            ].sort_values("Turn")
            x = frame["Turn"].to_numpy(dtype=int)
            mean = frame["Mean"].to_numpy(dtype=float)
            low = frame["CI95Low"].to_numpy(dtype=float)
            high = frame["CI95High"].to_numpy(dtype=float)
            axis.plot(
                x,
                mean,
                marker="o",
                markersize=marker_size,
                linewidth=line_width,
                color=metric_colors[metric],
                label=label,
            )
            axis.fill_between(x, low, high, color=metric_colors[metric], alpha=0.14)
        axis.set_xlabel("Conversation turn", fontsize=font_size)
        axis.set_ylabel("Held-out performance", fontsize=font_size)
        axis.set_xlim(0.75, 20.25)
        axis.set_ylim(-0.02, 1.02)
        axis.set_xticks([1, 5, 10, 15, 20])
        axis.tick_params(labelsize=tick_size)
        axis.legend(frameon=False, loc="lower right", fontsize=legend_size)
        figure.tight_layout(pad=0.35)
        figure.savefig(output_dir / f"{output_name}.pdf", bbox_inches="tight")
        figure.savefig(output_dir / f"{output_name}.png", dpi=300, bbox_inches="tight")
        plt.close(figure)


def main() -> None:
    args = parse_args()
    config = MODEL_CONFIGS[args.model]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    module = load_module(config)
    samples = load_samples(module, config)
    print("Offline-only run: this script has no model-loading or API-generation stage.")
    if args.validate_only:
        print("Validation complete; no probes were trained.")
        return

    subjects = np.asarray(TRAINING_SUBJECTS, dtype=int)
    splitter = ShuffleSplit(
        n_splits=args.repeats,
        train_size=0.5,
        test_size=0.5,
        random_state=args.seed,
    )
    metadata_rows: list[dict[str, Any]] = []
    score_frames: list[pd.DataFrame] = []
    seen_test_sets: set[tuple[int, ...]] = set()
    for repeat, (train_indices, test_indices) in enumerate(
        splitter.split(subjects), start=1
    ):
        train_ids = set(int(value) for value in subjects[train_indices])
        test_ids = set(int(value) for value in subjects[test_indices])
        if len(train_ids) != 20 or len(test_ids) != 20 or train_ids & test_ids:
            raise RuntimeError(
                f"Repeat {repeat} is not an exact disjoint 20-subject/20-subject split"
            )
        if train_ids | test_ids != set(TRAINING_SUBJECTS):
            raise RuntimeError(f"Repeat {repeat} does not cover all 40 subjects")
        test_key = tuple(sorted(test_ids))
        if test_key in seen_test_sets:
            raise RuntimeError(f"Repeated test partition encountered at repeat {repeat}")
        seen_test_sets.add(test_key)
        metadata, scores = train_one_repeat(
            module,
            samples,
            train_ids,
            test_ids,
            repeat,
            args.seed,
            args.inner_folds,
            args.output_dir / f"repeat_{repeat:02d}",
        )
        metadata_rows.append(metadata)
        score_frames.append(scores)

    assignments = pd.DataFrame(metadata_rows)
    assignments.to_csv(args.output_dir / "split_assignments.csv", index=False)
    all_scores = pd.concat(score_frames, ignore_index=True)
    all_scores.to_csv(args.output_dir / "outer_test_scores.csv", index=False)
    repeated, per_turn = add_metrics(all_scores)
    repeated.to_csv(args.output_dir / "repeat_summary.csv", index=False)
    per_turn.to_csv(args.output_dir / "repeat_per_turn.csv", index=False)
    aggregate = aggregate_per_turn(per_turn)
    aggregate.to_csv(args.output_dir / "per_turn_mean_ci95.csv", index=False)
    plot_results(aggregate, args.output_dir)
    (args.output_dir / ".complete").write_text("complete\n", encoding="utf-8")
    print(f"Completed proper offline repeated 50:50 evaluation: {args.output_dir}")


if __name__ == "__main__":
    main()
