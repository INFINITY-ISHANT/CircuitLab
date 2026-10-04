"""Circuit-recovery trajectories indexed by cumulative experimental cost."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re
from typing import Any

from circuitlab.datasets import load_ioi_truth, normalise_attention_components, validate_ioi_truth
from circuitlab.evaluation.circuit import score_circuit_against_truth


DEFAULT_TRAJECTORY_DIRECTORY = Path(__file__).resolve().parents[2] / "results" / "trajectories"
_STRATEGY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def _normalise_steps(steps: object) -> list[dict[str, Any]]:
    """Validate a monotonic proposal trace without inferring an unstated circuit."""
    if not isinstance(steps, list) or not steps:
        raise ValueError("steps must be a non-empty list of trace objects.")
    current_circuit: set[str] = set()
    previous_passes = -1
    normalised: list[dict[str, Any]] = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            raise TypeError(f"steps[{index}] must be an object.")
        passes = step.get("forward_passes")
        if isinstance(passes, bool) or not isinstance(passes, int) or passes < 0:
            raise ValueError(f"steps[{index}].forward_passes must be a non-negative integer.")
        if passes < previous_passes:
            raise ValueError("steps must be ordered by non-decreasing forward_passes.")
        has_circuit, has_component = "circuit" in step, "component" in step
        if has_circuit == has_component:
            raise ValueError(
                f"steps[{index}] must provide exactly one of 'circuit' (the full current set) "
                "or 'component' (one additive proposal)."
            )
        if has_circuit:
            current_circuit = normalise_attention_components(step["circuit"])
        else:
            current_circuit.add(next(iter(normalise_attention_components([step["component"]]))))
        if not current_circuit:
            raise ValueError(f"steps[{index}] resolves to an empty circuit.")
        normalised.append({"step": index + 1, "forward_passes": passes, "components": sorted(current_circuit)})
        previous_passes = passes
    return normalised


def _write_trajectory(
    strategy: str, result: dict[str, Any], output_dir: Path | str,
) -> tuple[Path, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / f"{strategy}_recovery_trajectory.json"
    csv_path = directory / f"{strategy}_recovery_trajectory.csv"
    temporary_json = json_path.with_suffix(".tmp")
    temporary_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_json.replace(json_path)
    columns = ["step", "forward_passes", "circuit_size", "precision", "recall", "f1", "success"]
    temporary_csv = csv_path.with_suffix(".tmp")
    with temporary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in columns} for row in result["trajectory"]])
    temporary_csv.replace(csv_path)
    return json_path, csv_path


def evaluate_recovery_trajectory(
    strategy: str,
    steps: object,
    *,
    seed: int | None = None,
    truth: object | None = None,
    output_dir: Path | str = DEFAULT_TRAJECTORY_DIRECTORY,
) -> dict[str, Any]:
    """Score each trace step and persist a recall-vs-forward-passes trajectory.

    A trace step supplies cumulative ``forward_passes`` plus either the complete
    current ``circuit`` or a single additive ``component``. ``passes_to_target``
    is the first cumulative cost at which recall >= 0.80 and precision >= 0.70;
    it is ``None`` if no step reaches the target.
    """
    if not isinstance(strategy, str) or not _STRATEGY_PATTERN.fullmatch(strategy):
        raise ValueError("strategy must contain only letters, digits, '.', '_', or '-'.")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError("seed must be an integer or None.")
    source_truth = load_ioi_truth() if truth is None else validate_ioi_truth(truth, require_verified=True)
    trajectory: list[dict[str, Any]] = []
    passes_to_target: int | None = None
    for step in _normalise_steps(steps):
        score = score_circuit_against_truth(step["components"], source_truth)
        reached = bool(score["success"])
        if reached and passes_to_target is None:
            passes_to_target = step["forward_passes"]
        trajectory.append({
            **step,
            "circuit_size": len(step["components"]),
            "precision": score["precision"],
            "recall": score["recall"],
            "f1": score["f1"],
            "success": reached,
        })
    result: dict[str, Any] = {
        "strategy": strategy,
        "seed": seed,
        "truth_source": source_truth["source"],
        "target": {"recall": 0.80, "precision": 0.70},
        "passes_to_target": passes_to_target,
        "trajectory": trajectory,
    }
    json_path, csv_path = _write_trajectory(strategy, result, output_dir)
    result["json_path"] = str(json_path)
    result["csv_path"] = str(csv_path)
    return result
