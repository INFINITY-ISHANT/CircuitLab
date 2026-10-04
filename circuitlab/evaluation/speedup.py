"""Honest per-seed speedup summaries from circuit-recovery target costs."""

from __future__ import annotations

import csv
import math
from pathlib import Path
import statistics
from typing import Any


def _validate_target_record(record: object, index: int) -> tuple[int, int | None]:
    if not isinstance(record, dict):
        raise TypeError(f"records[{index}] must be an object.")
    seed = record.get("seed")
    passes = record.get("passes_to_target")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"records[{index}].seed must be an integer.")
    if passes is not None and (isinstance(passes, bool) or not isinstance(passes, int) or passes < 0):
        raise ValueError(f"records[{index}].passes_to_target must be a non-negative integer or None.")
    return seed, passes


def calculate_speedup(
    baseline_records: object,
    circuitlab_records: object,
) -> dict[str, Any]:
    """Calculate baseline/CircuitLab target-cost ratios without false infinities.

    Both inputs are lists of ``{"seed": int, "passes_to_target": int | None}``.
    A seed contributes a finite speedup only when both strategies reach the
    target and CircuitLab's target cost is positive. Other outcomes are retained
    in ``per_seed`` with an explicit status and ``speedup: None``.
    """
    if not isinstance(baseline_records, list) or not isinstance(circuitlab_records, list):
        raise TypeError("baseline_records and circuitlab_records must be lists.")
    baseline = dict(_validate_target_record(record, index) for index, record in enumerate(baseline_records))
    circuitlab = dict(_validate_target_record(record, index) for index, record in enumerate(circuitlab_records))
    if len(baseline) != len(baseline_records) or len(circuitlab) != len(circuitlab_records):
        raise ValueError("Each strategy may contain at most one record per seed.")

    per_seed: list[dict[str, Any]] = []
    valid_values: list[float] = []
    for seed in sorted(set(baseline) | set(circuitlab)):
        baseline_passes, circuitlab_passes = baseline.get(seed), circuitlab.get(seed)
        if seed not in baseline or seed not in circuitlab:
            status, speedup = "missing_strategy_record", None
        elif baseline_passes is None and circuitlab_passes is None:
            status, speedup = "neither_reached_target", None
        elif baseline_passes is None:
            status, speedup = "baseline_did_not_reach_target", None
        elif circuitlab_passes is None:
            status, speedup = "circuitlab_did_not_reach_target", None
        elif circuitlab_passes == 0:
            # A zero-cost target record is not a valid experimental comparison.
            status, speedup = "invalid_zero_circuitlab_target_cost", None
        else:
            status, speedup = "complete", baseline_passes / circuitlab_passes
            valid_values.append(speedup)
        per_seed.append({
            "seed": seed,
            "baseline_passes_to_target": baseline_passes,
            "circuitlab_passes_to_target": circuitlab_passes,
            "speedup": speedup,
            "status": status,
        })

    return {
        "per_seed": per_seed,
        "mean_speedup": statistics.fmean(valid_values) if valid_values else None,
        "std_speedup": statistics.stdev(valid_values) if len(valid_values) > 1 else (0.0 if valid_values else None),
        "successful_seeds": len(valid_values),
        "total_seeds": len(per_seed),
    }


def load_target_costs_from_csv(path: Path | str) -> list[dict[str, Any]]:
    """Load ``seed`` and ``passes_to_target`` from a trajectory summary CSV."""
    source = Path(path)
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"seed", "passes_to_target"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{source} is missing columns {sorted(missing)}.")
        records: list[dict[str, Any]] = []
        for row in reader:
            raw_passes = row["passes_to_target"].strip()
            records.append({"seed": int(row["seed"]), "passes_to_target": None if raw_passes == "" else int(raw_passes)})
    return records
