"""Resumable five-seed benchmark for CircuitLab circuit-discovery strategies.

This runner never launches exhaustive patching or invents a CircuitLab/ACDC
circuit. It consumes completed exhaustive CSVs and optional source traces, then
evaluates every available circuit on the same seeded IOI dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import pandas as pd

import lab_tools
from circuitlab.datasets import load_ioi_truth, normalise_attention_components
from circuitlab.evaluation.circuit import score_circuit_against_truth


SEEDS = (0, 1, 2, 3, 4)
RAW_COLUMNS = [
    "strategy", "seed", "forward_passes", "precision", "recall", "f1", "faithfulness", "circuit_size",
    "status", "unavailable_reason", "discovery_forward_passes", "evaluation_forward_passes", "dataset_size",
    "circuit", "source_path",
]
AGGREGATE_COLUMNS = [
    "strategy", "status", "n_complete", "n_unavailable", "forward_passes_mean", "forward_passes_std",
    "forward_passes_min", "forward_passes_max", "precision_mean", "precision_std", "precision_min", "precision_max",
    "recall_mean", "recall_std", "recall_min", "recall_max", "f1_mean", "f1_std", "f1_min", "f1_max",
    "faithfulness_mean", "faithfulness_std", "faithfulness_min", "faithfulness_max", "circuit_size_mean", "circuit_size_std",
]


def _read_csv(path: Path, columns: list[str]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(columns) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns {sorted(missing)}.")
        return [{column: row.get(column, "") for column in columns} for row in reader]


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _empty_row(strategy: str, seed: int, n: int, reason: str) -> dict[str, Any]:
    return {
        "strategy": strategy, "seed": seed, "forward_passes": "", "precision": "", "recall": "", "f1": "",
        "faithfulness": "", "circuit_size": "", "status": "unavailable", "unavailable_reason": reason,
        "discovery_forward_passes": "", "evaluation_forward_passes": "", "dataset_size": n,
        "circuit": "", "source_path": "",
    }


def _trace_runs(path: Path) -> dict[int, dict[str, Any]]:
    """Read a JSON, JSONL, or CSV trace containing seed/circuit/forward_passes."""
    if not path.exists():
        return {}
    if path.suffix.lower() == ".csv":
        records = list(pd.read_csv(path).to_dict(orient="records"))
    else:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return {}
        parsed = json.loads(text) if path.suffix.lower() == ".json" else [json.loads(line) for line in text.splitlines()]
        records = parsed.get("runs", []) if isinstance(parsed, dict) else parsed
    if not isinstance(records, list):
        raise ValueError(f"Trace {path} must contain a list of run objects or a JSON object with 'runs'.")
    output: dict[int, dict[str, Any]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Trace record {index} is not an object.")
        seed, circuit, passes = record.get("seed"), record.get("circuit"), record.get("forward_passes")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed not in SEEDS:
            raise ValueError(f"Trace record {index} has unsupported seed {seed!r}; expected one of {SEEDS}.")
        if isinstance(passes, bool) or not isinstance(passes, int) or passes < 0:
            raise ValueError(f"Trace record {index} must provide non-negative integer forward_passes.")
        if isinstance(circuit, str):
            try:
                circuit = json.loads(circuit)
            except json.JSONDecodeError as error:
                raise ValueError(f"Trace record {index} circuit string must be JSON list syntax.") from error
        components = sorted(normalise_attention_components(circuit))
        if not components:
            raise ValueError(f"Trace record {index} has an empty circuit.")
        if seed in output:
            raise ValueError(f"Trace {path} has duplicate seed {seed}.")
        output[seed] = {"circuit": components, "forward_passes": passes}
    return output


def _exhaustive_circuit(path: Path, circuit_size: int) -> tuple[list[str], int]:
    """Select deterministic top-effect heads only from a completed 144-head sweep."""
    if circuit_size <= 0:
        raise ValueError("circuit_size must be positive.")
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {"component", "effect", "cumulative_forward_passes"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Exhaustive file {path} is missing columns {sorted(missing)}.")
    if len(frame) != 144 or frame["component"].nunique() != 144:
        raise ValueError(f"Exhaustive file {path} must contain one row for all 144 GPT-2 Small heads.")
    if circuit_size > len(frame):
        raise ValueError(f"circuit_size {circuit_size} exceeds 144 heads.")
    ranked = frame.sort_values(["effect", "layer", "head"], ascending=[False, True, True], kind="stable")
    circuit = ranked.head(circuit_size)["component"].tolist()
    normalise_attention_components(circuit)
    return circuit, int(frame["cumulative_forward_passes"].max())


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for strategy in ("exhaustive", "circuitlab", "acdc"):
        strategy_rows = [row for row in rows if row["strategy"] == strategy]
        complete = [row for row in strategy_rows if row["status"] == "complete"]
        row: dict[str, Any] = {"strategy": strategy, "status": "complete" if complete else "unavailable", "n_complete": len(complete), "n_unavailable": len(strategy_rows) - len(complete)}
        for metric in ("forward_passes", "precision", "recall", "f1", "faithfulness", "circuit_size"):
            values = [float(item[metric]) for item in complete]
            row[f"{metric}_mean"] = statistics.fmean(values) if values else ""
            row[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else (0.0 if values else "")
            if metric != "circuit_size":
                row[f"{metric}_min"] = min(values) if values else ""
                row[f"{metric}_max"] = max(values) if values else ""
        output.append(row)
    return output


def run_benchmark(
    *, n: int = 100, circuit_size: int = 10, results_dir: Path | str = REPOSITORY_ROOT / "results",
    circuitlab_trace: Path | str | None = None, acdc_trace: Path | str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Resume five-seed comparisons, evaluating only source-backed circuits."""
    if n <= 0 or circuit_size <= 0:
        raise ValueError("n and circuit_size must be positive.")
    output_dir = Path(results_dir)
    raw_path, aggregate_path = output_dir / "benchmark_raw.csv", output_dir / "benchmark_aggregate.csv"
    rows = _read_csv(raw_path, RAW_COLUMNS)
    completed = {
        (row["strategy"], int(row["seed"]))
        for row in rows
        if row["status"] == "complete" and int(row["dataset_size"]) == n and int(row["circuit_size"]) == circuit_size
    }
    trace_paths = {"circuitlab": Path(circuitlab_trace) if circuitlab_trace else None, "acdc": Path(acdc_trace) if acdc_trace else None}
    traces = {strategy: _trace_runs(path) if path else {} for strategy, path in trace_paths.items()}
    try:
        truth = load_ioi_truth()
        truth_error: str | None = None
    except ValueError as error:
        truth = None
        truth_error = str(error)

    for seed in SEEDS:
        candidates: dict[str, tuple[list[str], int, str] | None] = {}
        exhaustive_path = output_dir / f"exhaustive_ioi_seed{seed}_n{n}_raw.csv"
        try:
            circuit, passes = _exhaustive_circuit(exhaustive_path, circuit_size)
            candidates["exhaustive"] = (circuit, passes, str(exhaustive_path))
        except (FileNotFoundError, ValueError) as error:
            candidates["exhaustive"] = None
            exhaustive_reason = f"Completed exhaustive results unavailable: {error}"
        for strategy in ("circuitlab", "acdc"):
            trace = traces[strategy].get(seed)
            candidates[strategy] = (trace["circuit"], trace["forward_passes"], str(trace_paths[strategy])) if trace else None

        for strategy in ("exhaustive", "circuitlab", "acdc"):
            if (strategy, seed) in completed:
                continue
            # An unavailable row is a snapshot of missing evidence, not a
            # completed experiment. Replace it on resume instead of growing
            # duplicate rows once sources or verified truth arrive.
            rows[:] = [
                row for row in rows
                if not (row["strategy"] == strategy and int(row["seed"]) == seed and int(row["dataset_size"]) == n)
            ]
            candidate = candidates[strategy]
            if candidate is None:
                reason = exhaustive_reason if strategy == "exhaustive" else f"No {strategy} trace record for seed {seed}."
                rows.append(_empty_row(strategy, seed, n, reason))
                _write_csv(raw_path, RAW_COLUMNS, rows)
                continue
            if truth is None:
                rows.append(_empty_row(strategy, seed, n, f"Verified IOI truth unavailable: {truth_error}"))
                _write_csv(raw_path, RAW_COLUMNS, rows)
                continue

            circuit, discovery_passes, source_path = candidate
            dataset = lab_tools.make_dataset("ioi", n=n, seed=seed)
            faithfulness = lab_tools.faithfulness(circuit, ds=dataset)
            truth_score = score_circuit_against_truth(circuit, truth)
            evaluation_passes = int(faithfulness["forward_passes"])
            rows.append({
                "strategy": strategy, "seed": seed, "forward_passes": discovery_passes + evaluation_passes,
                "precision": truth_score["precision"], "recall": truth_score["recall"], "f1": truth_score["f1"],
                "faithfulness": faithfulness["faithfulness"], "circuit_size": len(circuit), "status": "complete",
                "unavailable_reason": "", "discovery_forward_passes": discovery_passes,
                "evaluation_forward_passes": evaluation_passes, "dataset_size": n,
                "circuit": json.dumps(circuit), "source_path": source_path,
            })
            _write_csv(raw_path, RAW_COLUMNS, rows)

    aggregate = _aggregate(rows)
    _write_csv(aggregate_path, AGGREGATE_COLUMNS, aggregate)
    return rows, aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--circuit-size", type=int, default=10)
    parser.add_argument("--results-dir", type=Path, default=REPOSITORY_ROOT / "results")
    parser.add_argument("--circuitlab-trace", type=Path, help="JSON/JSONL/CSV records with seed, circuit, forward_passes.")
    parser.add_argument("--acdc-trace", type=Path, help="JSON/JSONL/CSV records with seed, circuit, forward_passes.")
    args = parser.parse_args()
    rows, aggregate = run_benchmark(
        n=args.n, circuit_size=args.circuit_size, results_dir=args.results_dir,
        circuitlab_trace=args.circuitlab_trace, acdc_trace=args.acdc_trace,
    )
    print(pd.DataFrame(rows)[["strategy", "seed", "status", "unavailable_reason"]].to_string(index=False))
    print("\nAggregate:")
    print(pd.DataFrame(aggregate).to_string(index=False))


if __name__ == "__main__":
    main()
