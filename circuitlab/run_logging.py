"""Append-only, reproducible run records for public toolkit experiments."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4


DEFAULT_RUN_LOG = Path(__file__).resolve().parents[1] / "results" / "runs.jsonl"
_WRITE_LOCK = Lock()


def _json_safe(value: Any) -> Any:
    """Convert bounded scalar/list metadata to JSON, rejecting tensors and objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    raise TypeError(f"Run records cannot contain {type(value).__name__}; log bounded summaries, not tensors or objects.")


def append_run_record(
    *,
    tool: str,
    task: str,
    seed: int,
    components: list[str],
    positions: str | None,
    dataset_size: int,
    metric: str,
    result_summary: dict[str, Any],
    passes_used: int,
    parameters: dict[str, Any],
    path: Path | str | None = None,
) -> dict[str, Any]:
    """Append one JSON object line and return the exact persisted record.

    The caller supplies concise scalar/list summaries only. The logger never
    reads environment variables or serializes model activations/tensors.
    """
    if not isinstance(tool, str) or not tool:
        raise ValueError("tool must be a non-empty string.")
    if not isinstance(task, str) or not task:
        raise ValueError("task must be a non-empty string.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")
    if not isinstance(components, list) or not all(isinstance(component, str) for component in components):
        raise TypeError("components must be a list of strings.")
    if positions is not None and not isinstance(positions, str):
        raise TypeError("positions must be a string or None.")
    if isinstance(dataset_size, bool) or not isinstance(dataset_size, int) or dataset_size <= 0:
        raise ValueError("dataset_size must be a positive integer.")
    if isinstance(passes_used, bool) or not isinstance(passes_used, int) or passes_used < 0:
        raise ValueError("passes_used must be a non-negative integer.")

    record = _json_safe({
        "run_id": str(uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool,
        "task": task,
        "seed": seed,
        "components": components,
        "positions": positions,
        "dataset_size": dataset_size,
        "metric": metric,
        "result_summary": result_summary,
        "passes_used": passes_used,
        "parameters": parameters,
    })
    log_path = Path(path) if path is not None else DEFAULT_RUN_LOG
    log_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(record, allow_nan=False, separators=(",", ":"))
    with _WRITE_LOCK:
        with log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded + "\n")
            handle.flush()
    return record
