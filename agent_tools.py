"""Agent-facing layer over lab_tools.

Agents can only send text and read plain JSON, while lab_tools works with dataset
objects and DataFrames. This module keeps datasets in a registry keyed by
dataset_id (ids look like "ioi-42-100", so a fresh process can rebuild one) and
turns every result into JSON-safe records. lab_server.py exposes these functions.
"""
from __future__ import annotations

import math
import os
import re
from typing import Any

import lab_tools

_MAX_SWEEP_SENDERS = 24
_DATASETS: dict[str, Any] = {}
_ID_PATTERN = re.compile(r"^ioi-(-?\d+)-(\d+)$")
_RANDOM_KEYS = ("candidate_faithfulness", "random_mean", "random_std", "percentile", "beats_random")


def _clean(value: Any) -> Any:
    """Make a result JSON-safe: numpy scalars to Python, NaN and inf to None, floats rounded."""
    if hasattr(value, "to_dict") and hasattr(value, "columns"):
        value = value.to_dict("records")
    if hasattr(value, "item") and not isinstance(value, (list, dict, str)):
        try:
            value = value.item()
        except (ValueError, TypeError):
            pass
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else round(value, 4)
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def _dataset(dataset_id: str) -> Any:
    if dataset_id in _DATASETS:
        return _DATASETS[dataset_id]
    match = _ID_PATTERN.fullmatch(str(dataset_id).strip())
    if match is None:
        raise ValueError(f"Unknown dataset_id {dataset_id!r}. Call make_dataset first and pass the dataset_id it returns.")
    seed, n = int(match.group(1)), int(match.group(2))
    ds = lab_tools.make_dataset("ioi", n=n, seed=seed)
    _DATASETS[ds.dataset_id] = ds
    return ds


def make_dataset(task: str = "ioi", n: int = 100, seed: int = 42) -> dict:
    """Build the clean and corrupted prompt set for a task (only "ioi" exists so far). Returns the dataset_id, which every other tool needs. Use first, once per run."""
    ds = lab_tools.make_dataset(task, n=n, seed=seed)
    _DATASETS[ds.dataset_id] = ds
    return {"dataset_id": ds.dataset_id, "task": task.lower(), "n": n, "seed": seed}


def run_baseline(task: str = "ioi", dataset_id: str = "") -> dict:
    """Measure how well the full model does the task: accuracy and mean logit difference (IO minus subject). Pass the dataset_id from make_dataset."""
    ds = _dataset(dataset_id) if dataset_id else None
    result = lab_tools.run_baseline(task, ds=ds)
    result.pop("per_example_logit_diffs", None)
    return _clean(result)


def patch(components: list[str], dataset_id: str) -> list[dict]:
    """Activation patching: copy each listed head's clean output into corrupted prompts at the final token and report how much of the answer comes back. Components are heads like "L9H9". One record per head, with effect and recovery (1.0 means the head alone restores the full behavior)."""
    return _clean(lab_tools.patch(components, ds=_dataset(dataset_id)))


def ablate(components: list[str], dataset_id: str, mode: str = "mean") -> list[dict]:
    """Mean-ablate each listed head (heads like "L9H9") on clean prompts and report the logit difference that remains. A big drop means the head matters."""
    return _clean(lab_tools.ablate(components, ds=_dataset(dataset_id), mode=mode))


def path_patch(sender: str, receivers: list[str], dataset_id: str) -> list[dict]:
    """Path patching: how much of the effect flows from the sender head to the receiver heads (heads like "L5H5"). The sender must be in an earlier layer than every receiver."""
    return _clean(lab_tools.path_patch(sender, receivers, ds=_dataset(dataset_id)))

def path_patch_sweep(receivers: list[str], senders: list[str], dataset_id: str, top_k: int = 10) -> list[dict]:
    """Path-patch several candidate sender heads (heads like "L5H5") into the same receiver heads, all receivers together, and return the top_k senders by path_effect. Pick at most 24 senders you have a reason to suspect, from the literature, earlier results or the hypothesis agent. Each sender must be in an earlier layer than every receiver. Cheaper than calling path_patch once per sender."""
    if not receivers:
        raise ValueError("Give at least one receiver head.")
    if not senders:
        raise ValueError("Give the candidate sender heads to test. Choose them with a reason, do not try every head.")
    if len(senders) > _MAX_SWEEP_SENDERS:
        raise ValueError(f"At most {_MAX_SWEEP_SENDERS} senders per sweep, got {len(senders)}.")
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    ds = _dataset(dataset_id)
    rows = []
    for sender in senders:
        frame = lab_tools.path_patch(sender, receivers, ds=ds, joint_only=True)
        row = frame.iloc[0]
        rows.append({"sender": sender, "path_effect": float(row["path_effect"]), "recovery": float(row["recovery"])})
    rows.sort(key=lambda item: item["path_effect"], reverse=True)
    return _clean(rows[:top_k])


def faithfulness(circuit: list[str], dataset_id: str, n_random: int = 10) -> dict:
    """How well a proposed circuit (list of heads like "L9H9") alone reproduces the model's behavior, from 0 to 1, together with a same-size random-circuit control. The circuit beats chance only if random_control.beats_random is true."""
    ds = _dataset(dataset_id)
    result = _clean(lab_tools.faithfulness(circuit, ds=ds))
    control = lab_tools.random_circuit_control(circuit, ds=ds, n_random=n_random)
    result["random_control"] = _clean({key: control[key] for key in _RANDOM_KEYS if key in control})
    return result


def score_vs_truth(circuit: list[str]) -> dict:
    """Precision and recall of a proposed IOI circuit against the published IOI circuit. If it says available is false, report the ground truth as unavailable and do not guess it."""
    try:
        return _clean(lab_tools.score_vs_truth(circuit))
    except ValueError as error:
        return {"available": False, "error": str(error)}


def budget() -> dict:
    """How many forward passes have been used and how many are left."""
    return _clean(lab_tools.budget())


def search_literature(query: str) -> list[dict]:
    """Search arXiv for papers on a topic (2 to 4 keywords). Returns up to 5 results with title, arxiv_id and a short snippet."""
    results = lab_tools.search_literature(query)
    blocked = [item.strip() for item in os.environ.get("CIRCUITLAB_BLOCK_ARXIV", "").split(",") if item.strip()]
    if blocked:
        results = [r for r in results if not any(str(r.get("arxiv_id", "")).startswith(b) for b in blocked)]
    return results


def present_claim(circuit: list[str], run_ids: list[str], arxiv_ids: list[str], summary: str) -> dict:
    """Show the final circuit to the human as an agent-generated hypothesis. Call once, only after the analyst reported faithfulness of at least 0.7 beating the random control and safety cleared the claim. The human must approve before it runs."""
    return lab_tools.present_claim(circuit, run_ids, arxiv_ids, summary)
