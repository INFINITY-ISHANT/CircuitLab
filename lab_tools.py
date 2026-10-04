"""Safe public API for CircuitLab agents.

This module accepts only typed IOI datasets and stable GPT-2 Small attention
head IDs (``L<layer>H<head>``). Hook implementation details remain internal.
"""

from __future__ import annotations

import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Final

import torch

from circuitlab.datasets import IOIDataset, make_ioi_dataset


_COMPONENT_PATTERN: Final = re.compile(r"^L(?P<layer>\d+)H(?P<head>\d+)$")
_UNSET: Final = object()


def _validate_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _validate_seed(seed: object) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")
    return seed


def _canonical_component(component: str) -> str:
    """Accept 'L9H9', 'l9h9' and '9.9' and return the stable form 'L9H9'."""
    text = component.strip()
    dotted = re.fullmatch(r"(\d+)\.(\d+)", text)
    if dotted:
        return f"L{dotted.group(1)}H{dotted.group(2)}"
    return text.upper() if re.fullmatch(r"[Ll]\d+[Hh]\d+", text) else text


def _validate_component(component: object, name: str = "component") -> str:
    if not isinstance(component, str):
        raise TypeError(f"{name} must be a string such as 'L9H9'.")
    component = _canonical_component(component)
    match = _COMPONENT_PATTERN.fullmatch(component)
    if match is None:
        raise ValueError(f"{name}={component!r} must use stable form 'L<layer>H<head>'.")
    layer, head = int(match["layer"]), int(match["head"])
    if not 0 <= layer <= 11:
        raise ValueError(f"{name} layer must be in [0, 11], got {layer}.")
    if not 0 <= head <= 11:
        raise ValueError(f"{name} head must be in [0, 11], got {head}.")
    return component


def _validate_components(components: object, name: str = "components") -> list[str]:
    raw = [components] if isinstance(components, str) else components
    if not isinstance(raw, list) or not raw:
        raise TypeError(f"{name} must be one component string or a non-empty list of strings.")
    validated = [_validate_component(component, f"{name}[{index}]") for index, component in enumerate(raw)]
    if len(set(validated)) != len(validated):
        raise ValueError(f"{name} must not contain duplicate heads.")
    return validated


def _validate_dataset(ds: object) -> IOIDataset:
    if not isinstance(ds, IOIDataset):
        raise TypeError("ds must be an IOIDataset created by make_dataset(...).")
    n = len(ds)
    if n <= 0 or len(ds.corrupted_prompts) != n or len(ds.io_names) != n or len(ds.subject_names) != n:
        raise ValueError("IOIDataset prompt/name fields must have equal non-zero lengths.")
    if not isinstance(ds.clean_tokens, torch.Tensor) or not isinstance(ds.corrupted_tokens, torch.Tensor):
        raise TypeError("IOIDataset token fields must be torch tensors.")
    if ds.clean_tokens.ndim != 2 or ds.corrupted_tokens.ndim != 2 or ds.clean_tokens.shape != ds.corrupted_tokens.shape:
        raise ValueError("IOIDataset clean_tokens and corrupted_tokens must have equal shape [batch, position].")
    if ds.clean_tokens.shape[0] != n:
        raise ValueError("IOIDataset token batch size must match prompt count.")
    for token_ids, field_name in ((ds.io_token_ids, "io_token_ids"), (ds.subject_token_ids, "subject_token_ids")):
        if not isinstance(token_ids, torch.Tensor) or token_ids.ndim != 1 or token_ids.shape[0] != n:
            raise ValueError(f"IOIDataset {field_name} must have shape [batch].")
    if any(io_name == subject_name for io_name, subject_name in zip(ds.io_names, ds.subject_names)):
        raise ValueError("IOIDataset requires distinct IO and subject names per example.")
    if ds.metadata.get("task") != "ioi":
        raise ValueError("IOIDataset metadata must identify task='ioi'.")
    lengths = ds.metadata.get("prompt_lengths")
    if not isinstance(lengths, list) or len(lengths) != n or any(
        not isinstance(length, int) or not 0 < length <= ds.clean_tokens.shape[1] for length in lengths
    ):
        raise ValueError("IOIDataset metadata.prompt_lengths must contain valid lengths for every example.")
    return ds


def _validate_positions(positions: object) -> str:
    values = [positions] if isinstance(positions, str) else positions
    if values != ["final"]:
        raise NotImplementedError("Only positions='final' is supported.")
    return "final"


def _dataframe_summary(frame: Any, metric_columns: list[str]) -> dict[str, Any]:
    """Extract scalar DataFrame metadata without placing rows or tensors in logs."""
    summary: dict[str, Any] = {"rows": int(len(frame)), "columns": [str(column) for column in frame.columns]}
    for column in metric_columns:
        if column in frame.columns and len(frame):
            values = frame[column]
            summary[f"mean_{column}"] = float(values.mean())
            summary[f"max_{column}"] = float(values.max())
    return summary


def _log_experiment(
    *, tool: str, dataset: IOIDataset, components: list[str], positions: str | None,
    metric: str, result_summary: dict[str, Any], passes_used: int,
    parameters: dict[str, Any], log_path: Path | str | None,
) -> None:
    from circuitlab.run_logging import append_run_record

    append_run_record(
        tool=tool,
        task="ioi",
        seed=dataset.seed,
        components=components,
        positions=positions,
        dataset_size=len(dataset),
        metric=metric,
        result_summary=result_summary,
        passes_used=passes_used,
        parameters={"dataset_id": dataset.dataset_id, **parameters},
        path=log_path,
    )


def make_dataset(task: str, n: int = 100, seed: int = 0) -> IOIDataset:
    """Create a deterministic tokenized IOI dataset."""
    if not isinstance(task, str) or task.lower() != "ioi":
        raise NotImplementedError("Only task='ioi' is implemented.")
    return make_ioi_dataset(n=_validate_positive_int(n, "n"), seed=_validate_seed(seed))


def run_baseline(task: str = "ioi", ds: IOIDataset | None = None, n: int = 100, seed: int = 42) -> dict[str, Any]:
    """Measure clean IOI baseline logits on a supplied or seeded dataset."""
    if not isinstance(task, str) or task.lower() != "ioi":
        raise NotImplementedError("Only task='ioi' is implemented.")
    dataset = _validate_dataset(ds) if ds is not None else make_dataset(task, n=n, seed=seed)
    from circuitlab.baseline import run_ioi_baseline

    return run_ioi_baseline(dataset)


def patch(
    components: str | list[str], positions: str | list[str] = "final", ds: IOIDataset | None = None,
    *, log_path: Path | str | None = None,
):
    """Patch clean head output into corrupted IOI prompts; return a DataFrame."""
    validated_components = _validate_components(components)
    _validate_positions(positions)
    if ds is None:
        raise TypeError("patch requires ds=IOIDataset.")
    from circuitlab.interpretability.patching import run_activation_patching

    dataset = _validate_dataset(ds)
    from circuitlab.budget import get_tracker

    passes_before = get_tracker().passes_used
    result = run_activation_patching(validated_components, "final", dataset)
    _log_experiment(
        tool="patch", dataset=dataset, components=validated_components, positions="final",
        metric="ioi_logit_difference", result_summary=_dataframe_summary(result, ["effect", "recovery"]),
        passes_used=get_tracker().passes_used - passes_before, parameters={}, log_path=log_path,
    )
    return result


def ablate(
    components: str | list[str], ds: IOIDataset | None = None, mode: str = "mean",
    *, log_path: Path | str | None = None,
):
    """Mean-ablate selected heads on clean IOI prompts; return a DataFrame."""
    validated_components = _validate_components(components)
    if mode != "mean":
        raise NotImplementedError("Only mode='mean' is supported; zero ablation is not implemented.")
    if ds is None:
        raise TypeError("ablate requires ds=IOIDataset.")
    from circuitlab.interpretability.ablation import run_mean_ablation

    dataset = _validate_dataset(ds)
    from circuitlab.budget import get_tracker

    passes_before = get_tracker().passes_used
    result = run_mean_ablation(validated_components, dataset, mode="mean")
    _log_experiment(
        tool="ablate", dataset=dataset, components=validated_components, positions="all_true_prompt_positions",
        metric="ioi_logit_difference", result_summary=_dataframe_summary(result, ["absolute_change", "relative_change"]),
        passes_used=get_tracker().passes_used - passes_before, parameters={"mode": "mean"}, log_path=log_path,
    )
    return result


def path_patch(
    sender: str, receivers: str | list[str], ds: IOIDataset | None = None,
    *, joint_only: bool = False, log_path: Path | str | None = None,
):
    """Trace a sender head through strictly later receiver head outputs."""
    validated_sender = _validate_component(sender, "sender")
    validated_receivers = _validate_components(receivers, "receivers")
    sender_layer = int(_COMPONENT_PATTERN.fullmatch(validated_sender)["layer"])
    if any(int(_COMPONENT_PATTERN.fullmatch(receiver)["layer"]) <= sender_layer for receiver in validated_receivers):
        raise ValueError("Every receiver must be in a strictly later layer than sender.")
    if ds is None:
        raise TypeError("path_patch requires ds=IOIDataset.")
    from circuitlab.interpretability.path_patching import run_path_patching

    dataset = _validate_dataset(ds)
    from circuitlab.budget import get_tracker

    passes_before = get_tracker().passes_used
    result = run_path_patching(validated_sender, validated_receivers, dataset, joint_only=joint_only)
    _log_experiment(
        tool="path_patch", dataset=dataset, components=[validated_sender, *validated_receivers],
        positions="all_true_prompt_positions", metric="ioi_logit_difference",
        result_summary=_dataframe_summary(result, ["path_effect", "recovery"]),
        passes_used=get_tracker().passes_used - passes_before,
        parameters={"sender": validated_sender, "receivers": validated_receivers}, log_path=log_path,
    )
    return result


def faithfulness(
    circuit: str | list[str], ds: IOIDataset | None = None, *, log_path: Path | str | None = None
) -> dict[str, Any]:
    """Measure simultaneous final-token recovery for a head circuit."""
    validated_circuit = _validate_components(circuit, "circuit")
    if ds is None:
        raise TypeError("faithfulness requires ds=IOIDataset.")
    from circuitlab.evaluation.circuit import evaluate_faithfulness

    dataset = _validate_dataset(ds)
    from circuitlab.budget import get_tracker

    passes_before = get_tracker().passes_used
    result = evaluate_faithfulness(validated_circuit, dataset)
    _log_experiment(
        tool="faithfulness", dataset=dataset, components=validated_circuit,
        positions="final", metric="ioi_logit_difference_recovery",
        result_summary={key: result[key] for key in ("clean_logit_diff", "corrupted_logit_diff", "circuit_patched_logit_diff", "faithfulness")},
        passes_used=get_tracker().passes_used - passes_before, parameters={}, log_path=log_path,
    )
    return result


def random_circuit_control(
    circuit: str | list[str], ds: IOIDataset | None = None, n_random: int = 20, seed: int = 42,
    *, log_path: Path | str | None = None,
) -> dict[str, Any]:
    """Compare a circuit with deterministic, same-size random head sets."""
    validated_circuit = _validate_components(circuit, "circuit")
    _validate_positive_int(n_random, "n_random")
    _validate_seed(seed)
    if ds is None:
        raise TypeError("random_circuit_control requires ds=IOIDataset.")
    from circuitlab.evaluation.circuit import random_circuit_control as run_random_circuit_control

    dataset = _validate_dataset(ds)
    from circuitlab.budget import get_tracker

    passes_before = get_tracker().passes_used
    result = run_random_circuit_control(validated_circuit, dataset, n_random=n_random, seed=seed)
    _log_experiment(
        tool="random_circuit_control", dataset=dataset, components=validated_circuit,
        positions="final", metric="ioi_logit_difference_recovery",
        result_summary={key: result[key] for key in ("candidate_faithfulness", "random_mean", "random_std", "percentile", "beats_random", "raw_results_path")},
        passes_used=get_tracker().passes_used - passes_before,
        parameters={"n_random": n_random, "random_seed": seed}, log_path=log_path,
    )
    return result


def score_vs_truth(circuit: str | list[str]) -> dict[str, Any]:
    """Score a stable head circuit against verified repository IOI truth."""
    from circuitlab.evaluation.circuit import score_against_ioi_truth

    return score_against_ioi_truth(_validate_components(circuit, "circuit"))


def recovery_trajectory(
    strategy: str, steps: list[dict[str, Any]], *, seed: int | None = None, output_dir: Path | str | None = None
) -> dict[str, Any]:
    """Persist precision/recall recovery after each cumulative-cost trace step."""
    from circuitlab.evaluation.recovery import DEFAULT_TRAJECTORY_DIRECTORY, evaluate_recovery_trajectory

    return evaluate_recovery_trajectory(
        strategy, steps, seed=seed, output_dir=DEFAULT_TRAJECTORY_DIRECTORY if output_dir is None else output_dir
    )


def speedup(baseline_records: list[dict[str, Any]], circuitlab_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Report finite per-seed baseline/CircuitLab target-cost speedups."""
    from circuitlab.evaluation.speedup import calculate_speedup

    return calculate_speedup(baseline_records, circuitlab_records)


def budget(*, max_passes: int | None | object = _UNSET, reset: bool = False) -> dict[str, Any]:
    """Read the budget, or reset it with an optional non-negative pass cap."""
    if not isinstance(reset, bool):
        raise TypeError("reset must be a boolean.")
    if max_passes is not _UNSET and max_passes is not None and (
        isinstance(max_passes, bool) or not isinstance(max_passes, int) or max_passes < 0
    ):
        raise ValueError("max_passes must be a non-negative integer or None.")
    from circuitlab.budget import get_budget, get_tracker

    if max_passes is not _UNSET:
        if not reset:
            raise ValueError("Setting max_passes requires reset=True so accounting starts from a defined state.")
        return get_tracker().reset(max_passes=max_passes)
    return get_tracker().reset() if reset else get_budget()


def search_literature(query: str) -> list[dict[str, Any]]:
    """Search arXiv for papers on a topic (a few keywords, e.g. "indirect object identification circuit"). Returns up to 5 results with title, arxiv_id and a short snippet. Use to find prior work on a behavior or a component role."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")
    terms = " AND ".join("all:" + w for w in query.split() if len(w) > 3)
    if not terms:
        raise ValueError("query needs at least one word longer than 3 letters.")
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": terms, "max_results": 5})
    ns = {"a": "http://www.w3.org/2005/Atom"}
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            root = ET.fromstring(r.read())
    except Exception as e:
        return [{"error": "search failed: " + str(e)}]
    out = []
    for e in root.findall("a:entry", ns):
        out.append({
            "title": " ".join(e.find("a:title", ns).text.split()),
            "arxiv_id": e.find("a:id", ns).text.rsplit("/abs/", 1)[-1],
            "snippet": " ".join(e.find("a:summary", ns).text.split())[:300],
        })
    return out


def present_claim(circuit: list[str], run_ids: list[str], arxiv_ids: list[str], summary: str) -> dict:
    """Show the final circuit to the human as an agent-generated hypothesis. Call this once, only after the analyst reported faithfulness of at least 0.7 beating the random control and safety cleared the claim. The human must approve before it runs."""
    return {
        "status": "presented",
        "label": "agent-generated hypothesis",
        "circuit": circuit,
        "run_ids": run_ids,
        "arxiv_ids": arxiv_ids,
        "summary": summary,
    }
