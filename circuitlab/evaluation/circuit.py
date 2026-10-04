"""Faithfulness, random controls, and ground-truth scoring for IOI circuits."""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import random
import re
import statistics
from typing import Any

import torch

from circuitlab.budget import get_tracker
from circuitlab.datasets import IOIDataset, load_ioi_truth, normalise_attention_components, validate_ioi_truth
from circuitlab.interpretability.patching import attention_head_output_hook_name, get_attention_head_output
from circuitlab.metrics import logit_diff
from circuitlab.model import get_model


RESULTS_DIRECTORY = Path(__file__).resolve().parents[2] / "results"
_COMPONENT_PATTERN = re.compile(r"^L(?P<layer>\d+)H(?P<head>\d+)$")
_ALL_HEADS = tuple(f"L{layer}H{head}" for layer in range(12) for head in range(12))


def _component_key(component: str) -> tuple[int, int]:
    match = _COMPONENT_PATTERN.fullmatch(component)
    if match is None:  # Already validated by normalise_attention_components.
        raise ValueError(f"Invalid component {component!r}.")
    return int(match["layer"]), int(match["head"])


def _length_groups(dataset: IOIDataset) -> dict[int, list[int]]:
    lengths = dataset.metadata.get("prompt_lengths")
    if not isinstance(lengths, list) or len(lengths) != len(dataset):
        raise ValueError("IOIDataset metadata must contain one prompt length per example.")
    groups: dict[int, list[int]] = defaultdict(list)
    for index, length in enumerate(lengths):
        if not isinstance(length, int) or length <= 0:
            raise ValueError(f"Invalid prompt length at index {index}: {length!r}")
        groups[length].append(index)
    return dict(groups)


def _final_position_multi_head_hook(clean_cache: dict, layer: int, heads: list[int]):
    """Patch all listed clean heads for one layer at the final prompt token."""
    hook_name = attention_head_output_hook_name(layer)
    clean_outputs = {head: get_attention_head_output(clean_cache, layer, head) for head in heads}

    def patch_heads(corrupted_result: torch.Tensor, hook: Any) -> torch.Tensor:
        if hook.name != hook_name:
            raise RuntimeError(f"Hook attached to {hook.name!r}, expected {hook_name!r}.")
        if corrupted_result.ndim != 4:
            raise ValueError(f"Expected [batch, position, head, d_model], got {tuple(corrupted_result.shape)}.")
        patched = corrupted_result.clone()
        for head, clean_output in clean_outputs.items():
            if corrupted_result.shape[0] != clean_output.shape[0] or corrupted_result.shape[1] != clean_output.shape[1]:
                raise ValueError("Clean and corrupted activations must have matching batch and position dimensions.")
            if corrupted_result.shape[3] != clean_output.shape[2]:
                raise ValueError("Clean and corrupted activations have different d_model widths.")
            patched[:, -1, head, :] = clean_output[:, -1, :].to(
                device=corrupted_result.device, dtype=corrupted_result.dtype
            )
        return patched

    return hook_name, patch_heads


def evaluate_faithfulness(circuit: object, dataset: IOIDataset) -> dict[str, Any]:
    """Measure final-token circuit recovery on paired clean/corrupted IOI prompts.

    Faithfulness is ``(circuit_patched_ld - corrupted_ld) / (clean_ld -
    corrupted_ld)`` using the canonical dataset-mean logit difference. A single
    hooked corrupt forward simultaneously patches every selected clean head,
    grouped by layer. This is the same final-token intervention used by
    ``patch``; it is not a claim that the selected heads form a complete causal
    circuit. One pass is tracked for each length-homogeneous clean cache,
    corrupted baseline, and circuit-patched forward.
    """
    if not isinstance(dataset, IOIDataset) or dataset.metadata.get("task") != "ioi":
        raise TypeError("dataset must be an IOIDataset for task='ioi'.")
    components = sorted(normalise_attention_components(circuit), key=_component_key)
    if not components:
        raise ValueError("faithfulness requires at least one attention head.")

    by_layer: dict[int, list[int]] = defaultdict(list)
    for component in components:
        layer, head = _component_key(component)
        by_layer[layer].append(head)
    model = get_model()
    model.eval()
    tracker = get_tracker()
    passes_before = tracker.passes_used
    hook_names = [attention_head_output_hook_name(layer) for layer in by_layer]
    groups = _length_groups(dataset)
    clean_scores = torch.empty(len(dataset), dtype=torch.float32)
    corrupted_scores = torch.empty(len(dataset), dtype=torch.float32)
    circuit_scores = torch.empty(len(dataset), dtype=torch.float32)

    for sequence_length, indices in groups.items():
        index_tensor = torch.tensor(indices, dtype=torch.long)
        clean_tokens = dataset.clean_tokens.index_select(0, index_tensor)[:, :sequence_length]
        corrupted_tokens = dataset.corrupted_tokens.index_select(0, index_tensor)[:, :sequence_length]
        io_ids = dataset.io_token_ids.index_select(0, index_tensor)
        subject_ids = dataset.subject_token_ids.index_select(0, index_tensor)

        tracker.consume(operation="faithfulness.cache_clean", batch_size=len(indices))
        with torch.inference_mode():
            clean_logits, clean_cache = model.run_with_cache(
                clean_tokens.to(model.cfg.device), names_filter=hook_names, return_type="logits"
            )
        missing = [name for name in hook_names if name not in clean_cache]
        if missing:
            raise RuntimeError(f"Clean cache is missing selected circuit hooks: {missing}")
        tracker.consume(operation="faithfulness.corrupted_baseline", batch_size=len(indices))
        with torch.inference_mode():
            corrupted_logits = model(corrupted_tokens.to(model.cfg.device), return_type="logits")
        hooks = [_final_position_multi_head_hook(clean_cache, layer, heads) for layer, heads in by_layer.items()]
        tracker.consume(operation="faithfulness.circuit_patch", batch_size=len(indices))
        with torch.inference_mode():
            circuit_logits = model.run_with_hooks(
                corrupted_tokens.to(model.cfg.device), fwd_hooks=hooks, return_type="logits"
            )

        clean_scores[index_tensor] = logit_diff(clean_logits, io_ids, subject_ids)["per_example"].cpu()
        corrupted_scores[index_tensor] = logit_diff(corrupted_logits, io_ids, subject_ids)["per_example"].cpu()
        circuit_scores[index_tensor] = logit_diff(circuit_logits, io_ids, subject_ids)["per_example"].cpu()

    clean_ld = float(clean_scores.mean().item())
    corrupted_ld = float(corrupted_scores.mean().item())
    circuit_ld = float(circuit_scores.mean().item())
    denominator = clean_ld - corrupted_ld
    if abs(denominator) < 1e-12:
        raise ValueError("Faithfulness is undefined because clean and corrupted logit differences are equal.")
    return {
        "circuit": components,
        "n": len(dataset),
        "clean_logit_diff": clean_ld,
        "corrupted_logit_diff": corrupted_ld,
        "circuit_patched_logit_diff": circuit_ld,
        "faithfulness": (circuit_ld - corrupted_ld) / denominator,
        "forward_passes": tracker.passes_used - passes_before,
        "budget_snapshot": tracker.snapshot(),
    }


def sample_random_circuits(circuit: object, n_random: int, seed: int) -> list[list[str]]:
    """Draw unique same-size random head sets, excluding the candidate set."""
    candidate = normalise_attention_components(circuit)
    size = len(candidate)
    if size == 0:
        raise ValueError("Random controls require a non-empty candidate circuit.")
    if not isinstance(n_random, int) or n_random <= 0:
        raise ValueError("n_random must be a positive integer.")
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer.")
    possible = math.comb(len(_ALL_HEADS), size) - 1
    if n_random > possible:
        raise ValueError(f"Requested {n_random} controls, but only {possible} non-candidate circuits exist.")

    rng = random.Random(seed)
    sampled_sets: set[frozenset[str]] = set()
    candidate_set = frozenset(candidate)
    while len(sampled_sets) < n_random:
        sampled = frozenset(rng.sample(_ALL_HEADS, size))
        if sampled != candidate_set:
            sampled_sets.add(sampled)
    return [sorted(sampled, key=_component_key) for sampled in sorted(sampled_sets, key=lambda item: tuple(sorted(item, key=_component_key)))]


def _control_path(dataset: IOIDataset, seed: int, size: int, results_dir: Path) -> Path:
    return results_dir / f"random_control_{dataset.dataset_id}_seed{seed}_k{size}.json"


def _write_control_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def random_circuit_control(
    circuit: object,
    ds: IOIDataset,
    n_random: int = 20,
    seed: int = 42,
    *,
    results_dir: Path | str = RESULTS_DIRECTORY,
) -> dict[str, Any]:
    """Compare candidate faithfulness with persisted same-size random controls.

    ``beats_random`` means the candidate faithfulness is strictly greater than
    the sampled random mean. ``percentile`` is its empirical percentile among
    random values: the percentage strictly below the candidate. The raw JSON is
    updated after the candidate and after every random circuit completes.
    """
    candidate = sorted(normalise_attention_components(circuit), key=_component_key)
    sampled_circuits = sample_random_circuits(candidate, n_random=n_random, seed=seed)
    candidate_result = evaluate_faithfulness(candidate, ds)
    path = _control_path(ds, seed, len(candidate), Path(results_dir))
    payload: dict[str, Any] = {
        "task": "ioi",
        "dataset_id": ds.dataset_id,
        "seed": seed,
        "n_random": n_random,
        "candidate_circuit": candidate,
        "candidate_result": candidate_result,
        "random_results": [],
        "status": "running",
    }
    _write_control_result(path, payload)

    random_values: list[float] = []
    for random_circuit in sampled_circuits:
        result = evaluate_faithfulness(random_circuit, ds)
        value = float(result["faithfulness"])
        random_values.append(value)
        payload["random_results"].append(result)
        _write_control_result(path, payload)

    candidate_value = float(candidate_result["faithfulness"])
    random_mean = statistics.fmean(random_values)
    random_std = statistics.pstdev(random_values)
    percentile = 100.0 * sum(value < candidate_value for value in random_values) / len(random_values)
    result = {
        "candidate_faithfulness": candidate_value,
        "random_mean": random_mean,
        "random_std": random_std,
        "random_values": random_values,
        "percentile": percentile,
        "beats_random": candidate_value > random_mean,
        "raw_results_path": str(path),
    }
    payload.update(result, status="complete")
    _write_control_result(path, payload)
    return result


def score_circuit_against_truth(circuit: object, truth: object) -> dict[str, Any]:
    """Score attention-head identities using set membership."""
    verified_truth = validate_ioi_truth(truth, require_verified=True)
    predicted = normalise_attention_components(circuit)
    expected = {entry["component"] for entry in verified_truth["heads"]}
    true_positive = predicted & expected
    false_positive = predicted - expected
    false_negative = expected - predicted
    tp, fp, fn = len(true_positive), len(false_positive), len(false_negative)
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    role_breakdown: dict[str, dict[str, Any]] = {}
    for role in sorted({entry["role"] for entry in verified_truth["heads"]}):
        role_expected = {entry["component"] for entry in verified_truth["heads"] if entry["role"] == role}
        role_recovered = predicted & role_expected
        role_breakdown[role] = {"truth_components": sorted(role_expected), "recovered_components": sorted(role_recovered), "recall": len(role_recovered) / len(role_expected)}

    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "success": recall >= 0.80 and precision >= 0.70, "role_breakdown": role_breakdown}


def score_against_ioi_truth(circuit: object) -> dict[str, Any]:
    """Load the verified repository source and score a predicted head circuit."""
    return score_circuit_against_truth(circuit, load_ioi_truth())
