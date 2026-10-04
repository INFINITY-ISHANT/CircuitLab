"""Clean-prompt baseline evaluation for IOI."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import torch

from circuitlab.datasets import IOIDataset
from circuitlab.budget import get_tracker
from circuitlab.metrics import logit_diff
from circuitlab.model import get_model


RESULTS_DIRECTORY = Path(__file__).resolve().parents[1] / "results"
DEFAULT_BATCH_SIZE = 32


def _clean_length_groups(dataset: IOIDataset) -> dict[int, list[int]]:
    """Group right-padded prompts so each forward sees no trailing padding."""
    lengths = dataset.metadata.get("prompt_lengths")
    if not isinstance(lengths, list) or len(lengths) != len(dataset):
        raise ValueError("IOIDataset metadata must contain one prompt length per clean example.")

    groups: dict[int, list[int]] = defaultdict(list)
    for index, length in enumerate(lengths):
        if not isinstance(length, int) or length <= 0:
            raise ValueError(f"Invalid clean prompt length at index {index}: {length!r}")
        groups[length].append(index)
    return dict(groups)


def _save_result(result: dict[str, Any]) -> str:
    """Persist scalar/list baseline data without overwriting source code artifacts."""
    RESULTS_DIRECTORY.mkdir(exist_ok=True)
    filename = f"ioi_baseline_seed{result['seed']}_n{result['n']}.json"
    path = RESULTS_DIRECTORY / filename
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path.relative_to(RESULTS_DIRECTORY.parent))


def run_ioi_baseline(dataset: IOIDataset, batch_size: int = DEFAULT_BATCH_SIZE) -> dict[str, Any]:
    """Run GPT-2 Small on clean IOI prompts and persist measured metrics.

    A pass is one model invocation on one length-homogeneous prompt batch. Each
    call is reserved in the shared budget tracker immediately before execution.
    """
    if dataset.metadata.get("task") != "ioi":
        raise ValueError("run_ioi_baseline requires an IOIDataset with task='ioi'.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    model = get_model()
    model.eval()
    device = torch.device(model.cfg.device)
    per_example = torch.empty(len(dataset), dtype=torch.float32)
    forward_passes = 0
    tracker = get_tracker()

    for sequence_length, indices in _clean_length_groups(dataset).items():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            index_tensor = torch.tensor(batch_indices, dtype=torch.long)
            clean_tokens = dataset.clean_tokens.index_select(0, index_tensor)[:, :sequence_length]
            io_ids = dataset.io_token_ids.index_select(0, index_tensor)
            subject_ids = dataset.subject_token_ids.index_select(0, index_tensor)

            tracker.consume(operation="baseline.clean", batch_size=len(batch_indices))
            with torch.inference_mode():
                logits = model(clean_tokens.to(device), return_type="logits")
            scores = logit_diff(logits, io_ids, subject_ids)
            per_example[index_tensor] = scores["per_example"].detach().cpu().to(torch.float32)
            forward_passes += 1

    result: dict[str, Any] = {
        "task": "ioi",
        "dataset_id": dataset.dataset_id,
        "n": len(dataset),
        "seed": dataset.seed,
        "accuracy": float((per_example > 0).sum().item() / len(dataset)),
        "mean_logit_diff": float(per_example.mean().item()),
        "std_logit_diff": float(per_example.std(unbiased=False).item()),
        "per_example_logit_diffs": [float(value) for value in per_example.tolist()],
        "forward_passes": forward_passes,
        "batch_size": batch_size,
        "device": str(device),
        "budget_snapshot": tracker.snapshot(),
    }
    result["result_path"] = _save_result(result)
    return result
