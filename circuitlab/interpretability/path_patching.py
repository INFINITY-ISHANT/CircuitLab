"""Two-stage attention-head path patching for aligned IOI prompt pairs.

The intervention estimates a sender head's effect mediated by a receiver head:
first patch the sender's clean output into a corrupted bridge run, then patch
the receiver's bridge output into an otherwise corrupted evaluation run.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd
import torch

from circuitlab.budget import get_tracker
from circuitlab.datasets import IOIDataset
from circuitlab.interpretability.patching import (
    AttentionHeadComponent,
    attention_head_output_hook_name,
    get_attention_head_output,
)
from circuitlab.metrics import logit_diff
from circuitlab.model import get_model


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


def _normalise_receivers(receivers: str | list[str] | tuple[str, ...]) -> list[AttentionHeadComponent]:
    raw = [receivers] if isinstance(receivers, str) else list(receivers)
    if not raw:
        raise ValueError("At least one receiver head is required.")
    parsed = [AttentionHeadComponent.parse(component) for component in raw]
    if len({component.name for component in parsed}) != len(parsed):
        raise ValueError("Duplicate receiver heads are not allowed.")
    return parsed


def _make_full_head_replacement_hook(
    source_cache: dict,
    component: AttentionHeadComponent,
):
    """Replace one head's complete ``[batch, position, d_model]`` output."""
    hook_name = attention_head_output_hook_name(component.layer)
    source_output = get_attention_head_output(source_cache, component.layer, component.head)

    def replace_head(target_result: torch.Tensor, hook: Any) -> torch.Tensor:
        if hook.name != hook_name:
            raise RuntimeError(f"Hook attached to {hook.name!r}, expected {hook_name!r}.")
        if target_result.ndim != 4:
            raise ValueError(f"Expected [batch, position, head, d_model], got {tuple(target_result.shape)}.")
        if target_result.shape[0] != source_output.shape[0] or target_result.shape[1] != source_output.shape[1]:
            raise ValueError(
                "Source and target head outputs must have equal batch and position dimensions: "
                f"source {tuple(source_output.shape)}, target {tuple(target_result.shape)}."
            )
        if target_result.shape[3] != source_output.shape[2]:
            raise ValueError("Source and target head-output widths differ.")
        patched = target_result.clone()
        patched[:, :, component.head, :] = source_output.to(device=target_result.device, dtype=target_result.dtype)
        return patched

    return hook_name, replace_head


def _make_multi_receiver_replacement_hooks(
    bridge_cache: dict,
    receivers: list[AttentionHeadComponent],
):
    """Build one full-output replacement hook per receiver; hooks may share a layer."""
    return [_make_full_head_replacement_hook(bridge_cache, receiver) for receiver in receivers]


def run_path_patching(
    sender: str,
    receivers: str | list[str] | tuple[str, ...],
    dataset: IOIDataset,
    joint_only: bool = False,
) -> pd.DataFrame:
    """Estimate clean sender information that reaches receiver outputs.

    Sender and receiver outputs are ``hook_result`` activations with shape
    ``[batch, position, head, d_model]``. The sender is patched at *all* true
    prompt positions into corrupted inputs. Each receiver's bridge-run output is
    then patched at all true positions into a fresh corrupted run. This preserves
    the corrupted computation everywhere except the evaluated receiver output.

    ``path_effect`` is ``path_patched_ld - corrupted_ld`` and ``recovery`` is
    that effect divided by ``clean_ld - corrupted_ld``. Rows are returned for
    each receiver, plus ``receiver='ALL'`` when multiple receivers are supplied.
    ``passes_used`` is cumulative within this call and includes shared clean,
    corrupted, and bridge forwards. With `joint_only=True` only the `ALL` row is
    computed, which costs far fewer forward passes when several receivers are given.
    """
    if not isinstance(dataset, IOIDataset) or dataset.metadata.get("task") != "ioi":
        raise TypeError("dataset must be an IOIDataset for task='ioi'.")
    sender_component = AttentionHeadComponent.parse(sender)
    receiver_components = _normalise_receivers(receivers)
    invalid_receivers = [receiver.name for receiver in receiver_components if receiver.layer <= sender_component.layer]
    if invalid_receivers:
        raise ValueError(
            "Receivers must be in a strictly later layer than the sender because same-layer attention heads "
            f"are computed in parallel; invalid receivers: {invalid_receivers}."
        )

    model = get_model()
    model.eval()
    tracker = get_tracker()
    passes_before = tracker.passes_used
    groups = _length_groups(dataset)
    cache_names = list(dict.fromkeys(
        [attention_head_output_hook_name(sender_component.layer)]
        + [attention_head_output_hook_name(receiver.layer) for receiver in receiver_components]
    ))
    clean_scores = torch.empty(len(dataset), dtype=torch.float32)
    corrupted_scores = torch.empty(len(dataset), dtype=torch.float32)
    bridge_caches: dict[int, dict] = {}

    # Shared setup: source clean outputs, corrupt baseline, then sender-patched
    # bridge outputs for all requested receivers.
    for sequence_length, indices in groups.items():
        index_tensor = torch.tensor(indices, dtype=torch.long)
        clean_tokens = dataset.clean_tokens.index_select(0, index_tensor)[:, :sequence_length]
        corrupted_tokens = dataset.corrupted_tokens.index_select(0, index_tensor)[:, :sequence_length]
        io_ids = dataset.io_token_ids.index_select(0, index_tensor)
        subject_ids = dataset.subject_token_ids.index_select(0, index_tensor)

        tracker.consume(operation="path_patching.cache_clean", batch_size=len(indices))
        with torch.inference_mode():
            clean_logits, clean_cache = model.run_with_cache(
                clean_tokens.to(model.cfg.device), names_filter=cache_names, return_type="logits"
            )
        missing = [name for name in cache_names if name not in clean_cache]
        if missing:
            raise RuntimeError(f"Clean cache is missing requested path-patching hooks: {missing}")
        tracker.consume(operation="path_patching.corrupted_baseline", batch_size=len(indices))
        with torch.inference_mode():
            corrupted_logits = model(corrupted_tokens.to(model.cfg.device), return_type="logits")

        sender_hook = _make_full_head_replacement_hook(clean_cache, sender_component)
        receiver_names = [attention_head_output_hook_name(receiver.layer) for receiver in receiver_components]
        tracker.consume(operation="path_patching.sender_bridge_cache", batch_size=len(indices))
        with torch.inference_mode():
            # TransformerLens 3.9's run_with_cache forwards unknown kwargs to
            # forward, so temporary hooks must be installed through this
            # context manager rather than passed as fwd_hooks=.
            with model.hooks(fwd_hooks=[sender_hook]):
                _, bridge_cache = model.run_with_cache(
                    corrupted_tokens.to(model.cfg.device), names_filter=receiver_names, return_type="logits"
                )
        missing = [name for name in receiver_names if name not in bridge_cache]
        if missing:
            raise RuntimeError(f"Bridge cache is missing receiver hooks: {missing}")

        clean_scores[index_tensor] = logit_diff(clean_logits, io_ids, subject_ids)["per_example"].cpu()
        corrupted_scores[index_tensor] = logit_diff(corrupted_logits, io_ids, subject_ids)["per_example"].cpu()
        bridge_caches[sequence_length] = bridge_cache

    clean_ld = float(clean_scores.mean().item())
    corrupted_ld = float(corrupted_scores.mean().item())
    denominator = clean_ld - corrupted_ld
    if joint_only and len(receiver_components) > 1:
        # One evaluation of all receivers together, skipping the per-receiver rows.
        receiver_sets = [receiver_components]
    else:
        receiver_sets = [[receiver] for receiver in receiver_components]
        if len(receiver_components) > 1:
            receiver_sets.append(receiver_components)
    rows: list[dict[str, object]] = []

    for receiver_set in receiver_sets:
        path_scores = torch.empty(len(dataset), dtype=torch.float32)
        for sequence_length, indices in groups.items():
            index_tensor = torch.tensor(indices, dtype=torch.long)
            corrupted_tokens = dataset.corrupted_tokens.index_select(0, index_tensor)[:, :sequence_length]
            io_ids = dataset.io_token_ids.index_select(0, index_tensor)
            subject_ids = dataset.subject_token_ids.index_select(0, index_tensor)
            hooks = _make_multi_receiver_replacement_hooks(bridge_caches[sequence_length], receiver_set)

            tracker.consume(operation="path_patching.receiver_evaluation", batch_size=len(indices))
            with torch.inference_mode():
                path_logits = model.run_with_hooks(
                    corrupted_tokens.to(model.cfg.device), fwd_hooks=hooks, return_type="logits"
                )
            path_scores[index_tensor] = logit_diff(path_logits, io_ids, subject_ids)["per_example"].cpu()

        path_ld = float(path_scores.mean().item())
        effect = path_ld - corrupted_ld
        receiver_names = [receiver.name for receiver in receiver_set]
        rows.append({
            "sender": sender_component.name,
            "receiver": receiver_names[0] if len(receiver_names) == 1 else "ALL",
            "receiver_components": ";".join(receiver_names),
            "positions": "all_true_prompt_positions",
            "clean_logit_diff": clean_ld,
            "corrupted_logit_diff": corrupted_ld,
            "path_patched_logit_diff": path_ld,
            "path_effect": effect,
            "recovery": float("nan") if abs(denominator) < 1e-12 else effect / denominator,
            "passes_used": tracker.passes_used - passes_before,
        })

    return pd.DataFrame(rows)
