"""Low-level TransformerLens helpers for technically validated head patching.

This module provides the cache, hook-name, one-head replacement, and shared
implementation used by the public component sweep.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from collections import defaultdict
import re
from typing import Any

import pandas as pd
import torch
from transformer_lens import utilities

from circuitlab.budget import get_tracker
from circuitlab.datasets import IOIDataset
from circuitlab.metrics import logit_diff
from circuitlab.model import get_model


HookFunction = Callable[[torch.Tensor, Any], torch.Tensor]
_COMPONENT_PATTERN = re.compile(r"^L(?P<layer>\d+)H(?P<head>\d+)$")


@dataclass(frozen=True)
class AttentionHeadComponent:
    """Stable external representation of one GPT-2 attention head."""

    layer: int
    head: int

    @property
    def name(self) -> str:
        return f"L{self.layer}H{self.head}"

    @classmethod
    def parse(cls, component: str) -> "AttentionHeadComponent":
        if not isinstance(component, str):
            raise TypeError("Components must be strings such as 'L9H9'.")
        match = _COMPONENT_PATTERN.fullmatch(component.upper())
        if match is None:
            raise ValueError(f"Invalid component {component!r}; use stable form 'L<layer>H<head>'.")
        parsed = cls(layer=int(match.group("layer")), head=int(match.group("head")))
        _validate_head_index(parsed.layer, parsed.head)
        return parsed


def attention_head_output_hook_name(layer: int) -> str:
    """Resolve and validate TransformerLens' attention-result hook name.

    ``hook_result`` has shape ``[batch, position, head, d_model]`` in GPT-2
    Small. It is each head's post-output-projection contribution to the residual
    stream, which is the activation patched by these helpers.
    """
    model = get_model()
    if not isinstance(layer, int) or not 0 <= layer < model.cfg.n_layers:
        raise ValueError(f"layer must be in [0, {model.cfg.n_layers}), got {layer!r}.")

    # TransformerLens registers this hook but does not calculate per-head
    # attention results until explicitly requested.
    if not model.cfg.use_attn_result:
        model.set_use_attn_result(True)

    hook_name = utilities.get_act_name("result", layer, "attn")
    if hook_name not in model.hook_dict:
        raise RuntimeError(f"TransformerLens model does not expose expected hook {hook_name!r}.")
    return hook_name


def _validate_head_index(layer: int, head: int) -> None:
    model = get_model()
    if not isinstance(head, int) or not 0 <= head < model.cfg.n_heads:
        raise ValueError(f"head must be in [0, {model.cfg.n_heads}), got {head!r}.")
    attention_head_output_hook_name(layer)


def _run_with_cache(tokens: torch.Tensor, hook_name: str, operation: str) -> tuple[torch.Tensor, dict]:
    """Run one token batch with a selected activation cached and budgeted."""
    if tokens.ndim != 2 or tokens.shape[0] == 0:
        raise ValueError(f"tokens must have non-empty shape [batch, position], got {tuple(tokens.shape)}.")

    model = get_model()
    tracker = get_tracker()
    tracker.consume(operation=operation, batch_size=tokens.shape[0])
    with torch.inference_mode():
        logits, cache = model.run_with_cache(
            tokens.to(model.cfg.device),
            names_filter=hook_name,
            return_type="logits",
        )
    if hook_name not in cache:
        raise RuntimeError(f"Requested hook {hook_name!r} was not present in the activation cache.")
    return logits, cache


def run_clean_with_cache(dataset: IOIDataset, layer: int) -> tuple[torch.Tensor, dict]:
    """Run clean prompts and cache one layer's attention-head outputs."""
    return _run_with_cache(
        dataset.clean_tokens,
        attention_head_output_hook_name(layer),
        operation="patching.cache_clean",
    )


def run_corrupted_with_cache(dataset: IOIDataset, layer: int) -> tuple[torch.Tensor, dict]:
    """Run ABC-corrupted prompts and cache one layer's attention-head outputs."""
    return _run_with_cache(
        dataset.corrupted_tokens,
        attention_head_output_hook_name(layer),
        operation="patching.cache_corrupted",
    )


def get_attention_head_output(cache: dict, layer: int, head: int) -> torch.Tensor:
    """Extract ``[batch, position, d_model]`` output for one attention head."""
    _validate_head_index(layer, head)
    hook_name = attention_head_output_hook_name(layer)
    if hook_name not in cache:
        raise KeyError(f"Activation cache has no entry for {hook_name!r}.")

    result = cache[hook_name]
    if result.ndim != 4:
        raise ValueError(
            f"Expected {hook_name} shape [batch, position, head, d_model], got {tuple(result.shape)}."
        )
    if head >= result.shape[2]:
        raise ValueError(f"Cached result has {result.shape[2]} heads; requested head {head}.")
    return result[:, :, head, :]


def make_attention_head_replacement_hook(
    clean_cache: dict,
    layer: int,
    head: int,
) -> tuple[str, HookFunction]:
    """Build a hook that copies one clean head's output into a corrupt forward.

    The returned hook preserves every other head and all positions. Clean and
    corrupted inputs must be batch- and position-aligned for this causal
    intervention; mismatches raise instead of silently broadcasting.
    """
    _validate_head_index(layer, head)
    hook_name = attention_head_output_hook_name(layer)
    clean_head_output = get_attention_head_output(clean_cache, layer, head)

    def replace_head_output(corrupt_result: torch.Tensor, hook: Any) -> torch.Tensor:
        if hook.name != hook_name:
            raise RuntimeError(f"Replacement hook attached to {hook.name!r}, expected {hook_name!r}.")
        if corrupt_result.ndim != 4:
            raise ValueError(
                f"Expected corrupt result shape [batch, position, head, d_model], "
                f"got {tuple(corrupt_result.shape)}."
            )
        if corrupt_result.shape[0] != clean_head_output.shape[0] or corrupt_result.shape[1] != clean_head_output.shape[1]:
            raise ValueError(
                "Clean and corrupted head outputs must have the same batch and position dimensions: "
                f"clean {tuple(clean_head_output.shape)}, corrupt {tuple(corrupt_result.shape)}."
            )
        if corrupt_result.shape[3] != clean_head_output.shape[2]:
            raise ValueError("Clean and corrupted head-output widths differ.")

        patched = corrupt_result.clone()
        # [batch, position, d_model] replaces exactly one [head] slice.
        patched[:, :, head, :] = clean_head_output.to(
            device=corrupt_result.device,
            dtype=corrupt_result.dtype,
        )
        return patched

    return hook_name, replace_head_output


def make_final_position_head_replacement_hook(
    clean_cache: dict,
    layer: int,
    head: int,
) -> tuple[str, HookFunction]:
    """Build a hook that patches one clean head only at its final token position.

    Inputs must be unpadded or length-homogeneous. The public patch tool groups
    prompts by sequence length before calling this helper, making ``-1`` the
    final relevant prompt token for every example in the batch.
    """
    _validate_head_index(layer, head)
    hook_name = attention_head_output_hook_name(layer)
    clean_head_output = get_attention_head_output(clean_cache, layer, head)

    def replace_final_head_output(corrupt_result: torch.Tensor, hook: Any) -> torch.Tensor:
        if hook.name != hook_name:
            raise RuntimeError(f"Replacement hook attached to {hook.name!r}, expected {hook_name!r}.")
        if corrupt_result.ndim != 4:
            raise ValueError(
                f"Expected corrupt result shape [batch, position, head, d_model], "
                f"got {tuple(corrupt_result.shape)}."
            )
        if corrupt_result.shape[0] != clean_head_output.shape[0] or corrupt_result.shape[1] != clean_head_output.shape[1]:
            raise ValueError(
                "Clean and corrupted head outputs must have the same batch and position dimensions: "
                f"clean {tuple(clean_head_output.shape)}, corrupt {tuple(corrupt_result.shape)}."
            )
        if corrupt_result.shape[3] != clean_head_output.shape[2]:
            raise ValueError("Clean and corrupted head-output widths differ.")

        patched = corrupt_result.clone()
        # At final relevant prompt position, replace only [head, :] for every batch item.
        patched[:, -1, head, :] = clean_head_output[:, -1, :].to(
            device=corrupt_result.device,
            dtype=corrupt_result.dtype,
        )
        return patched

    return hook_name, replace_final_head_output


def run_corrupted_with_head_replacement(
    dataset: IOIDataset,
    clean_cache: dict,
    layer: int,
    head: int,
) -> torch.Tensor:
    """Run corrupted prompts with a single clean attention-head output patched."""
    hook_name, replacement_hook = make_attention_head_replacement_hook(clean_cache, layer, head)
    model = get_model()
    tracker = get_tracker()
    tracker.consume(operation="patching.head_replacement", batch_size=len(dataset))
    with torch.inference_mode():
        return model.run_with_hooks(
            dataset.corrupted_tokens.to(model.cfg.device),
            fwd_hooks=[(hook_name, replacement_hook)],
            return_type="logits",
        )


def _length_groups(dataset: IOIDataset) -> dict[int, list[int]]:
    """Group right-padded paired prompts so the final index is never padding."""
    lengths = dataset.metadata.get("prompt_lengths")
    if not isinstance(lengths, list) or len(lengths) != len(dataset):
        raise ValueError("IOIDataset metadata must contain one prompt length per example.")

    groups: dict[int, list[int]] = defaultdict(list)
    for index, length in enumerate(lengths):
        if not isinstance(length, int) or length <= 0:
            raise ValueError(f"Invalid prompt length at index {index}: {length!r}")
        groups[length].append(index)
    return dict(groups)


def _normalise_components(components: str | list[str] | tuple[str, ...]) -> list[AttentionHeadComponent]:
    raw_components = [components] if isinstance(components, str) else list(components)
    if not raw_components:
        raise ValueError("At least one attention-head component is required.")
    parsed = [AttentionHeadComponent.parse(component) for component in raw_components]
    if len({component.name for component in parsed}) != len(parsed):
        raise ValueError("Duplicate components are not allowed in one patch call.")
    return parsed


def _normalise_positions(positions: str | list[str] | tuple[str, ...]) -> str:
    raw_positions = [positions] if isinstance(positions, str) else list(positions)
    if raw_positions != ["final"]:
        raise NotImplementedError("Only positions='final' is implemented for public activation patching.")
    return "final"


def run_activation_patching(
    components: str | list[str] | tuple[str, ...],
    positions: str | list[str] | tuple[str, ...],
    dataset: IOIDataset,
    *,
    on_result: Callable[[dict[str, object]], None] | None = None,
) -> pd.DataFrame:
    """Patch clean head output into corrupt final-token activations.

    ``recovery`` is ``(patched_ld - corrupted_ld) / (clean_ld - corrupted_ld)``
    using dataset-mean logit differences. ``effect`` is the unnormalised
    ``patched_ld - corrupted_ld``. Recovery is ``NaN`` when the denominator is
    numerically zero. ``passes_used`` is cumulative within this call: it
    includes shared clean/corrupt cache forwards plus each component's patched
    forwards up through that row.
    """
    if not isinstance(dataset, IOIDataset) or dataset.metadata.get("task") != "ioi":
        raise TypeError("dataset must be an IOIDataset for task='ioi'.")
    parsed_components = _normalise_components(components)
    position = _normalise_positions(positions)

    model = get_model()
    model.eval()
    tracker = get_tracker()
    passes_before = tracker.passes_used
    hook_names = [attention_head_output_hook_name(component.layer) for component in parsed_components]
    # Preserve order while caching each requested layer only once per clean/corrupt batch.
    hook_names = list(dict.fromkeys(hook_names))

    groups = _length_groups(dataset)
    clean_caches: dict[int, dict] = {}
    clean_per_example = torch.empty(len(dataset), dtype=torch.float32)
    corrupted_per_example = torch.empty(len(dataset), dtype=torch.float32)

    for sequence_length, indices in groups.items():
        index_tensor = torch.tensor(indices, dtype=torch.long)
        clean_tokens = dataset.clean_tokens.index_select(0, index_tensor)[:, :sequence_length]
        corrupted_tokens = dataset.corrupted_tokens.index_select(0, index_tensor)[:, :sequence_length]
        io_ids = dataset.io_token_ids.index_select(0, index_tensor)
        subject_ids = dataset.subject_token_ids.index_select(0, index_tensor)

        tracker.consume(operation="patching.sweep.cache_clean", batch_size=len(indices))
        with torch.inference_mode():
            clean_logits, clean_cache = model.run_with_cache(
                clean_tokens.to(model.cfg.device), names_filter=hook_names, return_type="logits"
            )
        tracker.consume(operation="patching.sweep.cache_corrupted", batch_size=len(indices))
        with torch.inference_mode():
            corrupted_logits, _ = model.run_with_cache(
                corrupted_tokens.to(model.cfg.device), names_filter=hook_names, return_type="logits"
            )
        missing = [hook_name for hook_name in hook_names if hook_name not in clean_cache]
        if missing:
            raise RuntimeError(f"Clean cache is missing requested attention-result hooks: {missing}")

        clean_per_example[index_tensor] = logit_diff(clean_logits, io_ids, subject_ids)["per_example"].cpu()
        corrupted_per_example[index_tensor] = logit_diff(corrupted_logits, io_ids, subject_ids)["per_example"].cpu()
        clean_caches[sequence_length] = clean_cache

    clean_ld = float(clean_per_example.mean().item())
    corrupted_ld = float(corrupted_per_example.mean().item())
    denominator = clean_ld - corrupted_ld
    rows: list[dict[str, object]] = []

    for component in parsed_components:
        patched_per_example = torch.empty(len(dataset), dtype=torch.float32)
        for sequence_length, indices in groups.items():
            index_tensor = torch.tensor(indices, dtype=torch.long)
            corrupted_tokens = dataset.corrupted_tokens.index_select(0, index_tensor)[:, :sequence_length]
            io_ids = dataset.io_token_ids.index_select(0, index_tensor)
            subject_ids = dataset.subject_token_ids.index_select(0, index_tensor)
            hook_name, replacement_hook = make_final_position_head_replacement_hook(
                clean_caches[sequence_length], component.layer, component.head
            )

            tracker.consume(operation="patching.sweep.head_replacement", batch_size=len(indices))
            with torch.inference_mode():
                patched_logits = model.run_with_hooks(
                    corrupted_tokens.to(model.cfg.device),
                    fwd_hooks=[(hook_name, replacement_hook)],
                    return_type="logits",
                )
            patched_per_example[index_tensor] = logit_diff(patched_logits, io_ids, subject_ids)["per_example"].cpu()

        patched_ld = float(patched_per_example.mean().item())
        effect = patched_ld - corrupted_ld
        recovery = float("nan") if abs(denominator) < 1e-12 else effect / denominator
        row: dict[str, object] = {
            "component": component.name,
            "layer": component.layer,
            "head": component.head,
            "position": position,
            "clean_logit_diff": clean_ld,
            "corrupted_logit_diff": corrupted_ld,
            "patched_logit_diff": patched_ld,
            "recovery": recovery,
            "effect": effect,
            "passes_used": tracker.passes_used - passes_before,
        }
        rows.append(row)
        # The exhaustive experiment persists this immediately, so an interrupt
        # loses at most the head currently being evaluated.
        if on_result is not None:
            on_result(row.copy())

    return pd.DataFrame(rows)


def patch_components(*args: object, **kwargs: object) -> list[dict]:
    """Reserved for the later public activation-patching sweep."""
    raise NotImplementedError("Public activation patching sweep is not implemented yet.")
