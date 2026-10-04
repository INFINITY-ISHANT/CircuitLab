"""Mean ablation of GPT-2 attention-head outputs for IOI.

The reference distribution is the paired ABC-corrupted IOI dataset. For every
length-homogeneous batch, the mean is taken *only over reference batch items*
(``dim=0``). It therefore retains position, head, and ``d_model`` dimensions:
``[reference_batch, position, head, d_model] -> [1, position, head, d_model]``.
The resulting selected-head slice is explicitly expanded to the clean batch;
no implicit broadcasting is used.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

import pandas as pd
import torch

from circuitlab.budget import get_tracker
from circuitlab.datasets import IOIDataset
from circuitlab.metrics import logit_diff
from circuitlab.model import get_model
from circuitlab.interpretability.patching import (
    AttentionHeadComponent,
    attention_head_output_hook_name,
    get_attention_head_output,
)


HookFunction = Callable[[torch.Tensor, Any], torch.Tensor]


def _length_groups(dataset: IOIDataset) -> dict[int, list[int]]:
    """Return index groups whose clean/corrupted prompts share a true length."""
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
        raise ValueError("Duplicate components are not allowed in one ablation call.")
    return parsed


def make_mean_head_ablation_hook(
    reference_cache: dict,
    layer: int,
    head: int,
) -> tuple[str, HookFunction]:
    """Create a hook that replaces a head with a corrupted-reference batch mean.

    The reference cache entry must have shape ``[reference_batch, position,
    head, d_model]``. Its selected head is averaged across ``reference_batch``
    only, producing ``[1, position, d_model]``. At intervention time that
    tensor is explicitly expanded across the clean batch and replaces the
    selected head at every non-padding position in its length-homogeneous group.
    """
    hook_name = attention_head_output_hook_name(layer)
    reference_head_output = get_attention_head_output(reference_cache, layer, head)
    if reference_head_output.ndim != 3 or reference_head_output.shape[0] == 0:
        raise ValueError(
            "Reference selected-head output must have non-empty shape "
            f"[batch, position, d_model], got {tuple(reference_head_output.shape)}."
        )
    # Deliberately mean over examples only; positions do not mix.
    reference_mean = reference_head_output.mean(dim=0, keepdim=True)

    def replace_with_reference_mean(clean_result: torch.Tensor, hook: Any) -> torch.Tensor:
        if hook.name != hook_name:
            raise RuntimeError(f"Mean-ablation hook attached to {hook.name!r}, expected {hook_name!r}.")
        if clean_result.ndim != 4:
            raise ValueError(
                "Expected clean attention result shape [batch, position, head, d_model], "
                f"got {tuple(clean_result.shape)}."
            )
        if clean_result.shape[1] != reference_mean.shape[1] or clean_result.shape[3] != reference_mean.shape[2]:
            raise ValueError(
                "Reference mean and clean activation disagree on position or d_model dimensions: "
                f"reference {tuple(reference_mean.shape)}, clean {tuple(clean_result.shape)}."
            )
        if not 0 <= head < clean_result.shape[2]:
            raise ValueError(f"Clean activation has {clean_result.shape[2]} heads; requested {head}.")

        # Explicit expansion prevents a batch-shaped source from being silently
        # assigned to an incompatible target batch.
        mean_for_clean_batch = reference_mean.to(device=clean_result.device, dtype=clean_result.dtype).expand(
            clean_result.shape[0], -1, -1
        )
        ablated = clean_result.clone()
        ablated[:, :, head, :] = mean_for_clean_batch
        return ablated

    return hook_name, replace_with_reference_mean


def _make_multi_head_mean_ablation_hook(
    reference_cache: dict,
    layer: int,
    heads: list[int],
) -> tuple[str, HookFunction]:
    """Mean-ablate several heads in one layer in a single hooked forward."""
    if not heads or len(set(heads)) != len(heads):
        raise ValueError("heads must be a non-empty list of distinct head indices.")
    hook_name = attention_head_output_hook_name(layer)
    reference_means = {
        head: get_attention_head_output(reference_cache, layer, head).mean(dim=0, keepdim=True)
        for head in heads
    }

    def replace_with_reference_means(clean_result: torch.Tensor, hook: Any) -> torch.Tensor:
        if hook.name != hook_name:
            raise RuntimeError(f"Mean-ablation hook attached to {hook.name!r}, expected {hook_name!r}.")
        if clean_result.ndim != 4:
            raise ValueError(f"Expected [batch, position, head, d_model], got {tuple(clean_result.shape)}.")
        ablated = clean_result.clone()
        for head, reference_mean in reference_means.items():
            if clean_result.shape[1] != reference_mean.shape[1] or clean_result.shape[3] != reference_mean.shape[2]:
                raise ValueError("Reference mean and clean activation disagree on position or d_model dimensions.")
            ablated[:, :, head, :] = reference_mean.to(
                device=clean_result.device, dtype=clean_result.dtype
            ).expand(clean_result.shape[0], -1, -1)
        return ablated

    return hook_name, replace_with_reference_means


def run_joint_mean_ablation(
    components: str | list[str] | tuple[str, ...],
    dataset: IOIDataset,
) -> dict[str, object]:
    """Mean-ablate all selected heads simultaneously on clean IOI prompts.

    This is intended for circuit-level experiments. The activation means use
    corrupted reference prompts and average over examples only, as in
    :func:`run_mean_ablation`. It reports only the ablated LD; callers that need
    a full-model comparison should obtain the clean baseline separately.
    """
    if not isinstance(dataset, IOIDataset) or dataset.metadata.get("task") != "ioi":
        raise TypeError("dataset must be an IOIDataset for task='ioi'.")
    parsed_components = _normalise_components(components)
    by_layer: dict[int, list[int]] = defaultdict(list)
    for component in parsed_components:
        by_layer[component.layer].append(component.head)

    model = get_model()
    model.eval()
    tracker = get_tracker()
    passes_before = tracker.passes_used
    hook_names = [attention_head_output_hook_name(layer) for layer in by_layer]
    ablated_per_example = torch.empty(len(dataset), dtype=torch.float32)

    for sequence_length, indices in _length_groups(dataset).items():
        index_tensor = torch.tensor(indices, dtype=torch.long)
        clean_tokens = dataset.clean_tokens.index_select(0, index_tensor)[:, :sequence_length]
        reference_tokens = dataset.corrupted_tokens.index_select(0, index_tensor)[:, :sequence_length]
        io_ids = dataset.io_token_ids.index_select(0, index_tensor)
        subject_ids = dataset.subject_token_ids.index_select(0, index_tensor)

        tracker.consume(operation="ablation.joint_reference_cache_corrupted", batch_size=len(indices))
        with torch.inference_mode():
            _, reference_cache = model.run_with_cache(
                reference_tokens.to(model.cfg.device), names_filter=hook_names, return_type="logits"
            )
        missing = [hook_name for hook_name in hook_names if hook_name not in reference_cache]
        if missing:
            raise RuntimeError(f"Corrupted reference cache is missing requested attention-result hooks: {missing}")
        hooks = [_make_multi_head_mean_ablation_hook(reference_cache, layer, heads) for layer, heads in by_layer.items()]
        tracker.consume(operation="ablation.joint_mean_replacement", batch_size=len(indices))
        with torch.inference_mode():
            ablated_logits = model.run_with_hooks(
                clean_tokens.to(model.cfg.device), fwd_hooks=hooks, return_type="logits"
            )
        ablated_per_example[index_tensor] = logit_diff(ablated_logits, io_ids, subject_ids)["per_example"].cpu()

    return {
        "components": [component.name for component in parsed_components],
        "mode": "mean",
        "mean_source": "corrupted_prompts_batch_mean_over_examples",
        "ablated_logit_diff": float(ablated_per_example.mean().item()),
        "forward_passes": tracker.passes_used - passes_before,
        "budget_snapshot": tracker.snapshot(),
    }


def run_mean_ablation(
    components: str | list[str] | tuple[str, ...],
    dataset: IOIDataset,
    *,
    mode: str = "mean",
) -> pd.DataFrame:
    """Individually mean-ablate requested heads during clean IOI evaluation.

    The full-model baseline is the clean prompt logit difference. Each row
    independently replaces one clean head with the matching corrupted-reference
    mean, so rows are directly comparable. ``signed_change`` is
    ``ablated_ld - baseline_ld``; ``absolute_change`` is its magnitude;
    ``relative_change`` is the signed change divided by ``abs(baseline_ld)``
    (or NaN if the baseline is zero). ``passes_used`` is cumulative within this
    call and includes clean baselines, reference-cache forwards, and ablations.
    """
    if not isinstance(dataset, IOIDataset) or dataset.metadata.get("task") != "ioi":
        raise TypeError("dataset must be an IOIDataset for task='ioi'.")
    if mode != "mean":
        raise NotImplementedError("Only mode='mean' is implemented; zero ablation is intentionally unsupported.")

    parsed_components = _normalise_components(components)
    model = get_model()
    model.eval()
    tracker = get_tracker()
    passes_before = tracker.passes_used
    groups = _length_groups(dataset)
    hook_names = list(dict.fromkeys(attention_head_output_hook_name(component.layer) for component in parsed_components))

    baseline_per_example = torch.empty(len(dataset), dtype=torch.float32)
    reference_caches: dict[int, dict] = {}
    for sequence_length, indices in groups.items():
        index_tensor = torch.tensor(indices, dtype=torch.long)
        clean_tokens = dataset.clean_tokens.index_select(0, index_tensor)[:, :sequence_length]
        reference_tokens = dataset.corrupted_tokens.index_select(0, index_tensor)[:, :sequence_length]
        io_ids = dataset.io_token_ids.index_select(0, index_tensor)
        subject_ids = dataset.subject_token_ids.index_select(0, index_tensor)

        tracker.consume(operation="ablation.clean_baseline", batch_size=len(indices))
        with torch.inference_mode():
            clean_logits = model(clean_tokens.to(model.cfg.device), return_type="logits")
        baseline_per_example[index_tensor] = logit_diff(clean_logits, io_ids, subject_ids)["per_example"].cpu()

        tracker.consume(operation="ablation.reference_cache_corrupted", batch_size=len(indices))
        with torch.inference_mode():
            _, reference_cache = model.run_with_cache(
                reference_tokens.to(model.cfg.device), names_filter=hook_names, return_type="logits"
            )
        missing = [hook_name for hook_name in hook_names if hook_name not in reference_cache]
        if missing:
            raise RuntimeError(f"Corrupted reference cache is missing requested attention-result hooks: {missing}")
        reference_caches[sequence_length] = reference_cache

    baseline_ld = float(baseline_per_example.mean().item())
    rows: list[dict[str, object]] = []
    for component in parsed_components:
        ablated_per_example = torch.empty(len(dataset), dtype=torch.float32)
        for sequence_length, indices in groups.items():
            index_tensor = torch.tensor(indices, dtype=torch.long)
            clean_tokens = dataset.clean_tokens.index_select(0, index_tensor)[:, :sequence_length]
            io_ids = dataset.io_token_ids.index_select(0, index_tensor)
            subject_ids = dataset.subject_token_ids.index_select(0, index_tensor)
            hook_name, mean_hook = make_mean_head_ablation_hook(
                reference_caches[sequence_length], component.layer, component.head
            )

            tracker.consume(operation="ablation.mean_head_replacement", batch_size=len(indices))
            with torch.inference_mode():
                ablated_logits = model.run_with_hooks(
                    clean_tokens.to(model.cfg.device), fwd_hooks=[(hook_name, mean_hook)], return_type="logits"
                )
            ablated_per_example[index_tensor] = logit_diff(ablated_logits, io_ids, subject_ids)["per_example"].cpu()

        ablated_ld = float(ablated_per_example.mean().item())
        signed_change = ablated_ld - baseline_ld
        rows.append(
            {
                "component": component.name,
                "layer": component.layer,
                "head": component.head,
                "mode": mode,
                "mean_source": "corrupted_prompts_batch_mean_over_examples",
                "baseline_logit_diff": baseline_ld,
                "ablated_logit_diff": ablated_ld,
                "signed_change": signed_change,
                "absolute_change": abs(signed_change),
                "relative_change": float("nan") if abs(baseline_ld) < 1e-12 else signed_change / abs(baseline_ld),
                "passes_used": tracker.passes_used - passes_before,
            }
        )

    return pd.DataFrame(rows)
