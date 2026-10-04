"""Canonical behavior metrics for CircuitLab experiments."""

from collections.abc import Sequence

import torch


def _validate_token_ids(
    token_ids: torch.Tensor | Sequence[int],
    *,
    name: str,
    batch_size: int,
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Convert and validate one answer-token ID for every batch item."""
    ids = torch.as_tensor(token_ids, dtype=torch.long, device=device)
    if ids.ndim != 1:
        raise ValueError(f"{name} must have shape [batch], got {tuple(ids.shape)}.")
    if ids.shape[0] != batch_size:
        raise ValueError(f"{name} has {ids.shape[0]} IDs, expected {batch_size}.")
    if torch.any(ids < 0) or torch.any(ids >= vocab_size):
        raise ValueError(f"{name} contains token IDs outside [0, {vocab_size}).")
    return ids


def logit_diff(
    logits: torch.Tensor,
    io_token_ids: torch.Tensor | Sequence[int],
    subject_token_ids: torch.Tensor | Sequence[int],
) -> dict[str, torch.Tensor]:
    """Compute IOI logit difference at each prompt's final sequence position.

    This is the single canonical definition used by future baselines,
    interventions, faithfulness checks, and controls. It operates on raw logits
    (not probabilities): for batch item ``i`` it returns
    ``logits[i, -1, io_token_ids[i]] - logits[i, -1, subject_token_ids[i]]``.

    Args:
        logits: Raw model output with shape ``[batch, sequence, vocab]``.
        io_token_ids: Correct indirect-object token IDs, shape ``[batch]``.
        subject_token_ids: Subject token IDs, shape ``[batch]``.

    Returns:
        A dictionary containing ``per_example`` (shape ``[batch]``) and
        ``mean`` (a scalar tensor).
    """
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a torch.Tensor.")
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [batch, sequence, vocab], got {tuple(logits.shape)}.")
    if logits.shape[0] == 0 or logits.shape[1] == 0:
        raise ValueError("logits must have non-empty batch and sequence dimensions.")

    batch_size, _, vocab_size = logits.shape
    io_ids = _validate_token_ids(
        io_token_ids,
        name="io_token_ids",
        batch_size=batch_size,
        vocab_size=vocab_size,
        device=logits.device,
    )
    subject_ids = _validate_token_ids(
        subject_token_ids,
        name="subject_token_ids",
        batch_size=batch_size,
        vocab_size=vocab_size,
        device=logits.device,
    )

    final_logits = logits[:, -1, :]
    batch_indices = torch.arange(batch_size, device=logits.device)
    per_example = final_logits[batch_indices, io_ids] - final_logits[batch_indices, subject_ids]
    return {"per_example": per_example, "mean": per_example.mean()}
