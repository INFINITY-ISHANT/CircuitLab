"""Central, in-process accounting for experimental model executions.

Counting convention: one pass is one call to ``HookedTransformer`` with one
batch of prompts. A batch of 1 prompt and a batch of 100 prompts each consume
one pass. Batch size is recorded by the calling experiment so comparisons use
the same dataset and batching policy.
"""

from __future__ import annotations

from threading import Lock
from typing import Final


_UNSET: Final = object()


class BudgetTracker:
    """Track model batch-forward calls and optionally enforce a maximum."""

    def __init__(self, max_passes: int | None = None) -> None:
        self._lock = Lock()
        self._max_passes = self._validate_max_passes(max_passes)
        self._passes_used = 0
        self._by_operation: dict[str, int] = {}

    @staticmethod
    def _validate_max_passes(max_passes: int | None) -> int | None:
        if max_passes is not None and (not isinstance(max_passes, int) or max_passes < 0):
            raise ValueError("max_passes must be a non-negative integer or None.")
        return max_passes

    @property
    def passes_used(self) -> int:
        with self._lock:
            return self._passes_used

    @property
    def passes_left(self) -> int | None:
        with self._lock:
            return None if self._max_passes is None else self._max_passes - self._passes_used

    def reset(self, max_passes: int | None | object = _UNSET) -> dict:
        """Clear use history, retaining the configured maximum unless replaced."""
        with self._lock:
            if max_passes is not _UNSET:
                self._max_passes = self._validate_max_passes(max_passes)  # type: ignore[arg-type]
            self._passes_used = 0
            self._by_operation.clear()
            return self._snapshot_unlocked()

    def consume(
        self,
        passes: int = 1,
        *,
        operation: str = "unspecified",
        batch_size: int | None = None,
    ) -> dict:
        """Reserve and record one or more experimental model batch forwards.

        Call immediately before model execution. If a maximum is configured,
        this fails atomically before an over-budget model call can begin.
        ``batch_size`` is provenance only and does not change pass cost.
        """
        if not isinstance(passes, int) or passes <= 0:
            raise ValueError("passes must be a positive integer.")
        if not isinstance(operation, str) or not operation:
            raise ValueError("operation must be a non-empty string.")
        if batch_size is not None and (not isinstance(batch_size, int) or batch_size <= 0):
            raise ValueError("batch_size must be a positive integer or None.")

        with self._lock:
            if self._max_passes is not None and self._passes_used + passes > self._max_passes:
                raise RuntimeError(
                    f"Forward-pass budget exceeded: requested {passes}, "
                    f"used {self._passes_used}, maximum {self._max_passes}."
                )
            self._passes_used += passes
            self._by_operation[operation] = self._by_operation.get(operation, 0) + passes
            return self._snapshot_unlocked()

    def _snapshot_unlocked(self) -> dict:
        passes_left = None if self._max_passes is None else self._max_passes - self._passes_used
        return {
            "passes_used": self._passes_used,
            "passes_left": passes_left,
            "max_passes": self._max_passes,
            "by_operation": dict(self._by_operation),
        }

    def snapshot(self) -> dict:
        """Return a JSON-compatible view of the current accounting state."""
        with self._lock:
            return self._snapshot_unlocked()


_TRACKER = BudgetTracker()


def get_tracker() -> BudgetTracker:
    """Return the process-wide tracker used by all scientific lab tools."""
    return _TRACKER


def get_budget() -> dict:
    """Return the process-wide forward-pass budget snapshot."""
    return _TRACKER.snapshot()
