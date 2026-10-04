"""Small, honest end-to-end validation for the CircuitLab scientific toolkit."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import lab_tools
from circuitlab.budget import get_tracker
from circuitlab.datasets import load_ioi_truth
from circuitlab.metrics import logit_diff
from circuitlab.model import get_model


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


def run_check(name: str, action: Callable[[], Any]) -> tuple[CheckResult, Any | None]:
    """Run one check and preserve its explicit exception for the final report."""
    try:
        value = action()
    except Exception as error:  # Continue safely; failures remain visible in the summary.
        return CheckResult(name, False, f"{type(error).__name__}: {error}"), None
    return CheckResult(name, True, "ok"), value


def unavailable(name: str, reason: str) -> tuple[CheckResult, None]:
    return CheckResult(name, False, f"not run because {reason}"), None


def _verify_metric() -> None:
    logits = torch.zeros((2, 3, 5))
    logits[0, -1, 1], logits[0, -1, 2] = 4.0, 1.0
    logits[1, -1, 3], logits[1, -1, 4] = 2.0, 5.0
    scores = logit_diff(logits, torch.tensor([1, 3]), torch.tensor([2, 4]))
    torch.testing.assert_close(scores["per_example"], torch.tensor([3.0, -3.0]))
    torch.testing.assert_close(scores["mean"], torch.tensor(0.0))


def main() -> int:
    results: list[CheckResult] = []
    model_result, model = run_check("1. GPT-2 loads", get_model)
    results.append(model_result)

    if model is None:
        dataset_result, dataset = unavailable("2. IOI dataset generation", "GPT-2 did not load")
    else:
        dataset_result, dataset = run_check("2. IOI dataset generation", lambda: lab_tools.make_dataset("ioi", n=1, seed=42))
    results.append(dataset_result)
    tracker = get_tracker()
    tracker.reset(max_passes=None)

    if dataset is None:
        for name in (
            "3. baseline IOI behavior",
            "5. single-head patching",
            "6. mean ablation",
            "7. path patching",
            "10. faithfulness",
            "11. random circuit control",
        ):
            result, _ = unavailable(name, "IOI dataset generation failed")
            results.append(result)
    else:
        result, _ = run_check("3. baseline IOI behavior", lambda: lab_tools.run_baseline(ds=dataset))
        results.append(result)

    result, _ = run_check("4. logit-difference metric", _verify_metric)
    results.append(result)

    if dataset is not None:
        result, _ = run_check("5. single-head patching", lambda: lab_tools.patch("L0H0", ds=dataset))
        results.append(result)
        result, _ = run_check("6. mean ablation", lambda: lab_tools.ablate("L0H0", ds=dataset))
        results.append(result)
        result, _ = run_check("7. path patching", lambda: lab_tools.path_patch("L0H0", "L1H0", ds=dataset))
        results.append(result)

    result, _ = run_check("8. ioi_truth.json loads", load_ioi_truth)
    results.append(result)
    result, _ = run_check("9. score_vs_truth", lambda: lab_tools.score_vs_truth(["L0H0"]))
    results.append(result)

    if dataset is not None:
        result, _ = run_check("10. faithfulness", lambda: lab_tools.faithfulness(["L0H0"], ds=dataset))
        results.append(result)
        result, _ = run_check(
            "11. random circuit control",
            lambda: lab_tools.random_circuit_control(["L0H0"], ds=dataset, n_random=2, seed=42),
        )
        results.append(result)

    def budget_check() -> None:
        snapshot = lab_tools.budget()
        if snapshot["passes_used"] <= 0 or not snapshot["by_operation"]:
            raise AssertionError(f"Expected tracked model executions, got {snapshot}.")

    result, _ = run_check("12. budget tracking", budget_check)
    results.append(result)

    print("CircuitLab toolkit validation")
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{status} | {result.name} | {result.detail}")
    failures = [result for result in results if not result.passed]
    print(f"\n{len(results) - len(failures)}/{len(results)} subsystems passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
