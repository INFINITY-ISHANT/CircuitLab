"""Resumable exhaustive final-token attention-head patching for IOI.

Each experiment evaluates the same clean-to-corrupted final-token intervention
and canonical logit-difference metric exposed by ``lab_tools.patch``. The raw
CSV keeps evaluation order; a second CSV is ranked by patching effect.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from circuitlab.budget import get_tracker
from circuitlab.datasets import make_ioi_dataset
from circuitlab.interpretability.patching import run_activation_patching


RESULT_COLUMNS = [
    "component", "layer", "head", "position", "clean_logit_diff",
    "corrupted_logit_diff", "patched_logit_diff", "recovery", "effect",
    "passes_used", "cumulative_forward_passes", "task", "n", "seed",
]


def all_head_components(max_layers: int = 12, heads_per_layer: int = 12) -> list[str]:
    """Return GPT-2 Small attention heads in deterministic layer/head order."""
    if not 1 <= max_layers <= 12:
        raise ValueError("max_layers must be between 1 and 12 for GPT-2 Small.")
    if heads_per_layer != 12:
        raise ValueError("GPT-2 Small has exactly 12 attention heads per layer.")
    return [f"L{layer}H{head}" for layer in range(max_layers) for head in range(heads_per_layer)]


def result_paths(n: int, seed: int, results_dir: Path) -> tuple[Path, Path]:
    """Return the persistent raw and ranked paths for a dataset specification."""
    stem = f"exhaustive_ioi_seed{seed}_n{n}"
    return results_dir / f"{stem}_raw.csv", results_dir / f"{stem}_ranked.csv"


def _read_resumable_rows(raw_path: Path, requested_components: list[str], resume: bool) -> list[dict[str, Any]]:
    if not resume or not raw_path.exists():
        return []
    existing = pd.read_csv(raw_path)
    missing_columns = set(RESULT_COLUMNS) - set(existing.columns)
    if missing_columns:
        raise ValueError(f"Cannot resume {raw_path}: missing columns {sorted(missing_columns)}.")
    invalid = set(existing["component"]) - set(requested_components)
    if invalid:
        raise ValueError(f"Cannot resume {raw_path}: unexpected components {sorted(invalid)}.")
    if existing["component"].duplicated().any():
        raise ValueError(f"Cannot resume {raw_path}: duplicate component rows.")
    return existing[RESULT_COLUMNS].to_dict(orient="records")


def _write_outputs(rows: list[dict[str, Any]], raw_path: Path, ranked_path: Path) -> None:
    """Atomically replace both CSVs after each completed component."""
    frame = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    raw_temp = raw_path.with_suffix(".raw.tmp")
    ranked_temp = ranked_path.with_suffix(".ranked.tmp")
    frame.to_csv(raw_temp, index=False)
    frame.sort_values("effect", ascending=False, kind="stable").to_csv(ranked_temp, index=False)
    raw_temp.replace(raw_path)
    ranked_temp.replace(ranked_path)


def run_exhaustive(
    *,
    n: int = 100,
    seed: int = 42,
    max_layers: int = 12,
    resume: bool = True,
    results_dir: Path | str = REPOSITORY_ROOT / "results",
) -> pd.DataFrame:
    """Evaluate requested heads and persist an interrupt-safe CSV after each.

    ``cumulative_forward_passes`` is the run total stored in the raw CSV. On a
    resumed process it begins at the final completed row and includes the new
    cache forwards required by that process. A forward pass is one model call on
    one batch, as defined by :class:`circuitlab.budget.BudgetTracker`.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    components = all_head_components(max_layers=max_layers)
    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path, ranked_path = result_paths(n, seed, output_dir)
    rows = _read_resumable_rows(raw_path, components, resume)
    completed = {str(row["component"]) for row in rows}
    remaining = [component for component in components if component not in completed]

    if not remaining:
        _write_outputs(rows, raw_path, ranked_path)
        return pd.DataFrame(rows, columns=RESULT_COLUMNS)

    # Materialise exactly the seeded dataset used for every newly evaluated
    # head. A fully completed resume does not load the model or execute a pass.
    dataset = make_ioi_dataset(n=n, seed=seed)

    cumulative = int(rows[-1]["cumulative_forward_passes"]) if rows else 0
    previous_local_passes = 0

    def persist_row(row: dict[str, object]) -> None:
        nonlocal cumulative, previous_local_passes
        local_passes = int(row["passes_used"])
        # Convert per-call cumulative accounting to a per-head increment.
        cumulative += local_passes - previous_local_passes
        previous_local_passes = local_passes
        saved_row: dict[str, Any] = dict(row)
        saved_row.update(cumulative_forward_passes=cumulative, task="ioi", n=n, seed=seed)
        rows.append(saved_row)
        _write_outputs(rows, raw_path, ranked_path)

    run_activation_patching(remaining, "final", dataset, on_result=persist_row)
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100, help="Number of seeded IOI examples (default: 100).")
    parser.add_argument("--seed", type=int, default=42, help="IOI dataset seed (default: 42).")
    parser.add_argument("--max-layers", type=int, default=12, help="Evaluate layers [0, max-layers); 12 runs all heads.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True, help="Resume existing raw CSV.")
    parser.add_argument("--results-dir", type=Path, default=REPOSITORY_ROOT / "results")
    args = parser.parse_args()

    results = run_exhaustive(n=args.n, seed=args.seed, max_layers=args.max_layers, resume=args.resume, results_dir=args.results_dir)
    raw_path, ranked_path = result_paths(args.n, args.seed, args.results_dir)
    ranked = results.sort_values("effect", ascending=False, kind="stable")
    print(f"Completed {len(results)}/{args.max_layers * 12} heads for IOI n={args.n}, seed={args.seed}.")
    print(f"Raw evaluation order: {raw_path}")
    print(f"Ranked by effect:     {ranked_path}")
    print("\nTop heads by patching effect:")
    print(ranked.head(10).to_string(index=False))
    print(f"\nForward-pass tracker: {get_tracker().snapshot()}")


if __name__ == "__main__":
    main()
