"""Source-gated measurement of the documented IOI backup-name-mover effect.

The experiment reads primary and backup name-mover identities exclusively from
the verified repository truth file. It records clean IOI performance, joint
mean ablation of primary movers, then primary-plus-backup ablations. A further
loss after removing a backup is a measurement compatible with compensation; it
is not by itself a scientific claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from circuitlab.baseline import run_ioi_baseline
from circuitlab.budget import get_tracker
from circuitlab.datasets import IOIDataset, load_ioi_truth
from circuitlab.interpretability.ablation import run_joint_mean_ablation


def _normalise_role(role: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", role.lower()).strip("_")
    return key[:-1] if key.endswith("s") else key


def documented_name_mover_heads(truth: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return source-listed primary and backup movers; never infer identities."""
    primary, backup = [], []
    for entry in truth["heads"]:
        role = _normalise_role(entry["role"])
        if role in {"name_mover", "primary_name_mover"}:
            primary.append(entry["component"])
        elif role == "backup_name_mover":
            backup.append(entry["component"])
    if not primary or not backup:
        roles = sorted({entry["role"] for entry in truth["heads"]})
        raise ValueError(
            "Verified IOI truth must explicitly contain both primary name-mover and backup name-mover roles; "
            f"found roles: {roles}. No heads were inferred."
        )
    return sorted(set(primary)), sorted(set(backup))


def _write_outputs(rows: list[dict[str, Any]], metadata: dict[str, Any], output_dir: Path, seed: int, n: int) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"backup_name_movers_ioi_seed{seed}_n{n}"
    json_path, csv_path = output_dir / f"{stem}.json", output_dir / f"{stem}.csv"
    json_path.write_text(json.dumps({"metadata": metadata, "measurements": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def run_backup_name_mover_experiment(
    ds: IOIDataset,
    *,
    output_dir: Path | str = REPOSITORY_ROOT / "results",
) -> dict[str, Any]:
    """Run source-backed backup-name-mover measurements and save raw outputs."""
    truth = load_ioi_truth()  # Rejects the unverified placeholder before any model pass.
    primary, backups = documented_name_mover_heads(truth)
    tracker = get_tracker()
    passes_before = tracker.passes_used
    rows: list[dict[str, Any]] = []

    baseline = run_ioi_baseline(ds)
    baseline_ld = float(baseline["mean_logit_diff"])
    rows.append({
        "condition": "clean_baseline",
        "components": "",
        "component_count": 0,
        "logit_diff": baseline_ld,
        "change_from_baseline": 0.0,
        "change_vs_primary_ablation": 0.0,
        "forward_passes": int(baseline["forward_passes"]),
        "cumulative_forward_passes": tracker.passes_used - passes_before,
    })

    def measure(condition: str, components: list[str], primary_ld: float | None = None) -> float:
        ablated = run_joint_mean_ablation(components, ds)
        ld = float(ablated["ablated_logit_diff"])
        rows.append({
            "condition": condition,
            "components": ";".join(components),
            "component_count": len(components),
            "logit_diff": ld,
            "change_from_baseline": ld - baseline_ld,
            "change_vs_primary_ablation": 0.0 if primary_ld is None else ld - primary_ld,
            "forward_passes": int(ablated["forward_passes"]),
            "cumulative_forward_passes": tracker.passes_used - passes_before,
        })
        return ld

    primary_ld = measure("primary_name_movers_mean_ablated", primary)
    for backup in backups:
        measure(f"primary_plus_{backup}_mean_ablated", primary + [backup], primary_ld)
    all_backup_ld = measure("primary_plus_all_backup_name_movers_mean_ablated", primary + backups, primary_ld)

    metadata = {
        "task": "ioi",
        "dataset_id": ds.dataset_id,
        "seed": ds.seed,
        "n": len(ds),
        "truth_source": truth["source"],
        "truth_status": truth["status"],
        "primary_name_movers": primary,
        "backup_name_movers": backups,
        "interpretation_note": (
            "A negative change_vs_primary_ablation after additionally ablating a backup is a measured "
            "compensation signal to inspect, not a standalone reproduction claim."
        ),
        "total_forward_passes": tracker.passes_used - passes_before,
        "budget_snapshot": tracker.snapshot(),
    }
    json_path, csv_path = _write_outputs(rows, metadata, Path(output_dir), ds.seed, len(ds))
    return {"metadata": metadata, "measurements": rows, "json_path": str(json_path), "csv_path": str(csv_path), "all_backups_logit_diff": all_backup_ld}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=Path, default=REPOSITORY_ROOT / "results")
    args = parser.parse_args()

    # Load the source before making/tokenizing a dataset or loading the model.
    try:
        truth = load_ioi_truth()
        primary, backups = documented_name_mover_heads(truth)
    except ValueError as error:
        parser.exit(2, f"Cannot run backup-name-mover experiment: {error}\n")
    from circuitlab.datasets import make_ioi_dataset

    result = run_backup_name_mover_experiment(make_ioi_dataset(n=args.n, seed=args.seed), output_dir=args.results_dir)
    print("IOI backup-name-mover experiment")
    print(f"Primary movers: {', '.join(primary)}")
    print(f"Backup movers:  {', '.join(backups)}")
    for row in result["measurements"]:
        print(f"{row['condition']}: LD={row['logit_diff']:.4f}, delta from baseline={row['change_from_baseline']:.4f}")
    print(f"Raw JSON: {result['json_path']}")
    print(f"Raw CSV:  {result['csv_path']}")
    print(f"Forward passes: {result['metadata']['total_forward_passes']}")


if __name__ == "__main__":
    main()
