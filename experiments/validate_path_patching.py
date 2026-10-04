"""Run one source-backed IOI attention-head path-patching validation."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from circuitlab.datasets import load_ioi_truth


def _role_key(role: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", role.lower()).strip("_")
    return key[:-1] if key.endswith("s") else key


def select_documented_path_pair(truth: dict[str, Any]) -> tuple[str, str]:
    """Select a source-listed earlier head and later name mover, without inference."""
    receiver_roles = {"name_mover", "primary_name_mover", "backup_name_mover"}
    receivers = [entry for entry in truth["heads"] if _role_key(entry["role"]) in receiver_roles]
    senders = [entry for entry in truth["heads"] if _role_key(entry["role"]) not in receiver_roles]
    pairs = [(sender, receiver) for sender in senders for receiver in receivers if sender["layer"] < receiver["layer"]]
    if not pairs:
        raise ValueError(
            "Verified truth must explicitly contain a non-name-mover head in an earlier layer and a later name mover "
            "for this validation; no sender/receiver pair was inferred."
        )
    sender, receiver = sorted(pairs, key=lambda pair: (pair[1]["layer"], pair[1]["head"], pair[0]["layer"], pair[0]["head"]))[0]
    return sender["component"], receiver["component"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=Path, default=REPOSITORY_ROOT / "results")
    args = parser.parse_args()
    try:
        truth = load_ioi_truth()
        sender, receiver = select_documented_path_pair(truth)
    except ValueError as error:
        parser.exit(2, f"Cannot run path-patching validation: {error}\n")

    from circuitlab.datasets import make_ioi_dataset
    from circuitlab.interpretability.path_patching import run_path_patching

    results = run_path_patching(sender, [receiver], make_ioi_dataset(n=args.n, seed=args.seed))
    args.results_dir.mkdir(parents=True, exist_ok=True)
    path = args.results_dir / f"ioi_path_patching_validation_seed{args.seed}_n{args.n}.json"
    path.write_text(json.dumps({"sender": sender, "receiver": receiver, "rows": results.to_dict(orient="records")}, indent=2) + "\n", encoding="utf-8")
    print(f"Path patch {sender} -> {receiver}")
    print(results.to_string(index=False))
    print(f"Raw result: {path}")


if __name__ == "__main__":
    main()
