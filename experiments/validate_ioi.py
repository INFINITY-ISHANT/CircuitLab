"""Run and print a reproducible clean IOI baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow `python experiments/validate_ioi.py` from the repository root.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from lab_tools import run_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate GPT-2 Small on clean IOI prompts.")
    parser.add_argument("--n", type=int, default=100, help="Number of seeded IOI prompts.")
    parser.add_argument("--seed", type=int, default=42, help="Dataset random seed.")
    args = parser.parse_args()

    result = run_baseline(task="ioi", n=args.n, seed=args.seed)
    print("IOI clean-prompt baseline")
    print(f"  dataset: {result['dataset_id']}")
    print(f"  device: {result['device']}")
    print(f"  prompts: {result['n']} (seed {result['seed']})")
    print(f"  accuracy: {result['accuracy']:.3f}")
    print(f"  mean logit difference: {result['mean_logit_diff']:.4f}")
    print(f"  std. logit difference: {result['std_logit_diff']:.4f}")
    print(f"  forward passes: {result['forward_passes']}")
    print(f"  saved: {result['result_path']}")


if __name__ == "__main__":
    main()
