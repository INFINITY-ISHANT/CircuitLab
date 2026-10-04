"""Technical proof that one TransformerLens attention-head patch executes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Allow `python experiments/diagnose_head_patching.py` from the repository root.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from circuitlab.budget import get_budget
from circuitlab.datasets import make_ioi_dataset
from circuitlab.interpretability.patching import (
    attention_head_output_hook_name,
    get_attention_head_output,
    run_clean_with_cache,
    run_corrupted_with_cache,
    run_corrupted_with_head_replacement,
)
from circuitlab.metrics import logit_diff


RESULTS_DIRECTORY = REPOSITORY_ROOT / "results"


def run_diagnostic(seed: int = 42, layer: int = 0, head: int = 0) -> dict:
    """Patch one head on one aligned IOI example; this is not an importance test."""
    dataset = make_ioi_dataset(n=1, seed=seed)
    hook_name = attention_head_output_hook_name(layer)
    clean_logits, clean_cache = run_clean_with_cache(dataset, layer)
    corrupt_logits, corrupt_cache = run_corrupted_with_cache(dataset, layer)
    patched_logits = run_corrupted_with_head_replacement(dataset, clean_cache, layer, head)

    clean_ld = logit_diff(clean_logits, dataset.io_token_ids, dataset.subject_token_ids)
    # Corrupt and patched runs are scored against the clean target A vs subject B.
    corrupt_ld = logit_diff(corrupt_logits, dataset.io_token_ids, dataset.subject_token_ids)
    patched_ld = logit_diff(patched_logits, dataset.io_token_ids, dataset.subject_token_ids)
    clean_head = get_attention_head_output(clean_cache, layer, head)
    corrupt_head = get_attention_head_output(corrupt_cache, layer, head)

    result = {
        "seed": seed,
        "layer": layer,
        "head": head,
        "hook_name": hook_name,
        "clean_logit_diff": float(clean_ld["mean"].item()),
        "corrupted_logit_diff": float(corrupt_ld["mean"].item()),
        "patched_logit_diff": float(patched_ld["mean"].item()),
        "clean_tokens_shape": tuple(dataset.clean_tokens.shape),
        "corrupted_tokens_shape": tuple(dataset.corrupted_tokens.shape),
        "clean_head_output_shape": tuple(clean_head.shape),
        "corrupted_head_output_shape": tuple(corrupt_head.shape),
        "budget_snapshot": get_budget(),
    }
    RESULTS_DIRECTORY.mkdir(exist_ok=True)
    path = RESULTS_DIRECTORY / f"ioi_head_patching_diagnostic_seed{seed}_l{layer}_h{head}.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["result_path"] = str(path.relative_to(REPOSITORY_ROOT))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one technical IOI head-patching diagnostic.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--head", type=int, default=0)
    args = parser.parse_args()
    result = run_diagnostic(seed=args.seed, layer=args.layer, head=args.head)

    print("One-head activation-patching diagnostic (technical check only)")
    print(f"  chosen layer/head: {result['layer']}.{result['head']}")
    print(f"  hook name: {result['hook_name']}")
    print(f"  clean LD: {result['clean_logit_diff']:.4f}")
    print(f"  corrupted LD: {result['corrupted_logit_diff']:.4f}")
    print(f"  patched LD: {result['patched_logit_diff']:.4f}")
    print(f"  clean tokens: {result['clean_tokens_shape']}")
    print(f"  corrupted tokens: {result['corrupted_tokens_shape']}")
    print(f"  clean selected-head output [batch, position, d_model]: {result['clean_head_output_shape']}")
    print(f"  corrupted selected-head output [batch, position, d_model]: {result['corrupted_head_output_shape']}")
    print(f"  passes used: {result['budget_snapshot']['passes_used']}")
    print(f"  saved: {result['result_path']}")


if __name__ == "__main__":
    main()
