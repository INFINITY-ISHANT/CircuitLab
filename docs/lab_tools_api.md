# Person A integration guide

Import only the public functions in `lab_tools.py`. The orchestration layer
does not need to import TransformerLens, PyTorch, internal CircuitLab modules,
or hook names.

```python
from lab_tools import (
    ablate,
    budget,
    faithfulness,
    make_dataset,
    patch,
    path_patch,
    run_baseline,
    score_vs_truth,
)
```

All tools currently support the `"ioi"` task and GPT-2 Small attention heads.
A component is always the exact string `"L<layer>H<head>"`, with both layer and
head in `0..11`, for example `"L9H9"`. Components are passed as one string or
a non-empty list of unique strings.

## Shared dataset

Create one dataset and pass that same object to every compared strategy. This
keeps prompts, answer tokens, and the IOI metric identical across calls.

```python
ds = make_dataset("ioi", n=100, seed=42)
```

`make_dataset(task, n=100, seed=0) -> IOIDataset`

| Input | Schema |
| --- | --- |
| `task` | Exact case-insensitive string `"ioi"`. |
| `n` | Positive integer number of examples. |
| `seed` | Integer. The same `n` and seed reproduce the same dataset. |

`IOIDataset` is an opaque object for orchestration purposes. It contains clean
and ABC-corrupted prompts/tokens, IO and subject answer token IDs, seed, and
prompt-length metadata. Treat it as read-only and retain it in the strategy
state. The dataset generator performs tokenization only; it does not run the
model.

## Baseline

```python
baseline = run_baseline("ioi", ds=ds)
print(baseline["mean_logit_diff"], baseline["accuracy"])
```

`run_baseline(task="ioi", ds=None, n=100, seed=42) -> dict`

If `ds` is supplied, `n` and `seed` are ignored and the supplied dataset is
used. The model runs on clean prompts. Its canonical metric is final-position
`logit(IO) - logit(subject)`.

| Output key | Meaning |
| --- | --- |
| `task`, `dataset_id`, `n`, `seed` | Dataset identity and provenance. |
| `accuracy` | Fraction of examples where IO logit is greater than subject logit. |
| `mean_logit_diff`, `std_logit_diff` | Dataset mean and population standard deviation. |
| `per_example_logit_diffs` | JSON-safe list of one score per example. |
| `forward_passes`, `batch_size`, `device` | Execution provenance. |
| `budget_snapshot` | Budget state after this call. |
| `result_path` | Saved scalar baseline JSON under `results/`. |

## Activation patching

```python
head_results = patch(
    components=["L9H9", "L10H0"],
    positions="final",
    ds=ds,
)
ranked = head_results.sort_values("effect", ascending=False)
```

`patch(components, positions="final", ds=..., *, log_path=None) -> pandas.DataFrame`

This runs corrupted prompts and replaces each selected head's clean
`hook_result` activation at the final relevant prompt token. A separate row is
returned for each head. `positions=["final"]` is also accepted. `"end"` is not
an alias; the only supported public position is exactly `"final"`.

| DataFrame column | Meaning |
| --- | --- |
| `component`, `layer`, `head`, `position` | Intervention identity. |
| `clean_logit_diff`, `corrupted_logit_diff`, `patched_logit_diff` | Dataset-mean canonical metric values. |
| `effect` | `patched_logit_diff - corrupted_logit_diff`. |
| `recovery` | `effect / (clean_logit_diff - corrupted_logit_diff)`; `NaN` if the denominator is zero. |
| `passes_used` | Cumulative passes within this tool call, including shared caches. |

## Mean ablation

```python
ablation_results = ablate(["L9H9", "L10H0"], ds=ds, mode="mean")
```

`ablate(components, ds=..., mode="mean", *, log_path=None) -> pandas.DataFrame`

Each row independently evaluates one selected head on clean prompts. Mean
ablation replaces its activation with the corresponding activation mean over
the **batch/examples of the corrupted reference prompts**. It does not average
across token positions, heads, or model dimensions. Only `mode="mean"` exists.

| DataFrame column | Meaning |
| --- | --- |
| `component`, `layer`, `head`, `mode`, `mean_source` | Ablation provenance. |
| `baseline_logit_diff`, `ablated_logit_diff` | Clean full-model and one-head-ablated means. |
| `signed_change` | `ablated_logit_diff - baseline_logit_diff`. |
| `absolute_change` | Absolute value of `signed_change`. |
| `relative_change` | `signed_change / abs(baseline_logit_diff)`; `NaN` for a zero baseline. |
| `passes_used` | Cumulative passes within this call. |

## Path patching

```python
paths = path_patch(
    sender="L5H5",
    receivers=["L9H9", "L10H0"],
    ds=ds,
)
```

`path_patch(sender, receivers, ds=..., *, log_path=None) -> pandas.DataFrame`

The sender must be earlier than every receiver. The tool measures a two-stage
intervention: it takes a sender's clean activation and tests its effect through
the requested receiver output(s), while using corrupted-reference activations
for the rest of the path-patching setup. It returns a single receiver row or an
`ALL` row for a joint receiver set.

| DataFrame column | Meaning |
| --- | --- |
| `sender`, `receiver`, `receiver_components`, `positions` | Path identity. |
| `clean_logit_diff`, `corrupted_logit_diff`, `path_patched_logit_diff` | Dataset-mean metric values. |
| `path_effect` | `path_patched_logit_diff - corrupted_logit_diff`. |
| `recovery` | `path_effect / (clean_logit_diff - corrupted_logit_diff)`; `NaN` if undefined. |
| `passes_used` | Cumulative passes within this call. |

## Circuit faithfulness

```python
faithfulness_result = faithfulness(["L9H9", "L10H0"], ds=ds)
print(faithfulness_result["faithfulness"])
```

`faithfulness(circuit, ds=..., *, log_path=None) -> dict`

This simultaneously patches all listed clean head outputs into the corrupted
run at each final token. It evaluates the proposed head set; it does not prove
that the set is a complete causal circuit.

| Output key | Meaning |
| --- | --- |
| `circuit`, `n` | Canonical sorted components and dataset size. |
| `clean_logit_diff`, `corrupted_logit_diff`, `circuit_patched_logit_diff` | Dataset means. |
| `faithfulness` | `(circuit_patched - corrupted) / (clean - corrupted)`. |
| `forward_passes`, `budget_snapshot` | Cost and ending budget state. |

## Ground-truth recovery score

```python
truth_score = score_vs_truth(["L9H9", "L10H0"])
```

`score_vs_truth(circuit) -> dict`

The score uses set membership of `LxHy` identities. It returns:

```python
{
    "tp": int,
    "fp": int,
    "fn": int,
    "precision": float,
    "recall": float,
    "f1": float,
    "success": bool,  # recall >= 0.80 and precision >= 0.70
    "role_breakdown": dict,
}
```

**Current repository status:** `circuitlab/datasets/ioi_truth.json` is an
unverified placeholder with no supplied head list. This function therefore
raises `ValueError` until a verified repository-owned source is populated. An
agent must report this as unavailable; it must not substitute an external or
guessed truth list.

## Budget

```python
budget(reset=True, max_passes=5000)  # start a new capped experiment
status = budget()
print(status)
```

`budget(*, max_passes=UNSET, reset=False) -> dict`

| Output key | Meaning |
| --- | --- |
| `passes_used` | Total model batch-forward calls since the latest reset. |
| `passes_left` | Remaining permitted calls, or `None` for no cap. |
| `max_passes` | Configured cap, or `None`. |
| `by_operation` | Count grouped by scientific operation. |

One forward pass means one `HookedTransformer` call on one prompt batch. A
batch of one and a batch of one hundred each count as one pass. Every public
model experiment reserves a pass immediately before execution. A capped run
raises before starting a call that would exceed the cap.

Call `budget(reset=True, max_passes=N)` to set a cap and reset counts. Call
`budget(reset=True)` to clear counts while retaining the existing cap. Supplying
`max_passes` without `reset=True` is rejected so a strategy cannot silently
change the accounting mid-run.

## Logging and orchestration rules

`patch`, `ablate`, `path_patch`, and `faithfulness` append concise JSONL run
records to `results/runs.jsonl` by default. Pass a `log_path` to direct a test
or isolated run elsewhere. Records contain a run ID, UTC timestamp, dataset
identity, components, scalar summary, parameters, and passes consumed. They do
not contain activations or environment secrets.

Use DataFrame scalar columns and dictionary scalar fields when handing results
back to planners. Do not pass tensors, model objects, hooks, or arbitrary
callables across the public boundary.

## Validation and exception contract

The public API intentionally fails fast. Handle these errors as experiment
configuration failures, record the message, and do not retry with guessed
arguments.

| Exception | Typical cause |
| --- | --- |
| `TypeError` | Invalid type, missing required `ds`, malformed dataset object, or non-string component. |
| `ValueError` | Invalid layer/head range, duplicate component, invalid dataset shape, invalid budget, or impossible path direction. |
| `NotImplementedError` | Unsupported task, position other than `"final"`, ablation mode other than `"mean"`, or unavailable literature capability. |
| `RuntimeError` | Budget exhausted or an internal model/cache invariant failed. |

An agent should reuse one dataset, set its budget before a strategy starts,
inspect `passes_used`, and preserve raw DataFrames plus run-log references for
later benchmark aggregation.
