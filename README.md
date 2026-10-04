# CircuitLab

CircuitLab is a team of AI agents that hunts for the circuit behind a behavior in GPT-2 small, treating every forward pass as an expensive experiment and spending as few as it can. It is built on Omnigent. The agents propose hypotheses, run patching and ablation experiments through one shared tool server, check their own circuit against a random control, and must get a human to approve any final claim.

The test case is indirect object identification (IOI), where the published circuit of 26 heads (arXiv:2211.00593) gives us an answer key to score against. The researchers who benefit are those who want to map a model behavior without sweeping every component by hand.

## Results

- Exhaustive baseline: activation patching of all 144 heads takes 292 forward passes; its top 10 hold 8 of the 26 published IOI heads.
- Best agent run (run 1): L9H9, L10H0, L9H6, faithfulness 0.8827 vs 0.0275 for a random circuit, recall 3/26, 366 passes.
- Upstream: no upstream head was confirmed by path patching yet; run 3 proposed 4 real upstream heads among 14 candidates but ended before testing them.

The agents have not beaten the exhaustive baseline on forward passes. Details and limits: [docs/RESULTS.md](docs/RESULTS.md). One-page report: [docs/CircuitLab_OnePager.pdf](docs/CircuitLab_OnePager.pdf).

## How it works

- `circuitlab/config.yaml` is the Planner. `circuitlab/agents/` holds lit_scout, task_designer, hypothesis, experimentalist, analyst and safety.
- `lab_server.py` serves the tools over MCP (`agent_tools.py`, a JSON layer over `lab_tools.py`), so every agent shares one forward-pass counter.
- `lab_tools.py` and `circuitlab/` are the TransformerLens toolkit: IOI dataset, logit-difference metric, activation patching, mean ablation, path patching, faithfulness with random controls, and scoring against the published circuit.
- Guardrails: a human approves every final claim (`claim_gate.py`), a human approves any patch or ablate over 24 heads and any odd path-patch sweep, and Omnigent's cost_budget policy caps spend.
- `demo_view.py` is a live Streamlit view of the research record (`research_record.jsonl`, plus `research_record_run1.jsonl` for run 1).

## Hosted demo

`demo_view.py` needs no GPU and no API key: it replays the recorded agent runs in `research_record_run1.jsonl` (run 1) and `research_record.jsonl` (run 3). To host it, deploy this repository on Streamlit Community Cloud with `demo_view.py` as the main file. Live agent runs need the local setup below.

## Setup

Requirements: Windows with Git for Windows, Python 3.12, an NVIDIA GPU with CUDA (the pinned torch wheel is `cu130`; on a CPU-only machine edit `requirements-lab.txt`), Omnigent 0.16.0 and an Anthropic API key scoped to a workspace (agents only; the toolkit and tests need no key).

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lab.txt
$env:HF_HOME = Join-Path (Get-Location) '.model-cache'
.venv\Scripts\python.exe smoke_model.py
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Dependencies are pinned in `requirements-lab.txt` (the root `requirements.txt` holds only `streamlit`, for the hosted demo): transformer-lens 3.9.0, torch 2.14.1 (cu130), transformers 5.18.0, numpy, pandas, matplotlib, `mcp<2`, streamlit. Omnigent is installed separately with `pip install uv` then `uv tool install --python 3.12 omnigent`, and its key is added with `omnigent setup`. Never commit keys.

## Running the agents

1. With the Python that has the requirements, run `python set_paths.py`. It points every agent spec at this folder's `lab_server.py` and at that Python (add `--http` to use the shared server URL).
2. In a new terminal run `env.bat` (Git `sh` on PATH, UTF-8, this folder on PYTHONPATH so `claim_gate.py` can be found). It also sets `CIRCUITLAB_BLOCK_ARXIV` to hide the IOI and ACDC papers from the literature agent for benchmark runs; remove that line for a discovery run. Then run `omnigent stop` and `omnigent start`.
3. Start the shared tool server with `python lab_server.py --http --max-passes 5000`, then `omnigent run circuitlab`.
4. In another terminal, `streamlit run demo_view.py` for the live view of the research record.

Heads are written `L9H9` (layer 9, head 9). `9.9` is accepted too and converted.

## Reproducing the baselines

```powershell
.venv\Scripts\python.exe experiments\validate_toolkit.py
.venv\Scripts\python.exe experiments\exhaustive.py
.venv\Scripts\python.exe experiments\run_report.py --log results\run3_calls.jsonl --since 2026-10-04T10:41
```

`run_report.py` totals the forward passes in a run log after the given UTC time and scores the circuit the Planner's closing rule would give. `results/run3_calls.jsonl` is the slice of the tool-call log that belongs to run 3 and reproduces `results/run3_report.json` (add `--check` for the faithfulness check, which needs the GPU). New runs append to `results/runs.jsonl`, which is not tracked. The exhaustive result for n=100, seed 42 is already in `results/`.

## Data

There is no external dataset. IOI prompts are generated by `circuitlab/datasets/ioi.py` from 12 templates (clean and name-swapped corrupted pairs, seed 42, n=100). The published circuit is in `circuitlab/datasets/ioi_truth.json`, and the measured results are in `results/`.

## Repo layout

- `circuitlab/` agent specs, TransformerLens toolkit, datasets, metrics
- `experiments/` baseline, benchmark, validation and report scripts
- `tests/` unit tests (run without an API key)
- `results/` cited result files
- `docs/` results write-up (`RESULTS.md`), one-page report and submission text

## Technical notes

### Model loading

`model_loader.get_model()` lazily loads GPT-2 Small through
`HookedTransformer.from_pretrained("gpt2", device=...)` and reuses one instance
per Python process. Device selection is centralized: CUDA when PyTorch reports
it available, otherwise CPU. The returned model is in eval mode. Inference
callers should use `torch.inference_mode()` and should not move the shared model.

TransformerLens is pinned to 3.9.0 because 4.0 removed `HookedTransformer`.
The setup below targets Python 3.12 and installs the CUDA 13.0 PyTorch wheel
used by the RTX 5050. The first smoke run downloads model weights.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lab.txt
$env:HF_HOME = Join-Path (Get-Location) '.model-cache'
.venv\Scripts\python.exe smoke_model.py
.venv\Scripts\python.exe experiments\validate_toolkit.py
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The smoke script prints the chosen device, checks `[batch, sequence, vocab]`
logits dimensions, and prints five next-token predictions for the IOI example.
It performs one real forward call on one prompt; this is model-loading validation,
not an IOI benchmark. Lifecycle unit tests use mocks and do not validate model
predictions.

### Forward-pass accounting

`circuitlab.budget.BudgetTracker` is the process-wide accounting mechanism for
scientific experiments. One pass means one call to `HookedTransformer` on a
batch, irrespective of the number of prompts in that batch. This matches the
project's planned per-batch patching comparison. Every result also records its
batch size, dataset identity, and exact passes used, so strategies must use the
same dataset and batching policy for comparisons to be meaningful.

`budget()` returns the current JSON-compatible snapshot. A maximum is optional:
`get_tracker().reset(max_passes=5000)` configures one, while `reset()` clears
usage and retains the existing maximum. Experiments reserve a pass immediately
before calling the model, which prevents an over-budget invocation.

### Activation patching convention

Public activation patching accepts stable attention-head names such as `L9H9`.
For `positions="final"`, it copies that head's clean `hook_result` activation
into the corrupt run only at each prompt's final relevant token. Results use the
canonical logit difference. `effect = patched_ld - corrupted_ld`; `recovery =
effect / (clean_ld - corrupted_ld)`, both computed from dataset-mean logit
differences. Recovery is `NaN` if the denominator is zero. Cache setup runs are
shared across requested heads, and each row's `passes_used` is cumulative within
the tool call.
