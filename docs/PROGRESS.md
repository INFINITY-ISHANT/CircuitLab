# CircuitLab progress

Shared status file for both teammates and both coding agents. Read this first, update it at the end of every work session, commit and push it.

Plan doc: https://claude.ai/code/artifact/01699a2a-aff6-4881-a331-3ee4a50b99c5
Challenge brief: Hack-Nation x Databricks, "Agentic Scientific Discovery" (Omnigent is mandatory).

## How to update this file

- Edit "Current status" and "Next up" in place. Keep them short.
- Add one line to "Log" for every meaningful thing done. Newest on top. Format: `YYYY-MM-DD HH:MM | A or B | what changed | where`.
- Anything the other person must know goes under "Decisions" or "Open questions", never only in chat.
- If you change a function signature in `lab_tools.py`, say so under "Interface changes" and tell the other person.

## Goal in one paragraph

Agents (built on Omnigent) reverse-engineer which attention heads and MLPs in GPT-2 small implement a behavior. Every forward pass is the "expensive experiment". Validation: rediscover the known IOI circuit (arXiv 2211.00593) and compare forward passes needed for 80% recall against exhaustive patching and ACDC (arXiv 2304.14997). Discovery run: apply the same lab to a behavior with no published circuit and report a labelled hypothesis.

## Roles

- Person A (Ishant): Omnigent setup, the 7 agents, handoffs, research record, human gates, demo view.
- Person B (teammate): TransformerLens toolkit behind `lab_tools.py`, IOI dataset, baselines, metrics, results.
- Contract between them: `lab_tools.py` (signatures below).

## Decisions so far

- 2026-10-03: project is circuit discovery on GPT-2 small (earlier ideas dropped).
- 2026-10-04: both laptops have an NVIDIA 8 GB GPU, so all experiments run on GPU (`device="cuda"`). Ishant's is an RTX 4060 Laptop, CUDA confirmed.
- 2026-10-04: agents use hosted Claude models through Omnigent, not a local LLM, to keep VRAM free for GPT-2.
- 2026-10-04: Omnigent open source (not managed Databricks), running natively on Windows.
- 2026-10-04: agents reach `lab_tools` through an MCP server (`lab_server.py`) that runs in the conda Python, because Omnigent's `type: function` tools were silently dropped and Omnigent's own Python env has no torch. Person B only edits `lab_tools.py`; `lab_server.py` wraps its functions as they are.
- 2026-10-04: tools return plain dicts and lists (JSON), not DataFrames. All tools take a `dataset_id` string, not an object. Components are heads only for now.
- 2026-10-04: one forward pass means one `HookedTransformer` call on one batch,
  regardless of batch size. Each result records batch size and pass count; all
  compared strategies must use the same dataset and batching policy.
- 2026-10-04: attention-head output patching uses TransformerLens
  `blocks.<layer>.attn.hook_result`, with `model.set_use_attn_result(True)`
  enabled before caching. That activation has shape `[batch, position, head,
  d_model]` and is patched one head slice at a time.
- 2026-10-04: public `patch` is the exception to the JSON-only tool convention:
  it returns a pandas DataFrame so agents can rank component results directly.
- 2026-10-04: heads are written `L9H9` (layer 9, head 9). `9.9` and `l9h9` are accepted and converted to `L9H9`. MLP components are not supported yet.
- 2026-10-04: the two folders (SapientLabs and PersonB) were merged into one checkout, `CircuitLab/`. Person B's toolkit is the base; the agent specs, claim gate, demo view, literature search and `present_claim` from Person A were added on top. The tool-boundary gap (objects and DataFrames versus JSON) is open, see Next up.

## Environment setup (Windows)

1. `pip install uv`, then `uv tool install --python 3.12 omnigent`.
2. Git for Windows must be installed. Omnigent runs a key helper through `sh`, so put `C:\Program Files\Git\usr\bin` on PATH.
3. `set PYTHONUTF8=1` or the Omnigent host crashes with a `charmap` encoding error.
4. Run `env.bat` in every new terminal (sets 2 and 3), then `omnigent stop` and `omnigent start`.
5. Model credential: `omnigent setup`, add an Anthropic API key. The key must be scoped to a workspace in the Anthropic console (org-scoped keys fail). Never commit keys.
6. Agent layout: a folder with `config.yaml` for the planner and `agents/<name>/config.yaml` per sub-agent. Handoffs use `sys_session_send` and `sys_read_inbox`. Run with `omnigent run <folder>`.
7. MCP server deps: `pip install "mcp<2"` (in `requirements.txt`). Version 2 renamed `FastMCP` and breaks `lab_server.py`. After cloning, run `python set_paths.py` with the Python that has torch and transformer-lens so the agent specs point at it.
8. Custom policy modules (claim_gate.py) are imported by the Omnigent server, so the SapientLabs folder must be on PYTHONPATH (env.bat does it). Policies go under `guardrails: policies:` in the agent yaml.
9. Torch with CUDA for Person B: check with `python -c "import torch; print(torch.cuda.is_available())"`.

## Repo layout

```
lab_tools.py          the public toolkit API that agents use (contract), plus search_literature and present_claim
agent_tools.py        agent-facing wrapper: dataset ids in, plain JSON out (tests/test_agent_tools.py)
lab_server.py         MCP server exposing agent_tools and research_record; `--http` for one shared server
research_record.py    log_event / read_record, writes research_record.jsonl
claim_gate.py         Omnigent policy: human approval before present_claim
demo_view.py          Streamlit live view of the research record
set_paths.py          points the agent specs at this checkout and at the right Python
env.bat               PATH, UTF-8 and PYTHONPATH fix, run in every new terminal
circuitlab/           the Python package (model, datasets, budget, evaluation, interpretability)
circuitlab/config.yaml and circuitlab/agents/<name>/config.yaml   the Omnigent lab (Planner and 6 agents)
experiments/          exhaustive patching, benchmark, plots, validation runners
tests/                unit tests
docs/                 integration guide for the agent side
hello/                hello-world handoff, kept as a smoke test
results/              generated outputs (ignored by git)
```

Rules for `lab_tools.py` functions: type hints on every argument, a docstring that says what the tool does and when to use it (agents read it), JSON-serializable return values.

## Current status

Person A:
- [x] Omnigent installed and running
- [x] Hello-world handoff (planner, scout, critic) works on API credits
- [x] Stub tools used for rehearsal, now replaced by Person B's real toolkit in the merged `lab_tools.py`
- [x] Research record writer (JSON lines)
- [x] Planner wired to stub tools through the MCP server (budget, patch, budget verified)
- [x] First version of all 7 agent specs (Planner plus lit_scout, task_designer, hypothesis, experimentalist, analyst, safety) with per-agent tool allowlists
- [x] First full loop run on stub tools (handoffs and record work; analyst caught that stub scores do not change with the circuit)
- [x] Claim gate enforced by an Omnigent policy: `present_claim` tool + `claim_gate.py`, human gets an approval prompt (verified 2026-10-04)
- [x] Agent adapter over the real toolkit (`agent_tools.py`: dataset ids, JSON records, faithfulness bundled with the random control), unit tested without a GPU
- [ ] One shared tool server for all agents (`python lab_server.py --http --max-passes 5000` exists, agent specs still use one stdio server each)
- [ ] Budget gate (>25% of remaining forward-pass budget needs a human) as a policy, same pattern as the claim gate
- [x] Cost levers: `cost_budget` policy in Planner ($1.50 cap, ask at $1.00, parses but enforcement untested), worker agents pinned to `model: haiku`, Planner and analyst on Sonnet, 2 rounds while developing (set back to 4 for the demo run), 80 word reply rule
- [x] Literature scout on real arXiv search (stdlib, `search_literature`; keyword AND query, short words dropped)
- [x] Demo view: `streamlit run demo_view.py`, live Streamlit page over research_record.jsonl (archive the old record before a demo run)

Person B:
- [x] GPT-2 small loads on GPU in TransformerLens
- [x] Clean IOI baseline: seed 42, n=100, accuracy 0.96, mean logit difference 2.7633
- [x] IOI dataset generator (clean/ABC-corrupted prompts and GPT-2 tokens)
- [x] Canonical batched IOI logit-difference metric with synthetic tests
- [x] Shared forward-pass tracker integrated with the clean baseline
- [x] Technical one-head activation-patching diagnostic with cached clean/corrupt runs
- [x] Public final-token activation patching for one or more attention heads
- [x] Exhaustive final-token attention-head patching baseline with resume and CSV output
- [ ] Verified `ioi_truth.json`
- [x] Public `patch` sweep with forward-pass counter
- [x] Mean `ablate` with forward-pass counter
- [x] Two-stage final-token path patching with forward-pass counter
- [x] Set-based `score_vs_truth` implementation (blocked from real use until truth is verified)
- [x] Final-token circuit faithfulness and same-size random-circuit control
- [x] Source-gated backup-name-mover experiment (awaiting verified head roles)
- [x] Single small-dataset toolkit validation runner (truth checks fail until source is verified)
- [x] Resumable five-seed benchmark runner with explicit unavailable strategies
- [x] Cost-indexed circuit-recovery trajectory evaluator
- [x] Honest per-seed target-cost speedup evaluator
- [x] Source-backed recall-vs-forward-passes benchmark visualization
- [ ] Baselines: exhaustive sweep and ACDC

## Next up

1. Agent adapter: done (`agent_tools.py`). Still to verify with a real agent run on the GPU. (A)
2. One shared tool server over HTTP instead of one stdio process per agent, so the pass counter is shared and GPT-2 loads once: switch the agent specs to the server's URL. (A)
3. Populate `circuitlab/datasets/ioi_truth.json` from the IOI paper's table, with a source citation and `status: "verified"`. Unblocks `score_vs_truth` and every benchmark number. (A and B)
4. Full 144-head exhaustive baseline on the shared dataset (only layers 0 to 1 at n=8 so far). (B)
5. ACDC baseline in a separately pinned environment with a JSON bridge. (B)
6. Budget gate as an Omnigent policy, then switch rounds back to 4 and do a full real run. (A)
7. Export the agents' proposals with cumulative pass cost so `recovery_trajectory` can score the CircuitLab run. (A and B)
8. Discovery run on a behavior with no published circuit. (A and B)

## Interface: `lab_tools.py`

```
make_dataset(task, n=100, seed=0) -> dict        # includes dataset_id
run_baseline(task) -> dict                        # accuracy, mean_logit_diff
patch(components, dataset_id) -> list[dict]       # recovered fraction per component
ablate(components, ds, mode="mean") -> DataFrame  # clean vs mean-ablated LD per head
path_patch(sender, receivers, ds) -> DataFrame     # two-stage sender-to-receiver effect
faithfulness(circuit, ds) -> dict                  # final-token clean-to-corrupt recovery
random_circuit_control(circuit, ds, n_random=20, seed=42) -> dict
score_vs_truth(circuit) -> dict                   # precision, recall vs ioi_truth.json
budget(max_passes=..., reset=False) -> dict       # read/reset validated pass budget
search_literature(query) -> list[dict]            # title, arxiv_id, snippet
```

Interface changes since the plan: JSON return types and `dataset_id` strings (see Decisions).

- 2026-10-04: `make_dataset("ioi", n, seed)` now returns `IOIDataset` so the
  clean/corrupted token tensors and patching metadata remain available. Dataset
  registration by `dataset_id` will be added with the intervention tools.
- 2026-10-04: `run_baseline(task="ioi", ds=None, n=100, seed=42)` accepts an
  optional `IOIDataset`, otherwise creates a seeded dataset. It returns scalar
  metrics, per-example logit differences, and the exact forward-pass count.
- 2026-10-04: `budget()` returns the shared `BudgetTracker` snapshot with
  `passes_used`, `passes_left`, `max_passes`, and operation totals. A maximum
  is unconfigured until the team locks the experiment budget.
- 2026-10-04: `patch(components, positions="final", ds=IOIDataset)` accepts
  stable names like `L9H9` and returns a DataFrame. Recovery is
  `(patched_ld - corrupted_ld) / (clean_ld - corrupted_ld)` and effect is
  `patched_ld - corrupted_ld`, both computed on dataset means. `passes_used`
  is cumulative within that patch call, including shared cache setup.
- 2026-10-04: exhaustive patching writes raw evaluation order and a separate
  effect-ranked CSV after every head. `cumulative_forward_passes` persists the
  shared cache forwards plus each patched forward, including work across a
  resumed process.
- 2026-10-04: `ablate(components, ds=IOIDataset, mode="mean")` returns a
  DataFrame with clean baseline LD, ablated LD, signed/absolute/relative
  changes, mean source, and cumulative pass count. Only mean ablation is
  implemented; zero ablation raises `NotImplementedError`.
- 2026-10-04: mean ablation runs clean prompts and replaces one selected
  attention-head output at every position with a mean over the *batch dimension
  only* of paired ABC-corrupted reference prompts. Length-homogeneous groups
  retain positions and avoid padded-token means; each component is evaluated
  independently.
- 2026-10-04: `score_vs_truth` treats predicted and published attention heads
  as sets, so duplicate predictions cannot improve metrics. It requires a
  non-empty truth file with `status: "verified"`; the current placeholder
  intentionally raises instead of returning fictitious scores.
- 2026-10-04: circuit faithfulness is final-token clean-to-corrupted recovery:
  `(circuit_patched_ld - corrupted_ld) / (clean_ld - corrupted_ld)`. All heads
  in a candidate are patched simultaneously, and random controls use that exact
  dataset, metric, and intervention. `beats_random` means strictly greater than
  the sampled random mean; the empirical percentile is also returned.
- 2026-10-04: the backup-name-mover experiment reads primary and backup roles
  only from verified IOI truth entries. It jointly mean-ablates primary movers,
  then primary-plus-backup sets, and records the clean and ablated LDs. It does
  not execute while source roles are absent.
- 2026-10-04: path patching uses a two-stage sender-to-receiver intervention:
  patch a clean sender head output into corrupt prompts and cache later
  receiver outputs, then patch only those bridge receiver outputs into fresh
  corrupt prompts. Sender/receiver results are patched at all true prompt
  positions; same-layer or earlier receivers are rejected.
- 2026-10-04: `lab_tools.py` validates all public head IDs against GPT-2 Small
  bounds, checks IOI dataset shape/metadata, restricts patch positions and
  ablation modes, validates seeded sampling arguments, and keeps hook details
  internal. `budget(max_passes=..., reset=True)` is the only public budget
  mutation path and rejects invalid caps.
- 2026-10-04: public experiment calls append bounded, JSON-valid run records
  to `results/runs.jsonl`. Records include dataset and seed provenance,
  components, intervention positions, metrics, concise scalar summaries, pass
  cost, and parameters; tensors and environment data are rejected.
- 2026-10-04: benchmark comparisons use seeds 0-4 and one common IOI dataset
  per strategy/seed. A record's `forward_passes` is documented discovery cost
  plus the same-dataset faithfulness evaluation cost; both terms are retained.
  Missing exhaustive CSVs, CircuitLab traces, ACDC traces, or verified truth
  create resumable `unavailable` rows instead of fabricated metrics.
- 2026-10-04: recovery trajectories score each cumulative-cost proposal against
  verified IOI truth and persist JSON plus plot-ready CSV. `passes_to_target`
  is the first point meeting recall >= 0.80 and precision >= 0.70, or `None`
  when no recorded proposal reaches it.
- 2026-10-04: headline speedup is `baseline_passes_to_target /
  circuitlab_passes_to_target` only for seeds where both strategies reach the
  target at positive cost. All missing/unreached/zero-cost cases retain an
  explicit status and `None` speedup rather than an infinity or fabricated zero.
- 2026-10-04: the recall-cost plot reads only completed `benchmark_raw.csv`
  rows and seed-labelled recovery trajectories. It draws mean and ±1 standard
  deviation only where multiple observed seed points share a cost, omits
  unavailable strategies, and exports the exact plotted rows as CSV.

## Open questions

- Which Python runs `lab_server.py`: it needs torch, transformer-lens and `mcp<2` in one environment. Person B uses a `.venv`; Person A has been using conda base. Pick one.
- `requirements.txt` pins the CUDA 13.0 PyTorch wheel (written for an RTX 5050). Check it installs on the RTX 4060 laptop, or use the cu12x wheel there.
- Budget total for forward passes (stub uses 5000). Lock it at the next sync.
- Success thresholds: 80% recall, 70% precision, 0.7 faithfulness (from the plan, confirm).
- IOI ground-truth source file is required before populating `ioi_truth.json`:
  the repository currently contains only an empty `unverified_placeholder`
  citing arXiv:2211.00593, not the published category-to-head assignments.

## Local artifact replication

Large files are deliberately not versioned: `.venv/`, UV caches, `.model-cache/`
(including downloaded GPT-2 weights), and generated `results/*` artifacts are
ignored. Person A can recreate the local environment and model cache from a
fresh clone with:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:HF_HOME = Join-Path (Get-Location) '.model-cache'
.venv\Scripts\python.exe smoke_model.py
```

The smoke run downloads GPT-2 on first use. Regenerate small toolkit outputs
with `.venv\Scripts\python.exe experiments\validate_toolkit.py`; regenerate the
benchmark plot only after source-backed benchmark trajectories exist with
`.venv\Scripts\python.exe experiments\plot_recall_vs_forward_passes.py`.

## Log

2026-10-04 16:52 | A+B | Cleaned the repo for submission: only code, tests, cited results and docs; raw run log stays untracked | `.gitignore`
2026-10-04 16:50 | A+B | Removed score_vs_truth from the analyst's tools and prompt so future runs get no answer-key feedback | `circuitlab/agents/analyst/config.yaml`
2026-10-04 16:48 | A+B | Added `experiments/run_report.py` (totals passes from results/runs.jsonl, applies the Planner's closing rule, scores the circuit); wrote docs/RESULTS.md and docs/DEMO_SCRIPT.md | `experiments/run_report.py`, `docs/`
2026-10-04 16:46 | A+B | Run 3 (IOI and ACDC papers hidden, upstream trace required): 786 logged passes, no claim presented, no agent path-patch calls logged, last tested 31-head candidate faithfulness 0.5884 vs random mean 0.3276 | `results/run3_report.json`, `research_record.jsonl`
2026-10-04 16:44 | A+B | Run 2 aborted by hand: open-ended upstream sweep (60 senders x 10 receivers) cost about 1700 passes; led to joint-only path patching and the 24-sender limit | `agent_tools.py`
2026-10-04 | A | Added agent_tools.py (agent layer over lab_tools) and an --http mode with --max-passes for lab_server.py; 6 new tests pass without a GPU; Person B's 68 tests pass on Person A's laptop | agent_tools.py, lab_server.py, tests/test_agent_tools.py
2026-10-04 | A | Merged SapientLabs and PersonB into CircuitLab/: added present_claim, real arXiv search_literature, L9H9/9.9 normalization, portable env.bat, set_paths.py, requirements (mcp, streamlit), .gitignore; updated two tests that assumed 9.9 was invalid and search_literature was a stub | CircuitLab/
2026-10-04 | A | Real arXiv literature search, Streamlit demo view, and `_norm` so tools accept both `L9H9` and `9.9` head names (agents write L9H9; Person B's real tools must accept both too) | lab_tools.py, demo_view.py
2026-10-04 | A | Claim gate works: agent specs take policies under `guardrails: policies:` (top-level `policies:` is ignored); custom handler `claim_gate.require_human_for_claim` found via PYTHONPATH in env.bat; present_claim added to lab_tools and lab_server | circuitlab/config.yaml, claim_gate.py
2026-10-04 | A | Stubs now react to input: score_vs_truth, faithfulness, ablate use a small IOI truth set (STUB_TRUTH) so the analyst sees real differences. Round-1 run cost under $1 | lab_tools.py
2026-10-04 | A | Added cost levers (cost_budget policy, haiku on 5 workers, 2 round cap, short replies). Round-1 run on stubs finished using 10 of 500 passes; cost number still to record | circuitlab/
2026-10-04 | A | Built circuitlab/ with Planner and 6 sub-agents (tool allowlists per agent), about to run first full loop | circuitlab/
2026-10-04 | A | Added research_record.py (log_event, read_record, JSON lines) | research_record.py
2026-10-04 | A | Planner called budget, patch, budget through MCP server and reported 3 passes used | lab_server.py, hello/
2026-10-04 | A | Added lab_tools.py stubs (9 tools) and lab_server.py MCP wrapper; function tools dropped by Omnigent so MCP is the route | lab_tools.py, lab_server.py
2026-10-04 15:45 | B | Verified `.venv/`, `.uv-cache/`, and `.cuda-uv-cache/` are absent from the current branch and history, then added explicit root ignore rules for all three so local Python environments and UV download caches cannot be added to future commits | `.gitignore`
2026-10-04 15:32 | B | Excluded large local/model/result artifacts from Git and documented fresh-clone environment, GPT-2 download, toolkit-validation, and plot-regeneration commands for Person A. Removed the already tracked 523 MiB GPT-2 cache from the Git index while retaining local files; the tracked repository now contains code and reproducibility instructions, not weights, environments, caches, or generated outputs | `.gitignore`, `PROGRESS.md`
2026-10-04 15:25 | B | Tightened `.gitignore` for local Python/model caches, secrets, editor files, test artifacts, runtime logs, serialized artifacts, and generated experiment results while retaining `results/.gitkeep` so the output directory exists in fresh clones | `.gitignore`
2026-10-04 15:20 | B | Added Person A public-API integration guide with copyable IOI dataset, baseline, patch, mean-ablation, path-patching, faithfulness, truth-score, and budget examples; documented schemas, exceptions, logging, budget semantics, and the unverified-truth limitation. Boundary, budget, and logging tests passed (13) | `docs/PERSON_A_INTEGRATION.md`
2026-10-04 14:55 | B | Added clean PNG/PDF recall-vs-forward-passes plotting with 80% target, observed-seed variability, and plotted-data CSV. Plot tests passed; current artifacts contain no curves because all benchmark strategies remain unavailable | `experiments/plot_recall_vs_forward_passes.py`, `tests/test_recall_plot.py`, `results/recall_vs_forward_passes.{png,pdf}`, `results/recall_vs_forward_passes_data.csv`
2026-10-04 14:45 | B | Added per-seed headline speedup summaries with finite mean/std only across seeds where both baseline and CircuitLab reach the target; synthetic success, failure, missing-record, and zero-cost tests passed | `circuitlab/evaluation/speedup.py`, `lab_tools.py`, `tests/test_speedup.py`
2026-10-04 14:40 | B | Added source-gated, plot-ready circuit-recovery trajectories with per-step forward passes, circuit size, precision, recall, F1, and success; synthetic target/non-target and trace-validation tests passed | `circuitlab/evaluation/recovery.py`, `lab_tools.py`, `tests/test_recovery_trajectory.py`
2026-10-04 14:32 | B | Added resumable five-seed benchmark runner for source-backed exhaustive, CircuitLab trace, and future ACDC trace circuits. It writes raw and aggregate CSVs, replaces unavailable rows on resume, and records all missing strategies explicitly; three non-model runner tests passed | `experiments/benchmark.py`, `tests/test_benchmark.py`, `results/benchmark_{raw,aggregate}.csv`
2026-10-04 14:23 | B | Added append-only reproducible JSONL run logging for public patch, ablation, path-patching, faithfulness, and random-control calls; unit tests passed. Toolkit smoke run emitted five schema-valid records without tensors | `circuitlab/run_logging.py`, `lab_tools.py`, `tests/test_run_logging.py`, `results/runs.jsonl`
2026-10-04 14:14 | B | Hardened the autonomous-agent `lab_tools.py` boundary with stable component, dataset, position, mode, seed, and budget validation; added API-level tests. Boundary tests (11) and public patch/ablation/path regressions (8) passed | `lab_tools.py`, `tests/test_lab_tools_api.py`, `tests/test_skeleton.py`
2026-10-04 14:04 | B | Added end-to-end toolkit validation script and README command; real n=1 smoke run passed 10/12 subsystems, with only truth loading/scoring failing honestly because the source file is unverified. It exercised baseline, patching, mean ablation, path patching, faithfulness, random controls, and budget tracking | `experiments/validate_toolkit.py`, `tests/test_validate_toolkit.py`, `README.md`
2026-10-04 13:55 | B | Inspected TransformerLens 3.9 patching APIs and implemented two-stage all-position attention-head path patching; the installed generic patch utility is single-activation, so bridge caching uses the documented hook context. Tensor and n=1 budget tests passed (4 forwards); source-gated documented validation correctly stopped on the unverified truth file | `circuitlab/interpretability/path_patching.py`, `lab_tools.py`, `experiments/validate_path_patching.py`, `tests/test_path_patching.py`
2026-10-04 13:43 | B | Added source-gated backup-name-mover experiment with JSON/CSV raw measurements and joint mean-ablation support; role tests passed. The CLI correctly stopped before model execution because the truth file remains unverified and empty | `experiments/backup_name_movers.py`, `circuitlab/interpretability/ablation.py`, `tests/test_backup_name_movers.py`
2026-10-04 13:36 | B | Implemented simultaneous circuit faithfulness and reproducible same-size random controls with incremental JSON persistence; sampling/hook tests passed. Technical n=2 candidate L0H0/L0H1 with two controls used 18 tracked forwards and saved raw results; no circuit claim made | `circuitlab/evaluation/circuit.py`, `lab_tools.py`, `tests/test_random_controls.py`, `results/random_control_ioi-42-2_seed42_k2.json`
2026-10-04 13:28 | B | Added validated IOI truth loader and set-based TP/FP/FN precision, recall, F1, threshold, and role-recall scorer; synthetic exact-score tests passed. The current unverified empty source intentionally cannot be scored | `circuitlab/datasets/truth.py`, `circuitlab/evaluation/circuit.py`, `lab_tools.py`, `tests/test_truth_scoring.py`
2026-10-04 13:24 | B | Audited repository-provided IOI sources for published category/head assignments; none were present beyond the empty unverified placeholder, so ground truth was intentionally not populated | `circuitlab/datasets/ioi_truth.json`, `PROGRESS.md`
2026-10-04 13:21 | B | Implemented corrupted-reference mean ablation for one or more attention heads with explicit batch-only means and shape checks; tests passed. n=4 seed=42 over L0H0/L0H1/L1H0 used 10 forwards (2 clean baseline + 2 reference cache + 6 ablations); no scientific conclusion drawn | `lab_tools.py`, `circuitlab/interpretability/ablation.py`, `tests/test_mean_ablation.py`
2026-10-04 13:15 | B | Added resumable exhaustive IOI head patching with raw/ranked incremental CSVs and CLI; validation n=8 over layers 0-1 completed 24 heads using 52 tracked forwards (2 clean cache + 2 corrupted cache + 48 patches), and resume used zero forwards | `experiments/exhaustive.py`, `results/exhaustive_ioi_seed42_n8_{raw,ranked}.csv`, `tests/test_exhaustive.py`
2026-10-04 13:07 | B | Implemented public final-token attention-head patching with clean-cache reuse and DataFrame output; n=4 sample over L0H0/L0H1/L1H0 used 10 passes and showed only near-zero effects; 23 tests passed. Public-patch implementation/testing used 20 calls total, including one failed pre-score attempt | `lab_tools.py`, `circuitlab/interpretability/patching.py`, `tests/test_public_patching.py`
2026-10-04 13:00 | B | Validated one-head TransformerLens patching at head 0.0: clean LD 2.8541, corrupt LD -4.6214, patched LD -4.5458; final saved diagnostic used 3 tracked forwards, while setup/debugging used 7 total calls | `circuitlab/interpretability/patching.py`, `experiments/diagnose_head_patching.py`, `results/ioi_head_patching_diagnostic_seed42_l0_h0.json`
2026-10-04 12:51 | B | Implemented central batch-forward BudgetTracker and integrated it with clean baseline; tracker recorded 4/4 baseline forwards, 20 tests passed | `circuitlab/budget.py`, `circuitlab/baseline.py`, `tests/test_budget.py`
2026-10-04 12:46 | B | Implemented clean IOI baseline and validation script; n=100 seed=42 measured accuracy 0.96, mean logit difference 2.7633, std 1.4496, 4 forward passes on CUDA | `circuitlab/baseline.py`, `experiments/validate_ioi.py`, `results/ioi_baseline_seed42_n100.json`
2026-10-04 12:38 | B | Added canonical final-position IOI logit-difference metric with per-example and mean outputs; three hand-calculated synthetic tests passed | `circuitlab/metrics.py`, `tests/test_metrics.py`
2026-10-04 12:35 | B | Implemented reproducible 12-template clean/ABC-corrupted IOI generator with leading-space GPT-2 answer IDs; 12 tests passed and no model forward ran | `circuitlab/datasets/ioi.py`, `lab_tools.py`, `tests/test_ioi_dataset.py`
2026-10-04 12:23 | B | Added Person B package skeleton, honest `NotImplementedError` API stubs, and CUDA dependency configuration; GPT-2 GPU smoke test passed | `circuitlab/`, `lab_tools.py`, `requirements.txt`
2026-10-04 | A | Hello-world planner/scout/critic handoff works with API key; fixed Windows issues (UTF-8, Git sh on PATH, workspace-scoped key) | hello/
2026-10-04 | A | Installed Omnigent 0.16.0, confirmed CUDA on RTX 4060 | local
2026-10-03 | A+B | Chose circuit discovery on GPT-2 small, wrote build plan | plan doc

2026-10-04 15:12 | B | Investigated canonical ACDC as an external baseline. Its pinned TransformerLens 1.6.1 and torch <2.0 conflict with this toolkit's TransformerLens 3.9.0 and torch 2.14.1; recommend a separately pinned environment and subprocess/JSON bridge under baselines/acdc only after approval. No toolkit code or dependencies changed | assessment only
