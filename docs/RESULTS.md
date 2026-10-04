# Results

## 1. Setup

The model is GPT-2 small and the task is indirect object identification (IOI), on a dataset of n=100 prompts with seed 42. A forward pass is one model call on one prompt batch. The ground truth is the 26 heads of the IOI circuit from arXiv:2211.00593 Appendix K, stored in `circuitlab/datasets/ioi_truth.json`. The agents are an Omnigent Planner (Sonnet) with lit_scout, task_designer, hypothesis, experimentalist, safety (Haiku) and analyst (Sonnet). Every experiment goes through one shared MCP tool server, so all agents use the same forward-pass counter.

## 2. Baselines

| Method | Forward passes | What it finds |
|---|---|---|
| Exhaustive activation patching (all 144 heads, final token) | 292 | Top 10 heads hold 8 circuit heads: L9H9, L10H0, L9H6 (name movers) and L10H6, L10H10, L11H2, L10H2, L10H1 (backup name movers). L11H1 and L11H3 are not in the circuit. Final-token patching cannot see upstream heads. File: `results/exhaustive_ioi_seed42_n100_ranked.csv` |
| ACDC (arXiv:2304.14997) | not recorded (cited, not re-run) | Its code needs TransformerLens 1.6.1 and torch<2, which conflict with this toolkit. It reports edge-level ROC/AUC and no forward-pass counts, and it misses negative name movers when optimizing logit difference. |

## 3. Agent runs

| Run | Rule change | Forward passes | Final circuit | Recall vs 26 | Faithfulness vs random control | Human interventions |
|---|---|---|---|---|---|---|
| 1 | Stop when faithfulness is at least 0.7 | 366 | L9H9, L10H0, L9H6 | 3/26 | 0.8827 vs 0.0275 | Approved the final claim. Spend cap hit at $1.50. |
| 2 | Open-ended upstream path-patching sweep (60 senders x 10 receivers) | about 1700 (about 28 per sender) | none, stopped by hand | not recorded | not recorded | Stopped by hand |
| 3 | IOI and ACDC papers hidden; stopping rule requires an upstream trace | 786 | none (no claim presented) | not recorded | not recorded | Approved sweeps over 58 and 101 heads, refused one over 31 heads |

Run 3's circuit would have been assembled by `experiments/run_report.py` from the agents' own experiments using the Planner's closing rule, because the run ended before presenting a claim. That rule gave an empty circuit (see `results/run3_report.json`): the log holds no agent path-patch calls for run 3, so there were no receivers and no senders to add. The 786 passes are the tool calls logged for run 3 (ablate 510, patch 132, faithfulness 12, random control 132).

The last circuit the agents tested in run 3 was a 31-head candidate. Its joint faithfulness was 0.5884 against a random-circuit mean of 0.3276 (80th percentile, `beats_random` true; `results/random_control_ioi-42-100_seed42_k31.json`). It matches 13 of the 26 published heads, but that number is optimistic (see section 6) and it was never presented as a claim.

## 4. What the agents found that exhaustive patching cannot

Nothing confirmed. No upstream head has a path-patch recovery above 0.05 in `results/run3_report.json`, and its final circuit is empty. In run 3 the agents proposed 14 early-layer senders, 4 of them real upstream circuit heads (L0H10 duplicate token, L2H2 and L4H11 previous token, L5H8 induction), against about 1.5 expected by chance from those layers. These are proposals only. They were not confirmed by path patching.

## 5. Safety and human control

- Claim gate: a human approves every final claim. In run 1 the human approved the claim L9H9, L10H0, L9H6.
- Sweep gate: a human approves any patch or ablate over 24 heads. In run 3 the human approved sweeps over 58 and 101 heads and refused one over 31 heads.
- Spend cap: a hard dollar limit through Omnigent's cost_budget policy. In run 1 it blocked the models at $1.50.

## 6. Honest limits

- The agents have not beaten the exhaustive baseline on forward passes: 366 in run 1 and 786 in run 3, against 292.
- Faithfulness saturates with 3 heads (0.8827 in run 1), so it is a poor stopping rule on its own.
- In runs 1 and 3 the analyst could call `score_vs_truth`, which returns counts of correct heads. That is feedback from the answer key, so recall in those runs is optimistic. The analyst config no longer offers that tool.
- Run 3 hid the IOI paper from the literature agent, but the hypothesis agent still cited arXiv 2211.00593 from memory when it proposed the layer 9 and 10 heads (see `research_record.jsonl`). Treat the 13/26 for the 31-head candidate as leaked, not discovered.
- The pass metric counts batched calls, not prompts.
- One seed only (42).
- The raw tool-call log (`results/runs.jsonl`, not tracked) also holds test-suite and report-check entries. The run 3 numbers use only the calls between 10:41 and 11:12:20 UTC on 2026-10-04, kept in `results/run3_calls.jsonl`.

## 7. Next steps

- Hard per-experiment pass limits enforced in code.
- A recovery curve over 5 seeds: recall against cumulative passes for the agents, exhaustive patching and a random baseline.
- A discovery run on a behavior with no published circuit.
- A real ACDC run in a pinned environment.
