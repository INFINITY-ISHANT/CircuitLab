# Demo script (2 minutes)

Have open before starting: a terminal with `streamlit run demo_view.py` already running in the browser, `docs/RESULTS.md` in a second tab, and `results/recall_vs_forward_passes.png` ready. The lab server (`python lab_server.py --http --max-passes 5000`) does not need to be running for playback.

Do not show the run 2 record. Its head list is the paper's circuit copied in as candidates.

| Time | Say and show |
|---|---|
| 0:00-0:15 | **The problem.** Finding which attention heads implement a behavior in GPT-2 small is slow and costs hundreds of forward passes. Exhaustive patching of all 144 heads costs 292 and cannot see upstream heads. |
| 0:15-0:40 | **The lab and its agents.** A Planner (Sonnet) directs lit_scout, task_designer, hypothesis, experimentalist, safety and analyst. Every experiment goes through one shared tool server, so there is one honest pass counter. |
| 0:40-1:10 | **The live record.** In the Streamlit view use the sidebar to pick `research_record_run1.jsonl`, scroll through the rounds, then switch to `research_record.jsonl` (run 3) and show the planner dropping a broken assumption. |
| 1:10-1:30 | **The claim.** Run 1 claimed L9H9, L10H0, L9H6 with joint faithfulness 0.8827 against a random control of 0.0275, after a human approved it. |
| 1:30-1:50 | **The honest scorecard.** Open `docs/RESULTS.md`. Recall is 3/26 at 366 passes, against 292 for exhaustive. No upstream head is confirmed yet, and the analyst oracle makes recall optimistic. |
| 1:50-2:00 | **The guardrails.** A human approves every claim, a human approves sweeps over 24 heads (one was refused in run 3), and a hard spend cap stopped the models at $1.50 in run 1. |
