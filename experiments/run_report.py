"""Summarize one agent run from results/runs.jsonl and score the circuit it reached.

    python experiments/run_report.py --since 2026-10-04T10:40 --check

--since is a UTC timestamp prefix; every tool call logged after it counts toward the run.
--check also runs faithfulness with a random control on the final circuit (uses the GPU).

The final circuit applies the Planner's own closing rule: every receiver the agents
path-patched into, plus every sender whose joint path recovery is above the threshold.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_calls(path: Path, since: str) -> list[dict]:
    calls = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(record.get("timestamp", "")) >= since:
            calls.append(record)
    return calls


def summarize(calls: list[dict], threshold: float) -> dict:
    passes = Counter()
    for call in calls:
        passes[call.get("tool", "?")] += int(call.get("passes_used") or 0)
    senders: dict[str, float] = {}
    receivers: set[str] = set()
    for call in calls:
        if call.get("tool") != "path_patch":
            continue
        sender, *targets = call.get("components") or [None]
        recovery = (call.get("result_summary") or {}).get("max_recovery")
        if sender is None or recovery is None:
            continue
        receivers.update(targets)
        senders[sender] = max(recovery, senders.get(sender, float("-inf")))
    ranked = sorted(senders.items(), key=lambda item: item[1], reverse=True)
    circuit = sorted(receivers | {s for s, r in ranked if r > threshold})
    return {
        "tool_calls": len(calls),
        "passes_total": sum(passes.values()),
        "passes_by_tool": dict(passes),
        "top_senders": [{"sender": s, "recovery": round(r, 4)} for s, r in ranked[:10]],
        "circuit": circuit,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", required=True, help="UTC ISO prefix, e.g. 2026-10-04T10:40")
    parser.add_argument("--log", default=str(ROOT / "results" / "runs.jsonl"))
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--check", action="store_true", help="run faithfulness and a random control on the GPU")
    args = parser.parse_args()

    report = summarize(load_calls(Path(args.log), args.since), args.threshold)
    import lab_tools

    try:
        score = lab_tools.score_vs_truth(report["circuit"]) if report["circuit"] else None
        report["vs_published"] = None if score is None else {k: score[k] for k in ("tp", "fp", "fn", "precision", "recall")}
    except ValueError as error:
        report["vs_published"] = f"unavailable: {error}"
    if args.check and report["circuit"]:
        ds = lab_tools.make_dataset("ioi", n=100, seed=42)
        faith = lab_tools.faithfulness(report["circuit"], ds=ds)
        control = lab_tools.random_circuit_control(report["circuit"], ds=ds, n_random=10)
        report["faithfulness"] = round(float(faith["faithfulness"]), 4)
        report["random_control"] = {k: control[k] for k in ("random_mean", "beats_random") if k in control}
        report["check_passes"] = lab_tools.budget()["passes_used"]
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
