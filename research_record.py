"""Append-only research record. One JSON line per agent action."""
import json
import time
from pathlib import Path

RECORD = Path(__file__).parent / "research_record.jsonl"


def log_event(round: int, agent: str, action: str, hypothesis_id: str = "",
              components: list[str] | None = None, evidence: list[str] | None = None,
              passes_used: int = 0, decision: str = "", approved_by: str = "") -> dict:
    """Record one agent action. evidence holds arXiv IDs or run IDs that back the action.
    approved_by is "human", "safety_agent" or empty."""
    event = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "round": round, "agent": agent, "action": action,
        "hypothesis_id": hypothesis_id, "components": components or [],
        "evidence": evidence or [], "passes_used": passes_used,
        "decision": decision, "approved_by": approved_by,
    }
    with RECORD.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
    return {"logged": True, "line": sum(1 for _ in RECORD.open(encoding="utf-8"))}


def read_record(last_n: int = 20) -> list[dict]:
    """Read the most recent entries so an agent can see what has been decided so far."""
    if not RECORD.exists():
        return []
    lines = RECORD.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines[-last_n:]]