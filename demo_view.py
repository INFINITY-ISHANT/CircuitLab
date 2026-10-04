import json
from pathlib import Path
import streamlit as st

HERE = Path(__file__).parent
TRUTH = HERE / "circuitlab" / "datasets" / "ioi_truth.json"
BUDGET = 5000  # forward-pass budget, match what the Planner is given
EXHAUSTIVE_PASSES = 292  # experiments/exhaustive.py, IOI n=100 seed 42, all 144 heads


def load_events(path: Path) -> list[dict]:
    events = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return events


def load_truth() -> dict[str, str]:
    try:
        data = json.loads(TRUTH.read_text(encoding="utf-8"))
        return {h["component"]: h["role"] for h in data.get("heads", [])}
    except (OSError, ValueError, KeyError):
        return {}


def latest_circuit(events: list[dict]) -> list[str]:
    """Components of the newest planner or analyst event that names any."""
    for e in reversed(events):
        if e.get("agent") in ("planner", "analyst") and e.get("components"):
            return [str(c).upper() for c in e["components"]]
    return []


st.set_page_config(page_title="CircuitLab", layout="wide")
st.title("CircuitLab")
st.caption("Agents hunting for the circuit behind a behavior in GPT-2 small. Replay of the recorded research runs (pick one in the sidebar).")

records = sorted(HERE.glob("research_record*.jsonl"))
names = [p.name for p in records] or ["research_record.jsonl"]
default = names.index("research_record_run1.jsonl") if "research_record_run1.jsonl" in names else 0
chosen = st.sidebar.selectbox("Run record", names, index=default)
truth = load_truth()


@st.fragment(run_every=2)
def live():
    events = load_events(HERE / chosen)
    used = sum(int(e.get("passes_used") or 0) for e in events if e.get("agent") == "experimentalist")
    circuit = latest_circuit(events)
    hits = [c for c in circuit if c in truth]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Round", max([e.get("round") or 0 for e in events], default=0))
    c2.metric("Passes in experimentalist records only", f"{used} / {BUDGET}")
    c3.metric("Exhaustive patching", f"{EXHAUSTIVE_PASSES} passes",
              delta=f"{used - EXHAUSTIVE_PASSES:+d} passes for the agents", delta_color="inverse")
    if truth:
        c4.metric("Proposed heads in published circuit", f"{len(hits)} / {len(truth)}")
    c5.metric("Human approvals", sum(1 for e in events if e.get("approved_by")))
    st.progress(min(1.0, used / BUDGET))

    if circuit:
        st.subheader("Heads under consideration")
        st.markdown(" ".join(
            f"`{c}` ({truth[c].replace('_', ' ')})" if c in truth else f"`{c}` (not in published circuit)"
            for c in circuit
        ))
        if truth:
            st.caption("Latest head list a planner or analyst event names: proposed, not necessarily validated. Roles come from the 26 heads of arXiv:2211.00593, shown here for scoring only.")

    st.subheader("Research record")
    for e in reversed(events[-40:]):
        with st.container(border=True):
            st.markdown(f"**{e.get('agent', '?')}** · round {e.get('round')} · {e.get('ts', '')}")
            st.markdown(f"`{str(e.get('action', ''))[:90]}`")
            st.write(str(e.get("decision", ""))[:500])
            if e.get("components"):
                st.caption("components: " + ", ".join(e["components"]))
            if e.get("evidence"):
                st.caption("evidence: " + ", ".join(map(str, e["evidence"])))


live()
