def require_human_for_claim(event):
    """Ask a human before the planner can present a final claim."""
    if event.get("type") != "tool_call":
        return None
    name = str(event.get("data", {}).get("name", ""))
    if name.endswith("present_claim"):
        return {"result": "ASK", "reason": "Final claim. A human must approve before it is presented."}
    return None

def require_human_for_big_sweep(event):
    if event.get("type") != "tool_call":
        return None
    data = event.get("data", {})
    name = str(data.get("name", ""))
    if name.endswith("path_patch_sweep"):
        senders = (data.get("arguments") or {}).get("senders") or []
        if isinstance(senders, str):
            senders = [senders]
        if not senders or len(senders) > 24:
            count = len(senders) if senders else "every earlier-layer"
            return {
                "result": "ASK",
                "reason": f"Path-patching sweep over {count} sender heads. A human must approve before it uses this many forward passes.",
            }
        return None
    if not name.endswith(("patch", "ablate")):
        return None
    heads = (data.get("arguments") or {}).get("components") or []
    if isinstance(heads, str):
        heads = [heads]
    if len(heads) > 24:
        return {
            "result": "ASK",
            "reason": f"Sweep over {len(heads)} heads. A human must approve before it uses this many forward passes.",
        }
    return None