"""Validated loading of published IOI circuit ground truth.

This module does not supply a head list. It only validates and loads the
repository-owned source file once that file has been populated from an explicit
source. The current placeholder is deliberately rejected for scoring.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


IOI_TRUTH_PATH = Path(__file__).with_name("ioi_truth.json")
_COMPONENT_PATTERN = re.compile(r"^L(?P<layer>\d+)H(?P<head>\d+)$")
_REQUIRED_HEAD_FIELDS = frozenset({"layer", "head", "component", "role", "source"})


def _validate_component(component: object, layer: object, head: object, *, context: str) -> str:
    if not isinstance(layer, int) or not 0 <= layer <= 11:
        raise ValueError(f"{context}.layer must be an integer in [0, 11].")
    if not isinstance(head, int) or not 0 <= head <= 11:
        raise ValueError(f"{context}.head must be an integer in [0, 11].")
    if not isinstance(component, str):
        raise ValueError(f"{context}.component must be a string in form 'LxHy'.")
    match = _COMPONENT_PATTERN.fullmatch(component)
    if match is None or int(match["layer"]) != layer or int(match["head"]) != head:
        raise ValueError(f"{context}.component must exactly match layer/head as 'L{layer}H{head}'.")
    return component


def validate_ioi_truth(data: object, *, require_verified: bool = True) -> dict[str, Any]:
    """Validate an IOI head-list artifact without changing its scientific claims.

    Repeated components are allowed only when their ``role`` values differ;
    this is the explicit representation for a source that assigns multiple
    roles to one head. Exact duplicate component/role pairs are rejected.
    """
    if not isinstance(data, dict):
        raise ValueError("IOI truth data must be a JSON object.")
    if data.get("task") != "ioi":
        raise ValueError("IOI truth data must have task='ioi'.")
    if require_verified and data.get("status") != "verified":
        raise ValueError(
            "IOI ground truth is not verified. Populate ioi_truth.json from an explicit source before scoring."
        )
    if not isinstance(data.get("source"), str) or not data["source"]:
        raise ValueError("IOI truth metadata must include a non-empty source string.")
    heads = data.get("heads")
    if not isinstance(heads, list) or not heads:
        raise ValueError("Verified IOI ground truth must contain at least one attention-head entry.")

    component_roles: set[tuple[str, str]] = set()
    for index, entry in enumerate(heads):
        context = f"heads[{index}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{context} must be an object.")
        missing = _REQUIRED_HEAD_FIELDS - set(entry)
        if missing:
            raise ValueError(f"{context} is missing fields {sorted(missing)}.")
        component = _validate_component(entry["component"], entry["layer"], entry["head"], context=context)
        if not isinstance(entry["role"], str) or not entry["role"]:
            raise ValueError(f"{context}.role must be a non-empty string.")
        if not isinstance(entry["source"], str) or not entry["source"]:
            raise ValueError(f"{context}.source must be a non-empty string.")
        component_role = (component, entry["role"])
        if component_role in component_roles:
            raise ValueError(f"{context} duplicates component/role {component_role!r}.")
        component_roles.add(component_role)
    return data


def load_ioi_truth(path: Path | str = IOI_TRUTH_PATH, *, require_verified: bool = True) -> dict[str, Any]:
    """Load and validate the repository's IOI circuit source artifact."""
    source_path = Path(path)
    with source_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return validate_ioi_truth(data, require_verified=require_verified)


def normalise_attention_components(circuit: object) -> set[str]:
    """Validate stable attention-head identities and return their set identity."""
    if not isinstance(circuit, (list, tuple, set)):
        raise TypeError("circuit must be a list, tuple, or set of component strings such as 'L9H9'.")
    components: set[str] = set()
    for index, component in enumerate(circuit):
        if not isinstance(component, str):
            raise TypeError(f"circuit[{index}] must be a component string.")
        match = _COMPONENT_PATTERN.fullmatch(component)
        if match is None:
            raise ValueError(f"circuit[{index}]={component!r} is not in stable form 'LxHy'.")
        layer, head = int(match["layer"]), int(match["head"])
        _validate_component(component, layer, head, context=f"circuit[{index}]")
        components.add(component)
    return components
