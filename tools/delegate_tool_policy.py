"""Semantic specialist tool-authority policy for delegated children.

Semantic tool authority and depth-derived delegation capability are separate
axes.  This module composes them into the concrete-name authority ceiling
consumed by AIAgent/model_tools.
"""

from __future__ import annotations

from typing import Optional, Set


_SPECIALIST_TOOL_POLICY = {
    "analyst": frozenset({
        "read_file",
        "search_files",
        "execute_code",
    }),
    "coder": frozenset({
        "read_file",
        "search_files",
        "write_file",
        "patch",
        "execute_code",
        "terminal",
        "process_manage",
    }),
    "expert": frozenset(),
    "webworker": frozenset({
        "web_search",
        "web_extract",
    }),
}


def _specialist_allowed_tool_names(
    semantic_role: Optional[str],
) -> Optional[set[str]]:
    """Return the concrete-name authority ceiling for a specialist role.

    None means no recognized specialist policy applies. Returned policy sets
    are copies so callers may safely compose additional capability constraints
    without mutating the static policy.
    """
    if not semantic_role:
        return None

    role = str(semantic_role).strip().lower()
    policy = _SPECIALIST_TOOL_POLICY.get(role)
    if policy is None:
        return None

    return set(policy)


def _compose_specialist_tool_ceiling(
    semantic_allowed_tool_names: Optional[set[str]],
    effective_role: str,
) -> Optional[Set[str]]:
    """Compose semantic authority with depth-derived delegation capability.

    ``None`` means no semantic specialist policy applies, so no concrete-name
    ceiling is imposed here.

    A concrete set is copied before composition.  ``delegate_task`` is added
    only for an orchestrator-capable child; it is never part of the semantic
    specialist authority itself.
    """
    if effective_role not in {"leaf", "orchestrator"}:
        raise ValueError(f"unknown delegation capability role: {effective_role!r}")

    if semantic_allowed_tool_names is None:
        return None

    allowed = set(semantic_allowed_tool_names)

    if effective_role == "orchestrator":
        allowed.add("delegate_task")

    return allowed
