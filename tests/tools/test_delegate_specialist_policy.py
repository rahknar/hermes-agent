"""Tests for semantic specialist tool-authority policy composition."""

import pytest


def test_none_semantic_ceiling_preserves_unrestricted_behavior():
    from tools.delegate_tool_policy import _compose_specialist_tool_ceiling

    assert _compose_specialist_tool_ceiling(None, "leaf") is None
    assert _compose_specialist_tool_ceiling(None, "orchestrator") is None


def test_leaf_preserves_semantic_ceiling_without_delegation():
    from tools.delegate_tool_policy import _compose_specialist_tool_ceiling

    result = _compose_specialist_tool_ceiling(
        {"read_file", "execute_code"},
        "leaf",
    )

    assert result == {"read_file", "execute_code"}
    assert "delegate_task" not in result


def test_orchestrator_adds_depth_derived_delegation_capability():
    from tools.delegate_tool_policy import _compose_specialist_tool_ceiling

    result = _compose_specialist_tool_ceiling(
        {"read_file", "execute_code"},
        "orchestrator",
    )

    assert result == {"read_file", "execute_code", "delegate_task"}


def test_empty_semantic_ceiling_remains_empty_for_leaf():
    from tools.delegate_tool_policy import _compose_specialist_tool_ceiling

    assert _compose_specialist_tool_ceiling(set(), "leaf") == set()


def test_empty_semantic_ceiling_only_gains_depth_capability_for_orchestrator():
    from tools.delegate_tool_policy import _compose_specialist_tool_ceiling

    assert _compose_specialist_tool_ceiling(
        set(),
        "orchestrator",
    ) == {"delegate_task"}


def test_composition_does_not_mutate_policy_set():
    from tools.delegate_tool_policy import _compose_specialist_tool_ceiling

    policy = {"read_file"}

    result = _compose_specialist_tool_ceiling(policy, "orchestrator")

    assert policy == {"read_file"}
    assert result == {"read_file", "delegate_task"}


def test_unknown_effective_role_is_rejected():
    from tools.delegate_tool_policy import _compose_specialist_tool_ceiling

    with pytest.raises(ValueError):
        _compose_specialist_tool_ceiling({"read_file"}, "wizard")


# -------------------------------------------------------------------------
# Semantic specialist policy lookup
# -------------------------------------------------------------------------

def test_specialist_policy_recognizes_supported_semantic_roles():
    from tools.delegate_tool_policy import _specialist_allowed_tool_names

    for role in ("analyst", "coder", "expert", "webworker"):
        result = _specialist_allowed_tool_names(role)
        assert result is not None
        assert isinstance(result, set)


def test_specialist_policy_normalizes_role_name():
    from tools.delegate_tool_policy import _specialist_allowed_tool_names

    assert (
        _specialist_allowed_tool_names("  AnAlYsT  ")
        == _specialist_allowed_tool_names("analyst")
    )


def test_unknown_semantic_role_has_no_specialist_ceiling():
    from tools.delegate_tool_policy import _specialist_allowed_tool_names

    assert _specialist_allowed_tool_names("not-a-role") is None


def test_missing_semantic_role_has_no_specialist_ceiling():
    from tools.delegate_tool_policy import _specialist_allowed_tool_names

    assert _specialist_allowed_tool_names(None) is None
    assert _specialist_allowed_tool_names("") is None


def test_specialist_policy_returns_detached_copy():
    from tools.delegate_tool_policy import _specialist_allowed_tool_names

    first = _specialist_allowed_tool_names("analyst")
    second = _specialist_allowed_tool_names("analyst")

    assert first is not second

    first.add("delegate_task")

    assert "delegate_task" not in second
    assert "delegate_task" not in _specialist_allowed_tool_names("analyst")


# -------------------------------------------------------------------------
# Specialist ordinary-tool authority profiles
# -------------------------------------------------------------------------

def test_analyst_policy_is_read_and_compute_only():
    from tools.delegate_tool_policy import _specialist_allowed_tool_names

    assert _specialist_allowed_tool_names("analyst") == {
        "read_file",
        "search_files",
        "execute_code",
    }


def test_coder_policy_can_inspect_edit_execute_and_manage_processes():
    from tools.delegate_tool_policy import _specialist_allowed_tool_names

    assert _specialist_allowed_tool_names("coder") == {
        "read_file",
        "search_files",
        "write_file",
        "patch",
        "execute_code",
        "terminal",
        "process_manage",
    }


def test_expert_policy_has_no_ordinary_tools():
    from tools.delegate_tool_policy import _specialist_allowed_tool_names

    assert _specialist_allowed_tool_names("expert") == set()


def test_webworker_policy_is_web_retrieval_only():
    from tools.delegate_tool_policy import _specialist_allowed_tool_names

    assert _specialist_allowed_tool_names("webworker") == {
        "web_search",
        "web_extract",
    }


def test_semantic_policies_never_grant_delegation_capability():
    from tools.delegate_tool_policy import _specialist_allowed_tool_names

    for role in ("analyst", "coder", "expert", "webworker"):
        assert "delegate_task" not in _specialist_allowed_tool_names(role)


# -------------------------------------------------------------------------
# _build_child_agent integration
# -------------------------------------------------------------------------

def _build_with_mocked_agent(monkeypatch, semantic_role, *, parent_depth, max_spawn_depth):
    from unittest.mock import MagicMock

    import tools.delegate_tool as delegate_tool

    parent = MagicMock()
    parent._delegate_depth = parent_depth
    parent.enabled_toolsets = ["hermes-cli"]
    parent.disabled_toolsets = []
    parent.model = "orchestrator"
    parent.api_key = "test-key"
    parent.base_url = "http://localhost:1235/v1"
    parent.request_overrides = {}
    parent.prefill_messages = None
    parent.session_id = "parent-session"
    parent._subagent_id = None
    parent._fallback_chain = None

    child = MagicMock()

    mock_agent = MagicMock(return_value=child)
    monkeypatch.setattr("run_agent.AIAgent", mock_agent)
    monkeypatch.setattr(delegate_tool, "_get_max_spawn_depth", lambda: max_spawn_depth)
    monkeypatch.setattr(delegate_tool, "_get_orchestrator_enabled", lambda: True)
    monkeypatch.setattr(delegate_tool, "_open_child_session_db", lambda parent_agent: None)

    delegate_tool._build_child_agent(
        task_index=0,
        goal="test specialist authority",
        context=None,
        toolsets=None,
        model=None,
        max_iterations=10,
        parent_agent=parent,
        task_count=1,
        semantic_role=semantic_role,
    )

    return mock_agent.call_args.kwargs


def test_build_child_agent_passes_leaf_analyst_ceiling(monkeypatch):
    kwargs = _build_with_mocked_agent(
        monkeypatch,
        "analyst",
        parent_depth=0,
        max_spawn_depth=1,
    )

    assert kwargs["allowed_tool_names"] == {
        "read_file",
        "search_files",
        "execute_code",
    }


def test_build_child_agent_adds_delegation_only_when_depth_allows(monkeypatch):
    kwargs = _build_with_mocked_agent(
        monkeypatch,
        "analyst",
        parent_depth=0,
        max_spawn_depth=2,
    )

    assert kwargs["allowed_tool_names"] == {
        "read_file",
        "search_files",
        "execute_code",
        "delegate_task",
    }


def test_build_child_agent_passes_explicit_empty_expert_ceiling(monkeypatch):
    kwargs = _build_with_mocked_agent(
        monkeypatch,
        "expert",
        parent_depth=0,
        max_spawn_depth=1,
    )

    assert kwargs["allowed_tool_names"] == set()


def test_build_child_agent_preserves_generic_child_without_specialist_policy(monkeypatch):
    kwargs = _build_with_mocked_agent(
        monkeypatch,
        "not-a-role",
        parent_depth=0,
        max_spawn_depth=1,
    )

    assert kwargs["allowed_tool_names"] is None


# -------------------------------------------------------------------------
# Concrete authority intersection
# -------------------------------------------------------------------------

def _resolved_tool_names(*, enabled_toolsets, disabled_toolsets, allowed_tool_names):
    import model_tools

    definitions = model_tools.get_tool_definitions(
        enabled_toolsets=enabled_toolsets,
        disabled_toolsets=disabled_toolsets,
        allowed_tool_names=allowed_tool_names,
        quiet_mode=True,
        skip_tool_search_assembly=True,
    )

    return {
        definition["function"]["name"]
        for definition in definitions
    }


def test_coder_policy_cannot_manufacture_parent_authority():
    from tools.delegate_tool_policy import (
        _compose_specialist_tool_ceiling,
        _specialist_allowed_tool_names,
    )

    ceiling = _compose_specialist_tool_ceiling(
        _specialist_allowed_tool_names("coder"),
        "leaf",
    )

    # Parent/child authority contains only the file toolset.  Coder's semantic
    # policy mentions execution/process tools, but the ceiling must not create
    # capabilities absent from the underlying selected authority.
    names = _resolved_tool_names(
        enabled_toolsets=["file"],
        disabled_toolsets=[],
        allowed_tool_names=ceiling,
    )

    assert names == {
        "read_file",
        "search_files",
        "write_file",
        "patch",
    }


def test_analyst_policy_removes_mutating_file_authority():
    from tools.delegate_tool_policy import (
        _compose_specialist_tool_ceiling,
        _specialist_allowed_tool_names,
    )

    ceiling = _compose_specialist_tool_ceiling(
        _specialist_allowed_tool_names("analyst"),
        "leaf",
    )

    names = _resolved_tool_names(
        enabled_toolsets=["hermes-cli"],
        disabled_toolsets=[],
        allowed_tool_names=ceiling,
    )

    assert names == {
        "read_file",
        "search_files",
        "execute_code",
    }

    assert "write_file" not in names
    assert "patch" not in names
    assert "terminal" not in names
    assert "process_manage" not in names
    assert "delegate_task" not in names


def test_leaf_coder_cannot_regain_delegate_task():
    from tools.delegate_tool_policy import (
        _compose_specialist_tool_ceiling,
        _specialist_allowed_tool_names,
    )

    ceiling = _compose_specialist_tool_ceiling(
        _specialist_allowed_tool_names("coder"),
        "leaf",
    )

    names = _resolved_tool_names(
        enabled_toolsets=["hermes-cli", "delegation"],
        disabled_toolsets=[],
        allowed_tool_names=ceiling,
    )

    assert "delegate_task" not in names


def test_orchestrator_coder_can_retain_existing_delegate_authority():
    from tools.delegate_tool_policy import (
        _compose_specialist_tool_ceiling,
        _specialist_allowed_tool_names,
    )

    ceiling = _compose_specialist_tool_ceiling(
        _specialist_allowed_tool_names("coder"),
        "orchestrator",
    )

    names = _resolved_tool_names(
        enabled_toolsets=["hermes-cli", "delegation"],
        disabled_toolsets=[],
        allowed_tool_names=ceiling,
    )

    assert "delegate_task" in names


def test_orchestrator_ceiling_does_not_manufacture_delegate_authority():
    from tools.delegate_tool_policy import (
        _compose_specialist_tool_ceiling,
        _specialist_allowed_tool_names,
    )

    ceiling = _compose_specialist_tool_ceiling(
        _specialist_allowed_tool_names("coder"),
        "orchestrator",
    )

    names = _resolved_tool_names(
        enabled_toolsets=["file"],
        disabled_toolsets=[],
        allowed_tool_names=ceiling,
    )

    assert "delegate_task" not in names


def test_expert_policy_resolves_to_zero_ordinary_tools():
    from tools.delegate_tool_policy import (
        _compose_specialist_tool_ceiling,
        _specialist_allowed_tool_names,
    )

    ceiling = _compose_specialist_tool_ceiling(
        _specialist_allowed_tool_names("expert"),
        "leaf",
    )

    names = _resolved_tool_names(
        enabled_toolsets=["hermes-cli"],
        disabled_toolsets=[],
        allowed_tool_names=ceiling,
    )

    assert names == set()


def test_webworker_policy_resolves_to_web_retrieval_only():
    from tools.delegate_tool_policy import (
        _compose_specialist_tool_ceiling,
        _specialist_allowed_tool_names,
    )

    ceiling = _compose_specialist_tool_ceiling(
        _specialist_allowed_tool_names("webworker"),
        "leaf",
    )

    names = _resolved_tool_names(
        enabled_toolsets=["hermes-cli"],
        disabled_toolsets=[],
        allowed_tool_names=ceiling,
    )

    assert names == {
        "web_search",
        "web_extract",
    }


# -------------------------------------------------------------------------
# Tool Search bridge authority propagation
# -------------------------------------------------------------------------

def test_tool_search_bridge_rebuild_preserves_allowed_tool_names(monkeypatch):
    import model_tools
    from tools import tool_search as ts

    captured = {}

    def fake_get_tool_definitions(
        enabled_toolsets=None,
        disabled_toolsets=None,
        quiet_mode=False,
        skip_tool_search_assembly=False,
        allowed_tool_names=None,
    ):
        captured["enabled_toolsets"] = enabled_toolsets
        captured["disabled_toolsets"] = disabled_toolsets
        captured["skip_tool_search_assembly"] = skip_tool_search_assembly
        captured["allowed_tool_names"] = allowed_tool_names
        return []

    monkeypatch.setattr(model_tools, "get_tool_definitions", fake_get_tool_definitions)
    monkeypatch.setattr(ts, "is_bridge_tool", lambda name: True)
    monkeypatch.setattr(
        ts,
        "dispatch_tool_search",
        lambda args, current_tool_defs: "search-result",
    )

    ceiling = {
        "web_search",
        "web_extract",
    }

    result = model_tools._dispatch_bridge_tool(
        ts.TOOL_SEARCH_NAME,
        {},
        ["hermes-cli"],
        [],
        allowed_tool_names=ceiling,
    )

    assert result == ("search-result", None)
    assert captured["skip_tool_search_assembly"] is True
    assert captured["allowed_tool_names"] == ceiling


def test_direct_dispatch_rejects_tool_outside_authority_ceiling(monkeypatch):
    import json
    import model_tools

    dispatched = []

    def fake_dispatch(name, args, **kwargs):
        dispatched.append(name)
        return "SHOULD-NOT-RUN"

    monkeypatch.setattr(model_tools.registry, "dispatch", fake_dispatch)

    result = model_tools.handle_function_call(
        "terminal",
        {"command": "echo bypass"},
        enabled_toolsets=["hermes-cli"],
        disabled_toolsets=[],
        allowed_tool_names={"read_file", "search_files"},
        skip_pre_tool_call_hook=True,
        skip_tool_request_middleware=True,
        skip_tool_execution_middleware=True,
    )

    assert dispatched == []

    payload = json.loads(result)
    assert "error" in payload


def test_direct_dispatch_allows_tool_inside_authority_ceiling(monkeypatch):
    import model_tools

    dispatched = []

    def fake_dispatch(name, args, **kwargs):
        dispatched.append(name)
        return "allowed"

    monkeypatch.setattr(model_tools.registry, "dispatch", fake_dispatch)

    result = model_tools.handle_function_call(
        "read_file",
        {"path": "example.txt"},
        enabled_toolsets=["hermes-cli"],
        disabled_toolsets=[],
        allowed_tool_names={"read_file", "search_files"},
        skip_pre_tool_call_hook=True,
        skip_tool_request_middleware=True,
        skip_tool_execution_middleware=True,
    )

    assert dispatched == ["read_file"]
    assert result == "allowed"


def test_connector_batch_preserves_allowed_tool_names(monkeypatch):
    import json
    import model_tools
    import model_tools_connectors

    captured = []

    def fake_handle_function_call(name, arguments, *args, **kwargs):
        captured.append((name, kwargs.get("allowed_tool_names")))
        return json.dumps({"ok": True})

    monkeypatch.setattr(model_tools, "handle_function_call", fake_handle_function_call)

    connector_name = "connectors__example__search"
    ceiling = frozenset({connector_name})

    ids = model_tools._CallIds(
        task_id="task-1",
        session_id="session-1",
        tool_call_id="call-1",
        turn_id="turn-1",
        api_request_id="request-1",
    )

    result = model_tools_connectors.dispatch_connector_batch(
        [{"name": connector_name, "arguments": {"query": "test"}}],
        ids,
        user_task="test",
        enabled_tools=[connector_name],
        middleware_trace=[],
        enabled_toolsets=["hermes-cli"],
        disabled_toolsets=[],
        allowed_tool_names=ceiling,
    )

    assert captured == [
        (connector_name, ceiling),
    ]
    assert result
