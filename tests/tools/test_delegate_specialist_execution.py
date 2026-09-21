"""Tests for applying specialist execution policy to delegated child identity."""

from types import SimpleNamespace

import pytest

import tools.delegate_tool_child_run as child_run


def _make_run(semantic_role):
    child = SimpleNamespace(semantic_role=semantic_role)
    parent = SimpleNamespace(_current_task_id="parent-task")

    return child_run._ChildRun(
        child=child,
        parent_agent=parent,
        task_index=0,
        goal="test specialist execution policy",
        subagent_id="sa-test-child",
        child_progress_cb=None,
    )


@pytest.mark.parametrize(
    "semantic_role",
    ["analyst", "coder", "  AnAlYsT  "],
)
def test_seed_workspace_registers_docker_override_for_execution_specialist(
    monkeypatch,
    semantic_role,
):
    registered = []

    monkeypatch.setattr(
        child_run,
        "_create_isolated_worktree",
        lambda *args, **kwargs: None,
    )

    import tools.terminal_tool as terminal_tool

    monkeypatch.setattr(
        terminal_tool,
        "get_session_cwd",
        lambda task_id: "/workspace" if task_id == "parent-task" else None,
    )
    monkeypatch.setattr(
        terminal_tool,
        "record_session_cwd",
        lambda task_id, cwd: None,
    )
    monkeypatch.setattr(
        terminal_tool,
        "register_container_alias",
        lambda child_task_id, parent_task_id: None,
    )
    monkeypatch.setattr(
        terminal_tool,
        "register_task_env_overrides",
        lambda task_id, overrides: registered.append((task_id, dict(overrides))),
    )

    run = _make_run(semantic_role)
    run.seed_workspace()

    assert run.child_task_id == "sa-test-child"
    assert run.parent_task_id == "parent-task"
    assert registered == [
        ("sa-test-child", {"env_type": "docker"}),
    ]
    assert all(task_id != run.parent_task_id for task_id, _ in registered)


@pytest.mark.parametrize(
    "semantic_role",
    ["expert", "webworker", None, "", "not-a-role"],
)
def test_seed_workspace_does_not_register_execution_override_without_policy(
    monkeypatch,
    semantic_role,
):
    registered = []

    monkeypatch.setattr(
        child_run,
        "_create_isolated_worktree",
        lambda *args, **kwargs: None,
    )

    import tools.terminal_tool as terminal_tool

    monkeypatch.setattr(
        terminal_tool,
        "get_session_cwd",
        lambda task_id: "/workspace" if task_id == "parent-task" else None,
    )
    monkeypatch.setattr(
        terminal_tool,
        "record_session_cwd",
        lambda task_id, cwd: None,
    )
    monkeypatch.setattr(
        terminal_tool,
        "register_container_alias",
        lambda child_task_id, parent_task_id: None,
    )
    monkeypatch.setattr(
        terminal_tool,
        "register_task_env_overrides",
        lambda task_id, overrides: registered.append((task_id, dict(overrides))),
    )

    run = _make_run(semantic_role)
    run.seed_workspace()

    assert run.child_task_id == "sa-test-child"
    assert run.parent_task_id == "parent-task"
    assert registered == []


def test_specialist_override_breaks_child_container_alias_and_preserves_parent_backend(
    monkeypatch,
):
    """A specialist isolation override owns the child backend, never the parent."""
    import tools.terminal_tool as terminal_tool

    parent_task_id = "test-5e2-parent"
    child_task_id = "test-5e2-specialist"

    # Keep this test independent of the user's configured terminal backend.
    local_config = {"env_type": "local"}

    try:
        terminal_tool.register_container_alias(child_task_id, parent_task_id)

        # Before specialist policy is registered, the child has no isolation
        # override and retains the configured local execution backend.
        assert terminal_tool.resolve_task_env_type(
            parent_task_id,
            config=local_config,
        ) == "local"
        assert terminal_tool.resolve_task_env_type(
            child_task_id,
            config=local_config,
        ) == "local"

        terminal_tool.register_task_env_overrides(
            child_task_id,
            {"env_type": "docker"},
        )

        # The raw child override is an isolation signal and therefore wins
        # before ordinary child->parent container alias resolution.
        assert terminal_tool._resolve_container_task_id(child_task_id) == child_task_id
        assert terminal_tool.resolve_task_env_type(
            child_task_id,
            config=local_config,
        ) == "docker"

        # Specialist policy must never mutate the parent's execution backend.
        assert terminal_tool.resolve_task_env_type(
            parent_task_id,
            config=local_config,
        ) == "local"

    finally:
        terminal_tool.clear_task_env_overrides(child_task_id)
        terminal_tool.clear_task_env_overrides(parent_task_id)
