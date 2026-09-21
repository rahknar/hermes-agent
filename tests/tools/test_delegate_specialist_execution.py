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


# -------------------------------------------------------------------------
# Specialist execution-environment teardown
# -------------------------------------------------------------------------

def test_specialist_execution_cleanup_force_removes_before_clearing_policy(
    monkeypatch,
):
    import tools.delegate_tool_child_run as child_run

    events = []

    monkeypatch.setattr(
        "tools.terminal_tool_lifecycle.cleanup_vm",
        lambda task_id, *, force_remove=False: events.append(
            ("cleanup_vm", task_id, force_remove)
        ),
    )
    monkeypatch.setattr(
        "tools.terminal_tool.clear_task_env_overrides",
        lambda task_id: events.append(("clear_overrides", task_id)),
    )

    child_run._cleanup_specialist_execution(
        "sa-specialist",
        "analyst",
    )

    assert events == [
        ("cleanup_vm", "sa-specialist", True),
        ("clear_overrides", "sa-specialist"),
    ]


@pytest.mark.parametrize(
    "semantic_role",
    ["coder", "  AnAlYsT  "],
)
def test_specialist_execution_cleanup_applies_to_execution_specialists(
    monkeypatch,
    semantic_role,
):
    import tools.delegate_tool_child_run as child_run

    cleaned = []
    cleared = []

    monkeypatch.setattr(
        "tools.terminal_tool_lifecycle.cleanup_vm",
        lambda task_id, *, force_remove=False: cleaned.append(
            (task_id, force_remove)
        ),
    )
    monkeypatch.setattr(
        "tools.terminal_tool.clear_task_env_overrides",
        lambda task_id: cleared.append(task_id),
    )

    child_run._cleanup_specialist_execution(
        "sa-specialist",
        semantic_role,
    )

    assert cleaned == [("sa-specialist", True)]
    assert cleared == ["sa-specialist"]


@pytest.mark.parametrize(
    "semantic_role",
    ["expert", "webworker", None, "", "not-a-role"],
)
def test_specialist_execution_cleanup_ignores_children_without_execution_policy(
    monkeypatch,
    semantic_role,
):
    import tools.delegate_tool_child_run as child_run

    cleaned = []
    cleared = []

    monkeypatch.setattr(
        "tools.terminal_tool_lifecycle.cleanup_vm",
        lambda task_id, *, force_remove=False: cleaned.append(
            (task_id, force_remove)
        ),
    )
    monkeypatch.setattr(
        "tools.terminal_tool.clear_task_env_overrides",
        lambda task_id: cleared.append(task_id),
    )

    child_run._cleanup_specialist_execution(
        "sa-generic",
        semantic_role,
    )

    assert cleaned == []
    assert cleared == []


def test_specialist_execution_cleanup_ignores_missing_child_task_id(
    monkeypatch,
):
    import tools.delegate_tool_child_run as child_run

    cleaned = []
    cleared = []

    monkeypatch.setattr(
        "tools.terminal_tool_lifecycle.cleanup_vm",
        lambda task_id, *, force_remove=False: cleaned.append(
            (task_id, force_remove)
        ),
    )
    monkeypatch.setattr(
        "tools.terminal_tool.clear_task_env_overrides",
        lambda task_id: cleared.append(task_id),
    )

    child_run._cleanup_specialist_execution(
        "",
        "analyst",
    )

    assert cleaned == []
    assert cleared == []


def test_child_run_cleanup_releases_specialist_execution_environment(
    monkeypatch,
):
    import tools.delegate_tool_child_run as child_run

    events = []

    child = SimpleNamespace(
        semantic_role="analyst",
        _delegate_saved_tool_names=None,
        session_id="",
    )
    parent = SimpleNamespace()

    run = child_run._ChildRun(
        child=child,
        parent_agent=parent,
        task_index=0,
        goal="test",
        subagent_id=None,
        child_progress_cb=None,
        child_task_id="sa-cleanup",
    )

    class FakeHeartbeat:
        def stop(self):
            events.append(("heartbeat_stop",))

    monkeypatch.setattr(
        child_run,
        "_detach_child",
        lambda parent_agent, child_agent: events.append(("detach",)),
    )
    monkeypatch.setattr(
        child_run,
        "_close_child",
        lambda child_agent, message: events.append(("close_child",)),
    )
    monkeypatch.setattr(
        child_run,
        "_cleanup_specialist_execution",
        lambda task_id, role: events.append(
            ("execution_cleanup", task_id, role)
        ),
    )

    run.cleanup(
        heartbeat=FakeHeartbeat(),
        child_pool=None,
        leased_cred_id=None,
        close_deferred=False,
    )

    assert ("execution_cleanup", "sa-cleanup", "analyst") in events
    assert events.count(
        ("execution_cleanup", "sa-cleanup", "analyst")
    ) == 1


def test_child_run_cleanup_does_not_release_execution_environment_when_close_deferred(
    monkeypatch,
):
    import tools.delegate_tool_child_run as child_run

    events = []

    child = SimpleNamespace(
        semantic_role="coder",
        _delegate_saved_tool_names=None,
        session_id="",
    )
    parent = SimpleNamespace()

    run = child_run._ChildRun(
        child=child,
        parent_agent=parent,
        task_index=0,
        goal="test",
        subagent_id=None,
        child_progress_cb=None,
        child_task_id="sa-deferred",
    )

    class FakeHeartbeat:
        def stop(self):
            pass

    monkeypatch.setattr(child_run, "_detach_child", lambda *args: None)
    monkeypatch.setattr(
        child_run,
        "_close_child",
        lambda *args: events.append(("close_child",)),
    )
    monkeypatch.setattr(
        child_run,
        "_cleanup_specialist_execution",
        lambda task_id, role: events.append(
            ("execution_cleanup", task_id, role)
        ),
    )

    run.cleanup(
        heartbeat=FakeHeartbeat(),
        child_pool=None,
        leased_cred_id=None,
        close_deferred=True,
    )

    assert events == []


class _FakeDeferredFuture:
    def __init__(self):
        self.callbacks = []
        self._done = False

    def add_done_callback(self, callback):
        self.callbacks.append(callback)

    def done(self):
        return self._done

    def complete(self):
        self._done = True
        for callback in list(self.callbacks):
            callback(self)


def test_deferred_timeout_waits_for_future_before_specialist_cleanup(
    monkeypatch,
):
    import tools.delegate_tool_child_run as child_run

    events = []
    future = _FakeDeferredFuture()
    child = SimpleNamespace(
        semantic_role="analyst",
        _drain_transports_after_abandonment=None,
    )

    monkeypatch.setattr(
        child_run,
        "_close_child",
        lambda child_agent, message: events.append(("close_child",)),
    )
    monkeypatch.setattr(
        child_run,
        "_cleanup_specialist_execution",
        lambda task_id, role: events.append(
            ("execution_cleanup", task_id, role)
        ),
    )

    child_run._defer_close_after_timeout(
        child,
        future,
        child_task_id="sa-deferred",
    )

    assert events == []

    future.complete()

    assert events == [
        ("close_child",),
        ("execution_cleanup", "sa-deferred", "analyst"),
    ]


def test_deferred_timeout_specialist_cleanup_runs_exactly_once(
    monkeypatch,
):
    import tools.delegate_tool_child_run as child_run

    events = []
    future = _FakeDeferredFuture()
    child = SimpleNamespace(
        semantic_role="coder",
        _drain_transports_after_abandonment=None,
    )

    monkeypatch.setattr(child_run, "_close_child", lambda *args: None)
    monkeypatch.setattr(
        child_run,
        "_cleanup_specialist_execution",
        lambda task_id, role: events.append((task_id, role)),
    )

    child_run._defer_close_after_timeout(
        child,
        future,
        child_task_id="sa-deferred",
    )

    assert events == []

    future.complete()

    assert events == [("sa-deferred", "coder")]


def test_deferred_timeout_nonexecution_child_still_uses_cleanup_helper(
    monkeypatch,
):
    import tools.delegate_tool_child_run as child_run

    events = []
    future = _FakeDeferredFuture()
    child = SimpleNamespace(
        semantic_role="webworker",
        _drain_transports_after_abandonment=None,
    )

    monkeypatch.setattr(
        child_run,
        "_close_child",
        lambda *args: events.append(("close_child",)),
    )
    monkeypatch.setattr(
        child_run,
        "_cleanup_specialist_execution",
        lambda task_id, role: events.append(
            ("execution_cleanup", task_id, role)
        ),
    )

    child_run._defer_close_after_timeout(
        child,
        future,
        child_task_id="sa-webworker",
    )

    assert events == []

    future.complete()

    assert events == [
        ("close_child",),
        ("execution_cleanup", "sa-webworker", "webworker"),
    ]


def test_specialist_execution_cleanup_removes_real_child_environment_only(
    monkeypatch,
):
    import tools.delegate_tool_child_run as child_run
    import tools.terminal_tool as terminal_tool

    child_task_id = "sa-real-cleanup"
    parent_task_id = "parent-real-cleanup"
    cleanup_calls = []

    class FakeEnv:
        def __init__(self, name):
            self.name = name

        def cleanup(self, *, force_remove=False):
            cleanup_calls.append((self.name, force_remove))

    child_env = FakeEnv("child")
    parent_env = FakeEnv("parent")

    # Isolate this test from process-global terminal state.
    monkeypatch.setattr(
        terminal_tool,
        "_active_environments",
        {
            child_task_id: child_env,
            parent_task_id: parent_env,
        },
    )
    monkeypatch.setattr(
        terminal_tool,
        "_last_activity",
        {
            child_task_id: 1.0,
            parent_task_id: 1.0,
        },
    )
    monkeypatch.setattr(terminal_tool, "_creation_locks", {})
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.setattr(terminal_tool, "_container_aliases", {})

    terminal_tool.register_container_alias(
        child_task_id,
        parent_task_id,
    )
    terminal_tool.register_task_env_overrides(
        child_task_id,
        {"env_type": "docker"},
    )

    assert terminal_tool.resolve_task_env_type(
        child_task_id,
        {"env_type": "local"},
    ) == "docker"
    assert terminal_tool._resolve_container_task_id(
        child_task_id
    ) == child_task_id

    child_run._cleanup_specialist_execution(
        child_task_id,
        "analyst",
    )

    assert cleanup_calls == [("child", True)]

    assert child_task_id not in terminal_tool._active_environments
    assert child_task_id not in terminal_tool._last_activity
    assert child_task_id not in terminal_tool._task_env_overrides
    assert child_task_id not in terminal_tool._container_aliases

    assert terminal_tool._active_environments[parent_task_id] is parent_env
    assert terminal_tool._last_activity[parent_task_id] == 1.0
    assert cleanup_calls == [("child", True)]

    assert terminal_tool.resolve_task_env_type(
        parent_task_id,
        {"env_type": "local"},
    ) == "local"
