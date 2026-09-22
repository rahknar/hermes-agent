from tools.delegate_tool_policy import _specialist_execution_overrides


def test_analyst_requests_strict_specialist_containment():
    overrides = _specialist_execution_overrides("analyst")

    assert overrides == {
        "env_type": "docker",
        "specialist_containment": True,
    }


def test_coder_requests_strict_specialist_containment():
    overrides = _specialist_execution_overrides("coder")

    assert overrides == {
        "env_type": "docker",
        "specialist_containment": True,
    }


def test_nonexecuting_specialists_do_not_request_containment():
    assert _specialist_execution_overrides("expert") is None
    assert _specialist_execution_overrides("webworker") is None


def test_specialist_execution_overrides_returns_fresh_mapping():
    first = _specialist_execution_overrides("analyst")
    second = _specialist_execution_overrides("analyst")

    assert first is not second
    first["specialist_containment"] = False
    assert second["specialist_containment"] is True


def test_specialist_containment_reaches_container_config():
    """Strict specialist intent must survive container-config filtering."""
    from tools.terminal_tool_backends import _container_config_from_config

    config = {
        "env_type": "docker",
        "specialist_containment": True,
    }

    container_config = _container_config_from_config(config)

    assert container_config["specialist_containment"] is True


def test_specialist_containment_forces_strict_docker_constructor_policy(monkeypatch):
    """Contained specialists must not inherit permissive generic Docker settings."""
    from tools import terminal_tool_backends as backends

    captured = {}

    class FakeDockerEnvironment:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(backends, "_DockerEnvironment", FakeDockerEnvironment)
    from tools import terminal_tool

    monkeypatch.setattr(
        terminal_tool,
        "_maybe_reap_docker_orphans",
        lambda cc: None,
    )

    # Deliberately hostile ordinary Docker configuration.  Strict specialist
    # containment must override these rather than inherit them.
    cc = {
        "specialist_containment": True,
        "container_cpu": 2,
        "container_memory": 1024,
        "container_disk": 2048,
        "container_persistent": True,
        "docker_volumes": ["/host/secret:/secret:rw"],
        "docker_mount_cwd_to_workspace": True,
        "docker_forward_env": ["HOME", "SSH_AUTH_SOCK"],
        "docker_env": {"SECRET": "visible"},
        "docker_run_as_host_user": True,
        "docker_network": True,
        "docker_extra_args": ["--privileged"],
        "docker_persist_across_processes": True,
        "docker_shared_container_key": "shared-specialists",
        "docker_shm_size": "1g",
        "docker_snap_compat": True,
    }

    backends._build_docker_env(
        image="specialist-image",
        cwd="/workspace",
        timeout=300,
        cc=cc,
        task_id="sa-contained-child",
        host_cwd="/trusted/specialist-worktree",
    )

    assert captured["network"] is False
    assert captured["volumes"] == []
    assert captured["auto_mount_cwd"] is False
    assert captured["forward_env"] == []
    assert captured["env"] == {}
    assert captured["run_as_host_user"] is False
    assert captured["extra_args"] == []
    assert captured["persist_across_processes"] is False
    assert captured["shared_container_key"] == ""
    assert captured["snap_compat"] is False
    assert captured["persistent_filesystem"] is False

    # Resource limits remain positive rather than being disabled by the
    # containment profile.
    assert captured["cpu"] > 0
    assert captured["memory"] > 0

    # The only host path supplied to Docker is the Hermes-proven Working
    # Workspace.  Mount semantics themselves are enforced by DockerEnvironment.
    assert captured["host_cwd"] == "/trusted/specialist-worktree"


def test_docker_builder_passes_specialist_containment_to_environment(monkeypatch):
    """The Docker layer must know strict specialist policy is in force."""
    from tools import terminal_tool_backends as backends
    from tools import terminal_tool

    captured = {}

    class FakeDockerEnvironment:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(backends, "_DockerEnvironment", FakeDockerEnvironment)
    monkeypatch.setattr(
        terminal_tool,
        "_maybe_reap_docker_orphans",
        lambda cc: None,
    )

    backends._build_docker_env(
        image="specialist-image",
        cwd="/workspace",
        timeout=300,
        cc={
            "specialist_containment": True,
            "container_cpu": 1,
            "container_memory": 512,
        },
        task_id="sa-contained-child",
        host_cwd="/trusted/specialist-worktree",
    )

    assert captured["specialist_containment"] is True


def test_strict_specialist_requires_positive_cpu_and_memory(monkeypatch):
    """Strict containment must fail closed when required resource limits are disabled."""
    from tools import terminal_tool_backends as backends
    from tools import terminal_tool

    class FakeDockerEnvironment:
        def __init__(self, **kwargs):
            raise AssertionError("DockerEnvironment must not be constructed")

    monkeypatch.setattr(backends, "_DockerEnvironment", FakeDockerEnvironment)
    monkeypatch.setattr(
        terminal_tool,
        "_maybe_reap_docker_orphans",
        lambda cc: None,
    )

    import pytest

    with pytest.raises(RuntimeError, match="CPU.*memory|memory.*CPU"):
        backends._build_docker_env(
            image="specialist-image",
            cwd="/workspace",
            timeout=300,
            cc={
                "specialist_containment": True,
                "container_cpu": 0,
                "container_memory": 0,
            },
            task_id="sa-contained-child",
            host_cwd="/trusted/specialist-worktree",
        )


def test_strict_specialist_requires_working_workspace(monkeypatch):
    """Strict containment must never start without its proven host workspace."""
    from tools import terminal_tool_backends as backends
    from tools import terminal_tool

    class FakeDockerEnvironment:
        def __init__(self, **kwargs):
            raise AssertionError("DockerEnvironment must not be constructed")

    monkeypatch.setattr(backends, "_DockerEnvironment", FakeDockerEnvironment)
    monkeypatch.setattr(
        terminal_tool,
        "_maybe_reap_docker_orphans",
        lambda cc: None,
    )

    import pytest

    with pytest.raises(RuntimeError, match="Working Workspace"):
        backends._build_docker_env(
            image="specialist-image",
            cwd="/workspace",
            timeout=300,
            cc={
                "specialist_containment": True,
                "container_cpu": 1,
                "container_memory": 512,
            },
            task_id="sa-contained-child",
            host_cwd=None,
        )


def test_strict_specialist_mounts_only_working_workspace(monkeypatch, tmp_path):
    """Strict Docker mounts exactly the proven Working Workspace and ephemeral home/root."""
    from tools.environments import docker as docker_mod

    workspace = tmp_path / "specialist-worktree"
    workspace.mkdir()

    # Exercise the real DockerEnvironment constructor and argv assembly without
    # requiring or starting a Docker daemon.
    monkeypatch.setattr(docker_mod, "_ensure_docker_available", lambda: None)
    monkeypatch.setattr(docker_mod, "find_docker", lambda: "docker")
    monkeypatch.setattr(
        docker_mod,
        "_image_uses_init_entrypoint",
        lambda docker_exe, image: False,
    )
    monkeypatch.setattr(
        docker_mod,
        "_cgroup_limits_available",
        lambda image: True,
    )
    monkeypatch.setattr(
        docker_mod.DockerEnvironment,
        "_storage_opt_supported",
        lambda self: False,
    )
    monkeypatch.setattr(
        docker_mod.DockerEnvironment,
        "_docker_run",
        lambda self, cwd: "contained-test-container",
    )
    monkeypatch.setattr(
        docker_mod.DockerEnvironment,
        "init_session",
        lambda self: None,
    )
    monkeypatch.setattr(
        docker_mod.DockerEnvironment,
        "cleanup",
        lambda self, *args, **kwargs: None,
    )

    # Make any automatic credential/skill/cache mounts visible if the
    # constructor still attempts to add them.
    monkeypatch.setattr(
        docker_mod,
        "_readonly_skill_mount_args",
        lambda: ["-v", "/host/credential:/credential:ro"],
    )

    env = docker_mod.DockerEnvironment(
        image="specialist-image",
        cwd="/workspace",
        timeout=300,
        cpu=1,
        memory=512,
        disk=0,
        persistent_filesystem=False,
        task_id="sa-contained-child",
        volumes=[],
        forward_env=[],
        env={},
        network=False,
        host_cwd=str(workspace),
        auto_mount_cwd=False,
        run_as_host_user=False,
        extra_args=[],
        persist_across_processes=False,
        shared_container_key="",
        snap_compat=False,
        specialist_containment=True,
    )

    args = env._all_run_args

    assert "--privileged" not in args
    assert "no-new-privileges" in args
    assert not any(
        "docker.sock" in str(arg)
        for arg in args
    )

    workspace_mounts = [
        args[i + 1]
        for i in range(len(args) - 1)
        if args[i] == "-v" and args[i + 1].endswith(":/workspace")
    ]
    assert workspace_mounts == [f"{workspace}:/workspace"]

    assert "/workspace:rw,exec,size=10g" not in args
    assert "/home:rw,exec,size=1g" in args
    assert "/root:rw,exec,size=1g" in args

    assert "/host/credential:/credential:ro" not in args

    host_mounts = [
        args[i + 1]
        for i in range(len(args) - 1)
        if args[i] == "-v"
    ]
    assert host_mounts == [f"{workspace}:/workspace"]


def test_strict_specialist_suppresses_egress_proxy_injection(monkeypatch, tmp_path):
    """Strict specialists must receive no Hermes proxy CA, host alias, or proxy credentials."""
    from tools.environments import docker as docker_mod

    workspace = tmp_path / "specialist-worktree"
    workspace.mkdir()

    monkeypatch.setattr(docker_mod, "_ensure_docker_available", lambda: None)
    monkeypatch.setattr(docker_mod, "find_docker", lambda: "docker")
    monkeypatch.setattr(
        docker_mod,
        "_image_uses_init_entrypoint",
        lambda docker_exe, image: False,
    )
    monkeypatch.setattr(
        docker_mod,
        "_cgroup_limits_available",
        lambda image: True,
    )
    monkeypatch.setattr(
        docker_mod.DockerEnvironment,
        "_storage_opt_supported",
        lambda self: False,
    )
    monkeypatch.setattr(
        docker_mod.DockerEnvironment,
        "_docker_run",
        lambda self, cwd: "contained-test-container",
    )
    monkeypatch.setattr(
        docker_mod.DockerEnvironment,
        "init_session",
        lambda self: None,
    )
    monkeypatch.setattr(
        docker_mod.DockerEnvironment,
        "cleanup",
        lambda self, *args, **kwargs: None,
    )
    monkeypatch.setattr(
        docker_mod,
        "_readonly_skill_mount_args",
        lambda: [],
    )

    # Simulate the ordinary Hermes egress proxy being fully enabled.  Strict
    # containment must not consume any of these values.
    monkeypatch.setattr(
        docker_mod,
        "_egress_proxy_args_for_docker",
        lambda: (
            ["-v", "/host/proxy-ca.pem:/proxy-ca.pem:ro"],
            {
                "HTTPS_PROXY": "http://host.docker.internal:1234",
                "HERMES_PROXY_TOKEN_TEST": "secret-token",
            },
            ["--add-host", "host.docker.internal:host-gateway"],
        ),
    )

    env = docker_mod.DockerEnvironment(
        image="specialist-image",
        cwd="/workspace",
        timeout=300,
        cpu=1,
        memory=512,
        disk=0,
        persistent_filesystem=False,
        task_id="sa-contained-child",
        volumes=[],
        forward_env=[],
        env={},
        network=False,
        host_cwd=str(workspace),
        auto_mount_cwd=False,
        run_as_host_user=False,
        extra_args=[],
        persist_across_processes=False,
        shared_container_key="",
        snap_compat=False,
        specialist_containment=True,
    )

    args = env._all_run_args

    assert "/host/proxy-ca.pem:/proxy-ca.pem:ro" not in args
    assert "host.docker.internal:host-gateway" not in args
    assert "-e" not in args

    assert env._run_env_values == {}


def test_strict_specialist_fails_closed_without_cgroup_limits(monkeypatch):
    """Strict specialists must not run when CPU/memory/PID cgroups cannot be enforced."""
    import pytest
    from tools.environments import docker as docker_mod

    env = docker_mod.DockerEnvironment.__new__(docker_mod.DockerEnvironment)
    env._specialist_containment = True

    monkeypatch.setattr(
        docker_mod,
        "_cgroup_limits_available",
        lambda image: False,
    )

    with pytest.raises(RuntimeError, match="cgroup"):
        env._resource_args(
            image="specialist-image",
            cpu=1,
            memory=512,
            disk=0,
            network=False,
            shm_size="1g",
            extra_args=[],
        )


def test_generic_docker_preserves_cgroup_fallback(monkeypatch):
    """The strict fail-closed rule must not change ordinary Docker compatibility."""
    from tools.environments import docker as docker_mod

    env = docker_mod.DockerEnvironment.__new__(docker_mod.DockerEnvironment)
    env._specialist_containment = False

    monkeypatch.setattr(
        docker_mod,
        "_cgroup_limits_available",
        lambda image: False,
    )

    args = env._resource_args(
        image="generic-image",
        cpu=1,
        memory=512,
        disk=0,
        network=False,
        shm_size="1g",
        extra_args=[],
    )

    assert "--cpus" not in args
    assert "--memory" not in args
    assert "--pids-limit" not in args
    assert "--network=none" in args


def test_strict_specialist_emits_cpu_memory_and_pid_limits(monkeypatch):
    """Available cgroups must produce all required strict resource controls."""
    from tools.environments import docker as docker_mod

    env = docker_mod.DockerEnvironment.__new__(docker_mod.DockerEnvironment)
    env._specialist_containment = True

    monkeypatch.setattr(
        docker_mod,
        "_cgroup_limits_available",
        lambda image: True,
    )

    args = env._resource_args(
        image="specialist-image",
        cpu=1,
        memory=512,
        disk=0,
        network=False,
        shm_size="1g",
        extra_args=[],
    )

    assert args[args.index("--cpus") + 1] == "1"
    assert args[args.index("--memory") + 1] == "512m"
    assert "--pids-limit" in args
    assert args[args.index("--pids-limit") + 1]
    assert "--network=none" in args


def test_strict_specialist_resource_boundary_requires_positive_cpu_and_memory(monkeypatch):
    """DockerEnvironment itself must reject incomplete strict resource limits."""
    import pytest
    from tools.environments import docker as docker_mod

    env = docker_mod.DockerEnvironment.__new__(docker_mod.DockerEnvironment)
    env._specialist_containment = True

    monkeypatch.setattr(
        docker_mod,
        "_cgroup_limits_available",
        lambda image: True,
    )

    for cpu, memory in ((0, 512), (1, 0)):
        with pytest.raises(RuntimeError, match="positive CPU and memory"):
            env._resource_args(
                image="specialist-image",
                cpu=cpu,
                memory=memory,
                disk=0,
                network=False,
                shm_size="1g",
                extra_args=[],
            )
