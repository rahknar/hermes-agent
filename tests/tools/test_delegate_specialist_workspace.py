"""Gate 5E-4a: contained specialist Working Workspace boundary."""

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from tools import delegate_tool_child_run as child_run


def _git(args, cwd):
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )


def _make_repo(root: Path, name: str) -> Path:
    repo = root / name
    repo.mkdir()
    _git(["init", "-q"], repo)
    _git(["config", "user.email", "test@test"], repo)
    _git(["config", "user.name", "Test"], repo)
    (repo / "README.md").write_text(f"{name}\n", encoding="utf-8")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "seed"], repo)
    return repo


def _parent():
    return SimpleNamespace()


def test_specialist_workspace_uses_explicit_source_repo_not_parent_workspace(tmp_path):
    """Contained specialist derives its worktree from Specialist Repo."""
    main_repo = _make_repo(tmp_path, "main")
    specialist_repo = _make_repo(tmp_path, "specialist")
    parent = _parent()

    with (
        mock.patch(
            "tools.delegate_tool._get_worktree_isolation",
            return_value=True,
        ),
        mock.patch(
            "tools.terminal_tool.get_session_cwd",
            return_value=str(main_repo),
        ),
    ):
        info = child_run._create_isolated_worktree(
            parent,
            "parent-task",
            "child-source",
            source_repo=str(specialist_repo),
            required=True,
        )

    assert info is not None
    assert Path(info["repo_root"]).resolve() == specialist_repo.resolve()
    assert Path(info["path"]).resolve().is_relative_to(specialist_repo.resolve())
    assert not Path(info["path"]).resolve().is_relative_to(main_repo.resolve())

    # The Working Workspace contains Specialist Repo content, not Main Repo
    # content merely because the parent session started there.
    assert (Path(info["path"]) / "README.md").read_text(
        encoding="utf-8"
    ) == "specialist\n"


def test_required_specialist_workspace_creation_failure_does_not_fall_back(tmp_path):
    """Required specialist isolation raises rather than sharing Main Workspace."""
    specialist_repo = _make_repo(tmp_path, "specialist")
    parent = _parent()

    with (
        mock.patch(
            "tools.delegate_tool._get_worktree_isolation",
            return_value=True,
        ),
        mock.patch(
            "tools.subagent_worktree.create_subagent_worktree",
            return_value=None,
        ),
    ):
        with pytest.raises(RuntimeError, match="specialist workspace creation failed"):
            child_run._create_isolated_worktree(
                parent,
                "parent-task",
                "child-fail",
                source_repo=str(specialist_repo),
                required=True,
            )


def test_required_specialist_workspace_rejects_wrong_repo_provenance(tmp_path):
    """Returned worktree must belong to the configured Specialist Repo."""
    specialist_repo = _make_repo(tmp_path, "specialist")
    wrong_repo = _make_repo(tmp_path, "wrong")
    parent = _parent()

    fake_info = {
        "path": str(wrong_repo / ".worktrees" / "subagent-child"),
        "branch": "hermes-subagent/subagent-child",
        "repo_root": str(wrong_repo),
        "base_commit": "abc123",
    }

    with (
        mock.patch(
            "tools.delegate_tool._get_worktree_isolation",
            return_value=True,
        ),
        mock.patch(
            "tools.subagent_worktree.create_subagent_worktree",
            return_value=fake_info,
        ),
    ):
        with pytest.raises(
            RuntimeError,
            match="specialist workspace repo provenance mismatch",
        ):
            child_run._create_isolated_worktree(
                parent,
                "parent-task",
                "child-provenance",
                source_repo=str(specialist_repo),
                required=True,
            )


def test_required_specialist_workspace_does_not_depend_on_generic_worktree_flag(tmp_path):
    """Contained specialist isolation is mandatory, not a generic worktree opt-in."""
    specialist_repo = _make_repo(tmp_path, "specialist")
    parent = _parent()

    with mock.patch(
        "tools.delegate_tool._get_worktree_isolation",
        return_value=False,
    ):
        info = child_run._create_isolated_worktree(
            parent,
            "parent-task",
            "child-required",
            source_repo=str(specialist_repo),
            required=True,
        )

    assert info is not None
    assert Path(info["repo_root"]).resolve() == specialist_repo.resolve()
    assert Path(info["path"]).resolve().is_relative_to(specialist_repo.resolve())


@pytest.mark.parametrize(
    ("delegation_cfg", "expected"),
    [
        (
            {"specialist_workspace": {"repo": "/srv/hermes-specialist"}},
            "/srv/hermes-specialist",
        ),
        (
            {"specialist_workspace": {"repo": "  /srv/hermes-specialist  "}},
            "/srv/hermes-specialist",
        ),
        ({"specialist_workspace": {"repo": "   "}}, None),
        ({}, None),
    ],
)
def test_specialist_workspace_repo_config(delegation_cfg, expected):
    from tools import delegate_tool_config as cfg

    with mock.patch.object(cfg, "_cfg", return_value=delegation_cfg):
        assert cfg._get_specialist_workspace_repo() == expected
