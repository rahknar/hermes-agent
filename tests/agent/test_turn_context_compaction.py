"""Unit tests for ``agent.turn_context_compaction`` (turn-start compaction extracted
from ``build_turn_context``)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.turn_context_compaction import (
    CompactionOutcome,
    _codex_native_auto_compaction,
    _rearm_uncompressed_overflow_warn,
    _run_preflight_passes,
    run_turn_start_compaction,
)


def _agent(**kw):
    compressor = SimpleNamespace(
        protect_first_n=3, protect_last_n=3, threshold_tokens=1_000, context_length=8_000,
        summary_target_ratio=0.5,
    )
    base = dict(
        compression_enabled=False, context_compressor=compressor, session_id="s1",
        model="m", _clear_context_overflow_warn=MagicMock(),
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_codex_native_auto_compaction_gate():
    assert _codex_native_auto_compaction(
        SimpleNamespace(api_mode="codex_app_server", codex_app_server_auto_compaction="native")
    )
    assert _codex_native_auto_compaction(
        SimpleNamespace(api_mode="codex_app_server", codex_app_server_auto_compaction="OFF")
    )
    assert not _codex_native_auto_compaction(
        SimpleNamespace(api_mode="codex_app_server", codex_app_server_auto_compaction="hermes")
    )
    assert not _codex_native_auto_compaction(SimpleNamespace(api_mode="chat_completions"))


def test_disabled_compression_rearms_overflow_warn_when_under_window():
    agent = _agent()
    msgs = [{"role": "user", "content": "hi"}]
    out = run_turn_start_compaction(
        agent, messages=msgs, system_message=None, active_system_prompt="sys",
        conversation_history=None, current_turn_user_idx=0, user_message="hi",
        effective_task_id="t",
    )
    assert isinstance(out, CompactionOutcome)
    assert out.messages is msgs and out.current_turn_user_idx == 0
    assert out.compressed is False and out.blocked is False
    agent._clear_context_overflow_warn.assert_called_once()
    assert agent._turn_received_provider_response is False
    assert agent._turn_preflight_display_snapshot is None


def test_multimodal_content_forces_real_estimate():
    agent = _agent()
    msgs = [{"role": "user", "content": [{"type": "text", "text": "x"}]}]
    with patch(
        "agent.turn_context._preflight_request_tokens", return_value=9_999
    ) as est:
        _rearm_uncompressed_overflow_warn(agent, msgs, "sys")
    est.assert_called_once()
    agent._clear_context_overflow_warn.assert_not_called()


def test_preflight_gate_skips_small_transcripts():
    agent = _agent(compression_enabled=True)
    agent.context_compressor.should_compress = MagicMock()
    msgs = [{"role": "user", "content": "hi"}]
    with patch("agent.turn_context._preflight_request_tokens") as est:
        out = run_turn_start_compaction(
            agent, messages=msgs, system_message=None, active_system_prompt="sys",
            conversation_history=None, current_turn_user_idx=0, user_message="hi",
            effective_task_id="t",
        )
    est.assert_not_called()
    agent.context_compressor.should_compress.assert_not_called()
    assert out.messages is msgs


def test_preflight_multipass_stops_when_structural_compactability_is_exhausted():
    """Pressure alone must not cause another pass when can_compress() says no."""
    initial = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
    ]
    after_first = [
        {"role": "user", "content": "compressed"},
    ]

    compressor = SimpleNamespace(
        threshold_tokens=1_000,
        context_length=8_000,
        should_compress=MagicMock(return_value=True),
        can_compress=MagicMock(return_value=False),
    )

    agent = SimpleNamespace(
        model="m",
        session_id="s1",
        max_compression_attempts=3,
        _emit_status=MagicMock(),
        _compress_context=MagicMock(return_value=(after_first, "sys")),
    )

    out = CompactionOutcome(
        messages=initial,
        active_system_prompt="sys",
        conversation_history=list(initial),
        current_turn_user_idx=2,
    )

    with (
        patch(
            "agent.turn_context_compaction._clear_overflow_warn",
        ),
        patch(
            "agent.turn_context_compaction.automatic_compaction_status_message",
            return_value=None,
        ),
        patch(
            "agent.turn_context_compaction.compression_skipped_due_to_lock",
            return_value=False,
        ),
        patch(
            "agent.turn_context_compaction.conversation_history_after_compression",
            return_value=after_first,
        ),
        patch(
            "agent.turn_context_compaction._reset_retry_state_after_compaction",
        ),
        patch(
            "agent.turn_context._preflight_request_tokens",
            return_value=1_500,
        ),
        patch(
            "agent.turn_context.compression_made_progress",
            return_value=True,
        ),
        patch(
            "agent.turn_context._compression_warrants_another_preflight_pass",
        ) as another_pass,
    ):
        _run_preflight_passes(
            agent,
            out,
            compressor,
            2_000,
            "sys",
            "t",
        )

    assert agent._compress_context.call_count == 1
    compressor.should_compress.assert_called_once_with(1_500)
    compressor.can_compress.assert_called_once_with(
        after_first,
        prompt_tokens=1_500,
    )
    another_pass.assert_not_called()
    assert out.blocked is False


def test_preflight_multipass_continues_when_structural_compactability_remains():
    """Another pass is allowed when pressure and compactability both remain true."""
    initial = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
    ]
    after_first = [
        {"role": "user", "content": "compressed once"},
        {"role": "assistant", "content": "remaining backlog"},
    ]
    after_second = [
        {"role": "user", "content": "compressed twice"},
    ]

    compressor = SimpleNamespace(
        threshold_tokens=1_000,
        context_length=8_000,
        should_compress=MagicMock(side_effect=[True, False]),
        can_compress=MagicMock(return_value=True),
    )

    agent = SimpleNamespace(
        model="m",
        session_id="s1",
        max_compression_attempts=3,
        _emit_status=MagicMock(),
        _compress_context=MagicMock(
            side_effect=[
                (after_first, "sys"),
                (after_second, "sys"),
            ]
        ),
    )

    out = CompactionOutcome(
        messages=initial,
        active_system_prompt="sys",
        conversation_history=list(initial),
        current_turn_user_idx=2,
    )

    with (
        patch(
            "agent.turn_context_compaction._clear_overflow_warn",
        ),
        patch(
            "agent.turn_context_compaction.automatic_compaction_status_message",
            return_value=None,
        ),
        patch(
            "agent.turn_context_compaction.compression_skipped_due_to_lock",
            return_value=False,
        ),
        patch(
            "agent.turn_context_compaction.conversation_history_after_compression",
            side_effect=lambda _agent, messages, *_args: messages,
        ),
        patch(
            "agent.turn_context_compaction._reset_retry_state_after_compaction",
        ),
        patch(
            "agent.turn_context._preflight_request_tokens",
            side_effect=[1_500, 900],
        ),
        patch(
            "agent.turn_context.compression_made_progress",
            return_value=True,
        ),
        patch(
            "agent.turn_context._compression_warrants_another_preflight_pass",
            return_value=True,
        ) as another_pass,
    ):
        _run_preflight_passes(
            agent,
            out,
            compressor,
            2_000,
            "sys",
            "t",
        )

    assert agent._compress_context.call_count == 2
    compressor.can_compress.assert_called_once_with(
        after_first,
        prompt_tokens=1_500,
    )
    another_pass.assert_called_once_with(2_000, 1_500, 1_000)
    assert out.messages is after_second
    assert out.blocked is False


def test_compactability_veto_does_not_rearm_warning_while_over_threshold():
    """No compactable structure is not evidence that pressure is back below threshold."""
    compressor = SimpleNamespace(
        protect_first_n=3,
        protect_last_n=3,
        threshold_tokens=1_000,
        context_length=8_000,
        summary_target_ratio=0.5,
        should_compress=MagicMock(return_value=True),
        can_compress=MagicMock(return_value=False),
        get_active_compression_failure_cooldown=MagicMock(return_value=None),
    )

    agent = _agent(
        compression_enabled=True,
        context_compressor=compressor,
        _request_pressure_anchored=True,
        _warn_context_overflow_blocked=MagicMock(),
    )

    messages = [
        {"role": "user", "content": "1"},
        {"role": "assistant", "content": "2"},
        {"role": "user", "content": "3"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "5"},
        {"role": "assistant", "content": "6"},
        {"role": "user", "content": "7"},
    ]

    with (
        patch(
            "agent.turn_context._review_fork_first_request_pending",
            return_value=False,
        ),
        patch(
            "agent.turn_context._should_run_preflight_estimate",
            return_value=True,
        ),
        patch(
            "agent.turn_context._preflight_request_tokens",
            return_value=1_500,
        ),
    ):
        out = run_turn_start_compaction(
            agent,
            messages=messages,
            system_message=None,
            active_system_prompt="sys",
            conversation_history=list(messages),
            current_turn_user_idx=6,
            user_message="7",
            effective_task_id="t",
        )

    compressor.should_compress.assert_called_once_with(1_500)
    compressor.can_compress.assert_called_once_with(
        messages,
        prompt_tokens=1_500,
    )

    agent._clear_context_overflow_warn.assert_not_called()
    agent._warn_context_overflow_blocked.assert_not_called()

    assert out.compressed is False
    assert out.blocked is False


def test_enabled_preflight_rearms_warning_once_genuinely_below_threshold():
    """Warning dedup is re-armed only after request pressure falls below threshold."""
    compressor = SimpleNamespace(
        protect_first_n=3,
        protect_last_n=3,
        threshold_tokens=1_000,
        context_length=8_000,
        summary_target_ratio=0.5,
        should_compress=MagicMock(return_value=False),
        can_compress=MagicMock(return_value=False),
        get_active_compression_failure_cooldown=MagicMock(return_value=None),
    )

    agent = _agent(
        compression_enabled=True,
        context_compressor=compressor,
        _request_pressure_anchored=True,
        _warn_context_overflow_blocked=MagicMock(),
    )

    messages = [
        {"role": "user", "content": "1"},
        {"role": "assistant", "content": "2"},
        {"role": "user", "content": "3"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "5"},
        {"role": "assistant", "content": "6"},
        {"role": "user", "content": "7"},
    ]

    with (
        patch(
            "agent.turn_context._review_fork_first_request_pending",
            return_value=False,
        ),
        patch(
            "agent.turn_context._should_run_preflight_estimate",
            return_value=True,
        ),
        patch(
            "agent.turn_context._preflight_request_tokens",
            return_value=900,
        ),
    ):
        out = run_turn_start_compaction(
            agent,
            messages=messages,
            system_message=None,
            active_system_prompt="sys",
            conversation_history=list(messages),
            current_turn_user_idx=6,
            user_message="7",
            effective_task_id="t",
        )

    compressor.should_compress.assert_called_once_with(900)

    # Python short-circuits before structural compactability when pressure
    # itself no longer requests compression.
    compressor.can_compress.assert_not_called()

    agent._clear_context_overflow_warn.assert_called_once()
    agent._warn_context_overflow_blocked.assert_not_called()

    assert out.compressed is False
    assert out.blocked is False
