from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.turn_preflight import PreflightGateVerdict, run_preflight_compression


def _verdict(messages):
    return PreflightGateVerdict(
        action="fallthrough",
        pending_moa_prepared_request=None,
        messages=messages,
        active_system_prompt="sys",
        conversation_history=list(messages),
        api_call_count=1,
        compression_attempts=0,
        final_response=None,
        failed=False,
        _turn_exit_reason=None,
        _compression_timeout_exhausted=False,
        _preflight_compression_blocked=False,
        _provider_overflow_recovery_pending=False,
        _last_preflight_pressure=None,
    )


def test_provider_overflow_bypasses_structural_compactability_veto():
    """A provider-proven overflow must still enter forced recovery."""
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]

    compressor = SimpleNamespace(
        threshold_tokens=1_000,
        context_length=8_000,
        should_compress=MagicMock(return_value=True),
        can_compress=MagicMock(return_value=False),
        get_active_compression_failure_cooldown=MagicMock(return_value=None),
    )

    compressed_messages = [
        {"role": "user", "content": "compressed"},
    ]

    agent = SimpleNamespace(
        compression_enabled=True,
        model="test-model",
        context_compressor=compressor,
        _compress_context=MagicMock(
            return_value=(compressed_messages, "compressed-system")
        ),
        _emit_status=MagicMock(),
        _persist_session=MagicMock(),
    )

    verdict = _verdict(messages)

    with (
        patch(
            "agent.turn_preflight._review_fork_first_request_pending",
            return_value=False,
        ),
        patch(
            "agent.conversation_loop._maybe_grow_local_window",
            return_value=None,
        ),
        patch(
            "agent.turn_preflight._clear_overflow_warn",
        ),
        patch(
            "agent.turn_preflight.context_compression_timed_out",
            return_value=False,
        ),
        patch(
            "agent.turn_preflight.compression_skipped_due_to_lock",
            return_value=False,
        ),
        patch(
            "agent.turn_preflight.compression_blocked_transiently",
            return_value=False,
        ),
        patch(
            "agent.turn_preflight._reset_retry_state_after_compaction",
        ),
        patch(
            "agent.turn_preflight.conversation_history_after_compression",
            return_value=compressed_messages,
        ),
        patch(
            "agent.turn_preflight._refund_api_call",
            return_value=0,
        ),
        patch(
            "agent.turn_preflight.automatic_compaction_status_message",
            return_value=None,
        ),
        patch(
            "agent.conversation_loop._should_skip_model_call_for_reference_handoff",
            return_value=False,
        ),
    ):
        result = run_preflight_compression(
            agent,
            verdict,
            compressor=compressor,
            request_pressure_tokens=9_000,
            provider_overflow_preflight=True,
            defer_preflight=lambda _tokens: False,
            moa_prepared_request=None,
            system_message=None,
            user_message="hello",
            max_compression_attempts=3,
            effective_task_id="task",
        )

    compressor.should_compress.assert_called_once_with(9_000)

    # provider_overflow_preflight is the left side of the OR, so the
    # structural compactability hook must not even be consulted.
    compressor.can_compress.assert_not_called()
    agent._compress_context.assert_called_once()

    assert result.action == "continue"
    assert result.messages is compressed_messages
    assert result.compression_attempts == 1
