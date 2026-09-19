from types import SimpleNamespace

import pytest

import agent.turn_final_response as turn_final_response


class ReachedNormalFinalization(Exception):
    """Sentinel proving the zero-tool recovery branch did not claim the response."""


def _assistant(content):
    return SimpleNamespace(
        content=content,
        tool_calls=[],
    )


def _agent(*, valid_tool_names):
    return SimpleNamespace(
        valid_tool_names=valid_tool_names,
        _mute_post_response=False,
        _empty_content_retries=0,
        _thinking_prefill_retries=0,
        _stall_guards=False,
        _intent_ack_continuation=False,

        _has_content_after_think_block=lambda _text: True,
        _emit_pending_fallback_notice=lambda: None,
        _clear_status_buffer=lambda: None,

        _build_assistant_message=lambda message, finish_reason: {
            "role": "assistant",
            "content": message.content,
            "finish_reason": finish_reason,
        },

        # Exclusion tests deliberately stop here. If this is reached, the
        # zero-tool textual-call recovery correctly declined ownership.
        _strip_think_blocks=lambda _text: (_ for _ in ()).throw(
            ReachedNormalFinalization()
        ),

        _session_messages=None,
    )


def _call(
    agent,
    *,
    content="Planning first.\n<tool_call>{\"name\":\"fake\"}</tool_call>",
    zero_tool_textual_call_continuations=0,
):
    messages = []

    verdict = turn_final_response.finish_text_response(
        agent,
        assistant_message=_assistant(content),
        response=SimpleNamespace(),
        finish_reason="stop",
        messages=messages,
        api_messages=[],
        conversation_history=[],
        api_call_count=1,
        user_message="do the delegated task",
        active_system_prompt="system",
        final_response=None,
        _turn_exit_reason=None,
        _preflight_compression_blocked=False,
        codex_ack_continuations=0,
        zero_tool_textual_call_continuations=zero_tool_textual_call_continuations,
        truncated_response_parts=[],
        length_continue_retries=0,
        _pending_verification_response=None,
        _pending_verification_response_previewed=False,
    )

    return verdict, messages


def test_zero_tool_delegated_child_textual_tool_call_gets_one_direct_answer_retry(
    monkeypatch,
):
    monkeypatch.setattr(
        turn_final_response,
        "is_delegated_child_context",
        lambda: True,
    )

    agent = _agent(valid_tool_names=set())

    verdict, messages = _call(agent)

    assert verdict.action == "continue"
    assert verdict.final_response is None
    assert verdict.zero_tool_textual_call_continuations == 1

    assert len(messages) == 2

    interim, nudge = messages

    assert interim["role"] == "assistant"
    assert interim["finish_reason"] == "incomplete"
    assert interim["_zero_tool_textual_call_nudge"] is True
    assert "<tool_call>" in interim["content"]

    assert nudge["role"] == "user"
    assert nudge["_zero_tool_textual_call_nudge"] is True
    assert "No tools are available for this delegated task." in nudge["content"]
    assert "Do not emit tool-call markup" in nudge["content"]
    assert "return the substantive result now" in nudge["content"]

    assert agent._session_messages is messages


def test_zero_tool_retry_budget_is_once_per_turn(monkeypatch):
    monkeypatch.setattr(
        turn_final_response,
        "is_delegated_child_context",
        lambda: True,
    )

    agent = _agent(valid_tool_names=set())

    with pytest.raises(ReachedNormalFinalization):
        _call(
            agent,
            zero_tool_textual_call_continuations=1,
        )


def test_root_agent_textual_tool_call_is_not_claimed(monkeypatch):
    monkeypatch.setattr(
        turn_final_response,
        "is_delegated_child_context",
        lambda: False,
    )

    agent = _agent(valid_tool_names=set())

    with pytest.raises(ReachedNormalFinalization):
        _call(agent)


def test_tool_capable_delegated_child_textual_tool_call_is_not_claimed(monkeypatch):
    monkeypatch.setattr(
        turn_final_response,
        "is_delegated_child_context",
        lambda: True,
    )

    agent = _agent(valid_tool_names={"delegate_task"})

    with pytest.raises(ReachedNormalFinalization):
        _call(agent)
