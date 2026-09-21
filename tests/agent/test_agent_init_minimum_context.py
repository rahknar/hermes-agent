"""Tests for the Hermes Agent minimum-context admission guard."""

from types import SimpleNamespace

import pytest

from agent.agent_init import _enforce_minimum_context
from agent.delegation_context import delegated_child_context
from agent.model_metadata import MINIMUM_CONTEXT_LENGTH


def _agent(
    context_length: int,
    *,
    provider: str = "custom",
    configured_context_length=None,
):
    return SimpleNamespace(
        model="test-model",
        provider=provider,
        _config_context_length=configured_context_length,
        context_compressor=SimpleNamespace(context_length=context_length),
    )


def test_root_agent_below_minimum_context_is_rejected():
    agent = _agent(MINIMUM_CONTEXT_LENGTH - 1)

    with pytest.raises(ValueError, match="below the minimum"):
        _enforce_minimum_context(agent)


def test_delegated_child_below_minimum_context_is_allowed():
    agent = _agent(32_768)

    with delegated_child_context():
        _enforce_minimum_context(agent)


def test_root_agent_at_minimum_context_is_allowed():
    agent = _agent(MINIMUM_CONTEXT_LENGTH)

    _enforce_minimum_context(agent)


def test_lmstudio_explicit_context_below_minimum_remains_allowed():
    agent = _agent(
        32_768,
        provider="lmstudio",
        configured_context_length=32_768,
    )

    _enforce_minimum_context(agent)
