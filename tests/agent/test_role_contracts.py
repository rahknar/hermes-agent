from agent.role_contracts import ROLE_CONTRACTS, get_role_contract


def test_builtin_semantic_roles_are_defined():
    assert set(ROLE_CONTRACTS) == {
        "orchestrator",
        "analyst",
        "coder",
        "expert",
        "webworker",
    }


def test_role_lookup_normalizes_case_and_whitespace():
    assert get_role_contract("  AnAlYsT  ") == ROLE_CONTRACTS["analyst"]


def test_role_lookup_returns_none_for_missing_role():
    assert get_role_contract(None) is None
    assert get_role_contract("") is None
    assert get_role_contract("   ") is None


def test_role_lookup_returns_none_for_unknown_role():
    assert get_role_contract("not-a-role") is None


def test_contracts_identify_their_semantic_responsibility():
    expected_identity = {
        "orchestrator": "You are the Orchestrator.",
        "analyst": "You are the Analyst specialist.",
        "coder": "You are the Coder specialist.",
        "expert": "You are the Expert specialist.",
        "webworker": "You are the Webworker specialist.",
    }

    for role, opening in expected_identity.items():
        assert ROLE_CONTRACTS[role].startswith(opening)
