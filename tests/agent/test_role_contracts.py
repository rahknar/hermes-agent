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


def test_orchestrator_contract_expresses_specialist_selection_via_semantic_role():
    contract = ROLE_CONTRACTS["orchestrator"]

    assert "`semantic_role`" in contract
    for role in ("analyst", "coder", "expert", "webworker"):
        assert f"`{role}`" in contract

    assert "Omit `semantic_role` when no specialist role clearly applies." in contract
    assert (
        "`semantic_role` selects responsibility only. Do not use it to select a model, "
        "provider, backend, or delegation capability."
    ) in contract
