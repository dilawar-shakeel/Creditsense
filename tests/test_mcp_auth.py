import asyncio
from unittest.mock import patch

from mcp.server.auth.provider import AccessToken

from creditsense.mcp_server.auth import (
    SCOPE_READ,
    SCOPE_WRITE,
    ApiKeyTokenVerifier,
    AuthorizationError,
    caller_identity,
    require_scope,
    tool_scope,
)


def _verify(token: str) -> AccessToken | None:
    return asyncio.run(ApiKeyTokenVerifier().verify_token(token))


def test_read_key_grants_only_the_read_scope():
    result = _verify("creditsense-read-demo-key")
    assert result is not None
    assert result.scopes == [SCOPE_READ]


def test_write_key_grants_both_scopes():
    result = _verify("creditsense-write-demo-key")
    assert result is not None
    assert SCOPE_READ in result.scopes
    assert SCOPE_WRITE in result.scopes


def test_unknown_token_is_rejected():
    assert _verify("not-a-real-key") is None


def test_blank_token_is_rejected():
    assert _verify("") is None


def test_tool_scope_maps_write_tool_correctly():
    assert tool_scope("submit_underwriting_decision") == SCOPE_WRITE


def test_tool_scope_maps_read_tools_correctly():
    assert tool_scope("get_applicant_financials") == SCOPE_READ
    assert tool_scope("search_sbp_regulations") == SCOPE_READ
    assert tool_scope("get_portfolio_exposure") == SCOPE_READ


def test_require_scope_is_a_noop_with_no_access_token_in_context():
    """The stdio case: no token concept at all -- must not raise."""
    with patch("creditsense.mcp_server.auth.get_access_token", return_value=None):
        require_scope("submit_underwriting_decision")  # must not raise


def test_require_scope_allows_a_token_with_the_needed_scope():
    token = AccessToken(token="x", client_id="c", scopes=[SCOPE_READ, SCOPE_WRITE])
    with patch("creditsense.mcp_server.auth.get_access_token", return_value=token):
        require_scope("submit_underwriting_decision")  # must not raise


def test_require_scope_rejects_a_read_only_token_calling_the_write_tool():
    token = AccessToken(token="x", client_id="c", scopes=[SCOPE_READ])
    with patch("creditsense.mcp_server.auth.get_access_token", return_value=token):
        try:
            require_scope("submit_underwriting_decision")
            assert False, "expected AuthorizationError"
        except AuthorizationError:
            pass


def test_caller_identity_falls_back_to_stdio_local_with_no_token():
    with patch("creditsense.mcp_server.auth.get_access_token", return_value=None):
        assert caller_identity() == "stdio-local"


def test_caller_identity_uses_the_token_client_id_when_present():
    token = AccessToken(token="x", client_id="creditsense-write-client", scopes=[SCOPE_READ])
    with patch("creditsense.mcp_server.auth.get_access_token", return_value=token):
        assert caller_identity() == "creditsense-write-client"
