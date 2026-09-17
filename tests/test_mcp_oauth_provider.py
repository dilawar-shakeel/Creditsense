"""Unit tests for mcp_server/oauth_provider.py -- the auto-approving OAuth
authorization server added so Claude Desktop's chat connector (which does real OAuth
discovery + dynamic client registration and has nowhere to paste a bearer token) can
connect. Mirrors test_mcp_auth.py's style: direct calls, asyncio.run, no server.

The HTTP-level flow through the real routes is covered separately in
test_mcp_oauth_http.py.
"""

from __future__ import annotations

import asyncio
import time
from urllib.parse import parse_qs, urlparse

import pytest
from mcp.server.auth.provider import (
    AuthorizationParams,
    OAuthClientInformationFull,
    TokenError,
)
from pydantic import AnyUrl

from creditsense.mcp_server.auth import SCOPE_READ, SCOPE_WRITE
from creditsense.mcp_server.oauth_provider import CreditSenseOAuthProvider

REDIRECT_URI = "https://claude.ai/api/mcp/auth_callback"


def _client(client_id: str = "client-1") -> OAuthClientInformationFull:
    return OAuthClientInformationFull(
        client_id=client_id,
        redirect_uris=[AnyUrl(REDIRECT_URI)],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method="none",
    )


def _params(state: str = "state-abc") -> AuthorizationParams:
    return AuthorizationParams(
        state=state,
        scopes=[SCOPE_READ, SCOPE_WRITE],
        code_challenge="challenge-xyz",
        redirect_uri=AnyUrl(REDIRECT_URI),
        redirect_uri_provided_explicitly=True,
        resource="http://localhost:8000/mcp",
    )


def _authorize(provider: CreditSenseOAuthProvider, client: OAuthClientInformationFull) -> str:
    """Runs authorize() and returns the issued code from the redirect URL."""
    url = asyncio.run(provider.authorize(client, _params()))
    return parse_qs(urlparse(url).query)["code"][0]


def test_register_then_get_client_round_trips():
    provider = CreditSenseOAuthProvider()
    client = _client()

    asyncio.run(provider.register_client(client))

    assert asyncio.run(provider.get_client("client-1")) is client
    assert asyncio.run(provider.get_client("never-registered")) is None


def test_authorize_redirects_back_with_a_code_and_the_original_state():
    provider = CreditSenseOAuthProvider()

    url = asyncio.run(provider.authorize(_client(), _params(state="state-abc")))

    query = parse_qs(urlparse(url).query)
    assert url.startswith(REDIRECT_URI)
    assert query["state"] == ["state-abc"]
    assert query["code"][0]


def test_authorization_code_preserves_pkce_challenge_and_resource():
    """The provider does not verify PKCE itself (the SDK's token handler does that) --
    its job is to persist the challenge faithfully so that check can happen."""
    provider = CreditSenseOAuthProvider()
    client = _client()
    code = _authorize(provider, client)

    stored = asyncio.run(provider.load_authorization_code(client, code))

    assert stored is not None
    assert stored.code_challenge == "challenge-xyz"
    assert stored.resource == "http://localhost:8000/mcp"
    assert stored.scopes == [SCOPE_READ, SCOPE_WRITE]


def test_authorization_code_is_not_loadable_by_a_different_client():
    provider = CreditSenseOAuthProvider()
    code = _authorize(provider, _client("client-1"))

    assert asyncio.run(provider.load_authorization_code(_client("someone-else"), code)) is None


def test_expired_authorization_code_is_rejected(monkeypatch):
    provider = CreditSenseOAuthProvider()
    client = _client()
    code = _authorize(provider, client)

    # Captured before patching: oauth_provider imports the `time` module, so patching
    # time.time replaces it for this test's own calls too -- a lambda calling
    # time.time() here would recurse into itself.
    later = time.time() + 3600
    monkeypatch.setattr("creditsense.mcp_server.oauth_provider.time.time", lambda: later)

    assert asyncio.run(provider.load_authorization_code(client, code)) is None


def test_full_exchange_issues_a_token_that_authenticates_with_both_scopes():
    provider = CreditSenseOAuthProvider()
    client = _client()
    code = _authorize(provider, client)
    stored = asyncio.run(provider.load_authorization_code(client, code))

    token = asyncio.run(provider.exchange_authorization_code(client, stored))

    assert token.token_type == "Bearer"
    assert token.refresh_token
    access = asyncio.run(provider.load_access_token(token.access_token))
    assert access is not None
    assert access.scopes == [SCOPE_READ, SCOPE_WRITE]
    assert access.client_id == "client-1"


def test_authorization_code_is_single_use():
    provider = CreditSenseOAuthProvider()
    client = _client()
    code = _authorize(provider, client)
    stored = asyncio.run(provider.load_authorization_code(client, code))
    asyncio.run(provider.exchange_authorization_code(client, stored))

    with pytest.raises(TokenError):
        asyncio.run(provider.exchange_authorization_code(client, stored))


def test_refresh_rotates_both_tokens_and_invalidates_the_old_access_token():
    provider = CreditSenseOAuthProvider()
    client = _client()
    code = _authorize(provider, client)
    stored = asyncio.run(provider.load_authorization_code(client, code))
    first = asyncio.run(provider.exchange_authorization_code(client, stored))

    refresh = asyncio.run(provider.load_refresh_token(client, first.refresh_token))
    assert refresh is not None
    second = asyncio.run(provider.exchange_refresh_token(client, refresh, []))

    assert second.access_token != first.access_token
    assert second.refresh_token != first.refresh_token
    assert asyncio.run(provider.load_access_token(first.access_token)) is None
    assert asyncio.run(provider.load_access_token(second.access_token)) is not None


def test_refresh_can_narrow_scopes_but_never_widen_them():
    provider = CreditSenseOAuthProvider()
    client = _client()
    code = _authorize(provider, client)
    stored = asyncio.run(provider.load_authorization_code(client, code))
    first = asyncio.run(provider.exchange_authorization_code(client, stored))
    refresh = asyncio.run(provider.load_refresh_token(client, first.refresh_token))

    narrowed = asyncio.run(provider.exchange_refresh_token(client, refresh, [SCOPE_READ]))
    access = asyncio.run(provider.load_access_token(narrowed.access_token))
    assert access.scopes == [SCOPE_READ]

    widened = asyncio.run(
        provider.exchange_refresh_token(
            client,
            asyncio.run(provider.load_refresh_token(client, narrowed.refresh_token)),
            ["creditsense:admin"],
        )
    )
    access = asyncio.run(provider.load_access_token(widened.access_token))
    assert "creditsense:admin" not in access.scopes


def test_revoking_an_access_token_stops_it_authenticating():
    provider = CreditSenseOAuthProvider()
    client = _client()
    code = _authorize(provider, client)
    stored = asyncio.run(provider.load_authorization_code(client, code))
    issued = asyncio.run(provider.exchange_authorization_code(client, stored))
    access = asyncio.run(provider.load_access_token(issued.access_token))

    asyncio.run(provider.revoke_token(access))

    assert asyncio.run(provider.load_access_token(issued.access_token)) is None


def test_expired_access_token_is_rejected(monkeypatch):
    provider = CreditSenseOAuthProvider()
    client = _client()
    code = _authorize(provider, client)
    stored = asyncio.run(provider.load_authorization_code(client, code))
    issued = asyncio.run(provider.exchange_authorization_code(client, stored))

    later = time.time() + 7200
    monkeypatch.setattr("creditsense.mcp_server.oauth_provider.time.time", lambda: later)

    assert asyncio.run(provider.load_access_token(issued.access_token)) is None


def test_the_two_fixed_api_keys_still_authenticate_unchanged():
    """Backward compatibility: swapping build_server() from token_verifier to this
    provider must not break the bearer-key path test_mcp_auth.py covers."""
    provider = CreditSenseOAuthProvider()

    write = asyncio.run(provider.load_access_token("creditsense-write-demo-key"))
    read = asyncio.run(provider.load_access_token("creditsense-read-demo-key"))

    assert write is not None and SCOPE_WRITE in write.scopes
    assert read is not None and read.scopes == [SCOPE_READ]


def test_unknown_token_is_rejected():
    provider = CreditSenseOAuthProvider()

    assert asyncio.run(provider.load_access_token("not-a-real-token")) is None
