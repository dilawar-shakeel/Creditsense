"""The OAuth flow through the REAL mounted routes, end to end: discovery -> dynamic
client registration -> authorize (PKCE) -> token exchange -> an authenticated MCP
call. This is the closest thing to what Claude Desktop's connector actually does that
can run offline, and it exists because two failures here are invisible at the unit
level and both would break that connector completely:

  1. The SDK builds its `.well-known` routes inside the MCP app, which api/main.py
     mounts at /mcp -- so they'd answer only at /mcp/.well-known/..., while RFC
     8414/9728 clients look at <origin>/.well-known/<endpoint>/<issuer path>.
     main.py re-registers them at the origin root; these tests hold that in place.
  2. The SDK auto-enables DNS-rebinding protection allowing ONLY localhost when
     `host` is left at its default, so a request arriving with any other Host header
     (i.e. every request through a tunnel) is rejected with 421 before reaching auth.
     main.py allow-lists the configured public host instead of disabling the check.

`base_url` is set to the configured public base URL so the Host header matches that
allow-list -- the same thing a real client does, rather than TestClient's default
"testserver".
"""

from __future__ import annotations

import base64
import hashlib
import secrets
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

from creditsense.api.main import app

REDIRECT_URI = "https://claude.ai/api/mcp/auth_callback"
BASE_URL = "http://localhost:8000"
MCP_HEADERS = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
INITIALIZE = {
    "jsonrpc": "2.0", "id": 1, "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18", "capabilities": {},
        "clientInfo": {"name": "test-client", "version": "1"},
    },
}


@pytest.fixture(scope="module")
def client():
    # The `with` block runs the app lifespan, which starts the MCP session manager --
    # without it every /mcp call fails with "Task group is not initialized". Module
    # scoped deliberately: api/main.py builds ONE session manager at import, and its
    # run() raises if entered twice in a process, so the lifespan can be entered
    # exactly once per test module, not once per test.
    with TestClient(app, base_url=BASE_URL) as test_client:
        yield test_client


def _register(client) -> str:
    response = client.post("/mcp/register", json={
        "client_name": "Claude Desktop (test)",
        "redirect_uris": [REDIRECT_URI],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
    })
    assert response.status_code == 201, response.text
    return response.json()["client_id"]


def _pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()
    ).decode().rstrip("=")
    return verifier, challenge


def _complete_flow(client) -> dict:
    """Registration through token exchange -- the whole connector handshake."""
    client_id = _register(client)
    verifier, challenge = _pkce_pair()

    authorize = client.get("/mcp/authorize", params={
        "response_type": "code", "client_id": client_id, "redirect_uri": REDIRECT_URI,
        "code_challenge": challenge, "code_challenge_method": "S256",
        "state": "state-123", "resource": f"{BASE_URL}/mcp",
    }, follow_redirects=False)
    assert authorize.status_code == 302, authorize.text
    query = parse_qs(urlparse(authorize.headers["location"]).query)
    assert query["state"] == ["state-123"]

    token = client.post("/mcp/token", data={
        "grant_type": "authorization_code", "code": query["code"][0],
        "redirect_uri": REDIRECT_URI, "client_id": client_id,
        "code_verifier": verifier, "resource": f"{BASE_URL}/mcp",
    })
    assert token.status_code == 200, token.text
    return token.json()


@pytest.mark.parametrize("path", [
    "/.well-known/oauth-authorization-server",
    "/.well-known/oauth-authorization-server/mcp",
])
def test_authorization_server_metadata_is_served_from_the_origin_root(client, path):
    response = client.get(path)

    assert response.status_code == 200, response.text
    body = response.json()
    # Exact-string issuer match is what RFC 8414/9207 require and the most common way
    # this silently breaks -- it must be the mounted MCP path, not the bare origin.
    assert body["issuer"] == f"{BASE_URL}/mcp"
    assert body["authorization_endpoint"] == f"{BASE_URL}/mcp/authorize"
    assert body["token_endpoint"] == f"{BASE_URL}/mcp/token"
    assert body["registration_endpoint"] == f"{BASE_URL}/mcp/register"


def test_protected_resource_metadata_is_served_from_the_origin_root(client):
    response = client.get("/.well-known/oauth-protected-resource/mcp")

    assert response.status_code == 200, response.text
    assert response.json()["resource"] == f"{BASE_URL}/mcp"


def test_full_registration_authorize_token_flow_grants_both_scopes(client):
    token = _complete_flow(client)

    assert token["token_type"] == "Bearer"
    assert token["refresh_token"]
    assert set(token["scope"].split()) == {"creditsense:read", "creditsense:write"}


def test_an_issued_token_authenticates_a_real_mcp_call(client):
    token = _complete_flow(client)

    response = client.post("/mcp/", json=INITIALIZE, headers={
        **MCP_HEADERS, "Authorization": f"Bearer {token['access_token']}",
    })

    assert response.status_code == 200
    assert "creditsense" in response.text


def test_the_existing_fixed_api_key_still_authenticates(client):
    """The bearer-key path must survive swapping build_server() over to the OAuth
    provider -- this is the regression test for that swap."""
    response = client.post("/mcp/", json=INITIALIZE, headers={
        **MCP_HEADERS, "Authorization": "Bearer creditsense-write-demo-key",
    })

    assert response.status_code == 200


def test_missing_and_invalid_tokens_are_rejected(client):
    assert client.post("/mcp/", json=INITIALIZE, headers=MCP_HEADERS).status_code == 401
    assert client.post("/mcp/", json=INITIALIZE, headers={
        **MCP_HEADERS, "Authorization": "Bearer nonsense",
    }).status_code == 401


def test_a_host_outside_the_allow_list_is_rejected_before_auth(client):
    """DNS-rebinding protection stays ON -- the tunnel host is allow-listed, not the
    check disabled. An unknown Host must still be refused (421), even with a valid key.
    Sends the rogue Host header on the existing client rather than building a second
    one, since the app's lifespan can only be entered once (see the fixture)."""
    response = client.post("/mcp/", json=INITIALIZE, headers={
        **MCP_HEADERS,
        "Host": "evil.example.com",
        "Authorization": "Bearer creditsense-write-demo-key",
    })

    assert response.status_code == 421
