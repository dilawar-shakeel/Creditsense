"""Auth for the MCP server (P6.7).

Simplified for this project's actual scope: two fixed bearer tokens from Settings, not
a full OAuth authorization server. This is a real, working `TokenVerifier` -- a
missing or wrong token really is rejected, and `test_mcp_auth.py` proves it -- but it
authenticates by comparing a token against a configured value, not by running an
OAuth code-exchange or dynamic client registration flow. `mcp.server.auth` ships that
heavier machinery (`OAuthAuthorizationServerProvider`, the `/authorize`/`/token`/
`/register` routes); standing up a real authorization server is out of scope for a
project meant to be run and demoed by one person.

Two scopes are enough for four tools: read covers the three lookups, write covers the
one tool that changes anything.

Only meaningful on the HTTP transport. Over stdio there is no network boundary and no
token to present -- the client already has process-level access to run the server at
all, which is a stronger form of access than any token could grant or withhold.
`require_scope` reflects that: with no access token in context (the stdio case), it
does not enforce anything. Do not present the stdio demo as authenticated.
"""

from __future__ import annotations

from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken

from creditsense.config import get_settings

SCOPE_READ = "creditsense:read"
SCOPE_WRITE = "creditsense:write"

READ_TOOLS = frozenset(
    {"get_applicant_financials", "search_sbp_regulations", "get_portfolio_exposure"}
)
WRITE_TOOLS = frozenset({"submit_underwriting_decision"})


class AuthorizationError(PermissionError):
    """Raised by require_scope when an authenticated caller's token lacks the scope a
    tool needs. Never raised for an unauthenticated (stdio) caller -- see module
    docstring."""


class ApiKeyTokenVerifier:
    """Two fixed keys: a read-only key and a read+write key. `mcp.server.auth`
    consumes this structurally (it's a Protocol, not a base class to inherit) --
    providing `verify_token` with this signature is the entire contract."""

    async def verify_token(self, token: str) -> AccessToken | None:
        settings = get_settings()
        if token and token == settings.mcp_write_api_key:
            return AccessToken(
                token=token, client_id="creditsense-write-client",
                scopes=[SCOPE_READ, SCOPE_WRITE],
            )
        if token and token == settings.mcp_read_api_key:
            return AccessToken(
                token=token, client_id="creditsense-read-client", scopes=[SCOPE_READ],
            )
        return None


def tool_scope(tool_name: str) -> str:
    """The scope a tool call requires."""
    return SCOPE_WRITE if tool_name in WRITE_TOOLS else SCOPE_READ


def require_scope(tool_name: str) -> None:
    """Call at the top of every HTTP-transport tool wrapper. Raises
    AuthorizationError if the caller's token doesn't carry the scope this tool needs.
    A no-op when there is no access token in context at all -- that's the stdio case,
    where auth is not meaningful (see module docstring), not a caller to reject.
    """
    token = get_access_token()
    if token is None:
        return
    needed = tool_scope(tool_name)
    if needed not in token.scopes:
        raise AuthorizationError(
            f"Token for client {token.client_id!r} lacks required scope {needed!r} "
            f"for tool {tool_name!r}."
        )


def caller_identity() -> str:
    """The best available identity string for an audit log row. Falls back to a
    fixed label for stdio, where there is no token at all -- never guesses at an
    identity from something a caller supplied unauthenticated."""
    token = get_access_token()
    return token.client_id if token is not None else "stdio-local"
