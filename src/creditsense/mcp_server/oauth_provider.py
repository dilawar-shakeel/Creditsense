"""A minimal, auto-approving OAuth authorization server for the MCP HTTP transport.

Why this exists: `auth.py`'s `ApiKeyTokenVerifier` authenticates a caller that already
has one of two fixed bearer tokens, which is all the `mcp` CLI, the test suite, and any
scripted client ever need. Claude Desktop's chat connector is different -- it performs
real OAuth discovery and Dynamic Client Registration against whatever URL it's given,
and has no field to paste a bearer token into. A `TokenVerifier` has nothing to answer
that with, so this module adds the authorization-server half.

It is deliberately minimal and deliberately honest about it:

  - **Auto-approve, no login screen.** `authorize()` issues a code immediately rather
    than authenticating a human. There is no user directory in this project to
    authenticate against -- inventing a fake login page would be theatre, not security.
  - **In-memory only.** Clients, codes, and tokens live in this process and vanish when
    it restarts (the connector then re-authenticates, which is quick). Same category of
    limitation as ratelimit.py's per-process buckets.
  - **What this does and does not protect.** It satisfies the OAuth handshake Claude's
    connector requires; it does NOT establish who the caller is. Anyone who can reach
    the server's URL can complete this flow. That is acceptable for a single-person
    local demo behind an ad-hoc tunnel, and is NOT a substitute for real authentication
    if this were ever exposed properly.

The fixed-bearer-token path is unchanged and still works: `load_access_token` tries
`ApiKeyTokenVerifier` first and only then its own issued-token store, so both
mechanisms authenticate through one provider. Everything downstream (`auth.require_scope`,
`auth.caller_identity`, `server._run_tool`) consumes the same `AccessToken` either way
and needed no changes at all.
"""

from __future__ import annotations

import secrets
import time

from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    OAuthAuthorizationServerProvider,
    OAuthClientInformationFull,
    OAuthToken,
    RefreshToken,
    TokenError,
    construct_redirect_uri,
)

from creditsense.mcp_server.auth import SCOPE_READ, SCOPE_WRITE, ApiKeyTokenVerifier

AUTHORIZATION_CODE_TTL_SECONDS = 300
ACCESS_TOKEN_TTL_SECONDS = 3600
# Full read+write for any client that completes the flow -- a deliberate choice for
# this demo, not an oversight. A connector that can look regulations up but can't
# record a decision would not exercise the guardrail that is the whole point of
# submit_underwriting_decision.
GRANTED_SCOPES = [SCOPE_READ, SCOPE_WRITE]


def _new_secret() -> str:
    """32 url-safe bytes -- comfortably above RFC 6749 section 10.10's 128-bit floor
    for authorization codes, and used for tokens as well."""
    return secrets.token_urlsafe(32)


class CreditSenseOAuthProvider(
    OAuthAuthorizationServerProvider[AuthorizationCode, RefreshToken, AccessToken]
):
    def __init__(self) -> None:
        self._clients: dict[str, OAuthClientInformationFull] = {}
        self._authorization_codes: dict[str, AuthorizationCode] = {}
        self._access_tokens: dict[str, AccessToken] = {}
        self._refresh_tokens: dict[str, RefreshToken] = {}
        self._api_key_verifier = ApiKeyTokenVerifier()

    # --- Dynamic client registration -------------------------------------------------

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        return self._clients.get(client_id)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        self._clients[client_info.client_id] = client_info

    # --- Authorization ---------------------------------------------------------------

    async def authorize(
        self, client: OAuthClientInformationFull, params: AuthorizationParams
    ) -> str:
        """Auto-approves and redirects straight back with a fresh code. `code_challenge`
        is persisted verbatim; PKCE verification itself happens in the SDK's own token
        handler against what this stores, so getting it onto the AuthorizationCode is
        this method's whole responsibility there."""
        code = _new_secret()
        self._authorization_codes[code] = AuthorizationCode(
            code=code,
            scopes=list(GRANTED_SCOPES),
            expires_at=time.time() + AUTHORIZATION_CODE_TTL_SECONDS,
            client_id=client.client_id,
            code_challenge=params.code_challenge,
            redirect_uri=params.redirect_uri,
            redirect_uri_provided_explicitly=params.redirect_uri_provided_explicitly,
            resource=params.resource,
            subject="creditsense-demo-user",
        )
        return construct_redirect_uri(str(params.redirect_uri), code=code, state=params.state)

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> AuthorizationCode | None:
        code = self._authorization_codes.get(authorization_code)
        if code is None or code.client_id != client.client_id:
            return None
        if code.expires_at < time.time():
            del self._authorization_codes[authorization_code]
            return None
        return code

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode
    ) -> OAuthToken:
        # Single use: a replayed code must not mint a second token.
        if self._authorization_codes.pop(authorization_code.code, None) is None:
            raise TokenError(
                error="invalid_grant",
                error_description="Authorization code is unknown or already used",
            )
        return self._issue_tokens(
            client_id=client.client_id,
            scopes=authorization_code.scopes,
            resource=authorization_code.resource,
            subject=authorization_code.subject,
        )

    # --- Refresh ---------------------------------------------------------------------

    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> RefreshToken | None:
        token = self._refresh_tokens.get(refresh_token)
        if token is None or token.client_id != client.client_id:
            return None
        return token

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        """Rotates both tokens, per the protocol's own SHOULD. Requested `scopes` may
        only narrow what the refresh token already carries -- never widen it."""
        self._refresh_tokens.pop(refresh_token.token, None)
        for issued_token, access in list(self._access_tokens.items()):
            if access.client_id == refresh_token.client_id:
                del self._access_tokens[issued_token]

        granted = [s for s in scopes if s in refresh_token.scopes] if scopes else list(refresh_token.scopes)
        return self._issue_tokens(
            client_id=client.client_id,
            scopes=granted,
            resource=None,
            subject=refresh_token.subject,
        )

    # --- Token verification / revocation ---------------------------------------------

    async def load_access_token(self, token: str) -> AccessToken | None:
        """The verification path for every authenticated HTTP tool call. Checks the two
        fixed bearer keys first (unchanged behaviour, still covered by
        test_mcp_auth.py), then this provider's own issued tokens."""
        from_api_key = await self._api_key_verifier.verify_token(token)
        if from_api_key is not None:
            return from_api_key

        access = self._access_tokens.get(token)
        if access is None:
            return None
        if access.expires_at is not None and access.expires_at < time.time():
            del self._access_tokens[token]
            return None
        return access

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        self._access_tokens.pop(token.token, None)
        self._refresh_tokens.pop(token.token, None)

    # --- Internals -------------------------------------------------------------------

    def _issue_tokens(
        self, *, client_id: str, scopes: list[str], resource: str | None, subject: str | None
    ) -> OAuthToken:
        access_token = _new_secret()
        refresh_token = _new_secret()

        self._access_tokens[access_token] = AccessToken(
            token=access_token,
            client_id=client_id,
            scopes=list(scopes),
            expires_at=int(time.time() + ACCESS_TOKEN_TTL_SECONDS),
            resource=resource,
            subject=subject,
        )
        self._refresh_tokens[refresh_token] = RefreshToken(
            token=refresh_token,
            client_id=client_id,
            scopes=list(scopes),
            expires_at=None,
            subject=subject,
        )
        return OAuthToken(
            access_token=access_token,
            token_type="Bearer",
            expires_in=ACCESS_TOKEN_TTL_SECONDS,
            scope=" ".join(scopes),
            refresh_token=refresh_token,
        )
