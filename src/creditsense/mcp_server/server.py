"""MCPServer construction (P6.1) and per-call audit logging (P6.8).

Every tool is registered as a thin wrapper around its `tools.py` function, never bare
-- each wrapper, via `_run_tool`:
  1. Enforces the scope that tool needs (`auth.require_scope` -- a no-op on stdio,
     real over HTTP; see `auth.py`'s docstring for why).
  2. Opens exactly one `session_scope()` for the call. A session's lifetime is one
     call, never longer -- `retrieval.py` documents why a shared Session isn't safe
     across concurrent use.
  3. Calls the corresponding `tools.py` function.
  4. Writes one generic `audit_logs` row for the call itself
     (`action=f"mcp_call:{name}"`), independent of whatever domain-specific row
     `submit_underwriting_decision` writes for the decision it made. Reads are logged
     here too -- P6.8 asks for every tool call, not only writes.

The docstring on each `@server.tool()`-decorated function IS the tool description an
MCP client sees and reasons from when deciding whether to call it -- these are
load-bearing for the demo, not decoration.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Callable

from mcp.server import MCPServer
from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions, RevocationOptions

from creditsense.config import get_settings
from creditsense.db.models import AuditLog
from creditsense.db.session import session_scope
from creditsense.mcp_server import tools
from creditsense.mcp_server.auth import (
    SCOPE_READ,
    SCOPE_WRITE,
    WRITE_TOOLS,
    caller_identity,
    require_scope,
)
from creditsense.mcp_server.oauth_provider import CreditSenseOAuthProvider
from creditsense.ratelimit import RateLimitExceeded, TokenBucketLimiter

logger = logging.getLogger(__name__)

_settings = get_settings()
_write_limiter = TokenBucketLimiter(
    rate_per_minute=_settings.write_rate_limit_per_minute,
    burst=_settings.write_rate_limit_burst,
)


def _hash_args(kwargs: dict[str, Any]) -> str:
    """A hash of the call's arguments for the audit row -- lets an auditor confirm
    two calls used identical inputs without the row growing unbounded for a search
    tool's full query text every time. Full payload_json still carries `outcome` for
    context; see submit_underwriting_decision's own richer audit write for full
    argument transparency on the tool that actually changes something."""
    payload = json.dumps(kwargs, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _summarize_outcome(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return {k: v for k, v in result.items() if k in ("found", "accepted", "decision")}
    if isinstance(result, list):
        return {"count": len(result)}
    return {}


def _run_tool(tool_name: str, fn: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
    require_scope(tool_name)
    if tool_name in WRITE_TOOLS:
        # Only the write tool is limited (P7.4) -- what's actually being protected is
        # the OpenAI bill and the database behind submit_underwriting_decision's
        # re-verification pass, which makes several calls per invocation. The three
        # read tools stay unlimited.
        try:
            _write_limiter.check(caller_identity())
        except RateLimitExceeded as exc:
            raise RuntimeError(
                f"Rate limit exceeded for {tool_name}; retry after "
                f"{exc.retry_after_seconds:.1f}s."
            ) from exc
    with session_scope() as session:
        result = fn(session=session, **kwargs)
        session.add(
            AuditLog(
                actor=caller_identity(),
                action=f"mcp_call:{tool_name}",
                resource_type="mcp_tool",
                resource_id=kwargs.get("applicant_id"),
                payload_json={
                    "args_hash": _hash_args(kwargs),
                    "outcome": _summarize_outcome(result),
                },
                status="success",
            )
        )
        session.commit()
        return result


def build_server() -> MCPServer:
    settings = get_settings()
    # The externally-reachable URL of the mounted MCP app, NOT mcp_host:mcp_port --
    # nothing binds to those standalone (api/main.py mounts this app inside the
    # FastAPI app at /mcp). OAuth discovery compares the issuer by exact string, so
    # this has to be what a client actually typed to get here; set
    # MCP_PUBLIC_BASE_URL to a tunnel's https:// origin when connecting a remote
    # client like Claude Desktop's connector.
    base_url = f"{settings.mcp_public_base_url.rstrip('/')}/mcp"

    server = MCPServer(
        name="creditsense",
        instructions=(
            "Tools for SME credit underwriting: look up an applicant's financials, "
            "search SBP prudential regulations and internal policy, check portfolio "
            "sector exposure, and submit an underwriting decision. "
            "submit_underwriting_decision re-verifies any APPROVE against a live "
            "compliance check and refuses it if it conflicts with an open regulatory "
            "breach -- it cannot be used to bypass that guardrail."
        ),
        # An auth *server* provider rather than a bare token_verifier, so Claude
        # Desktop's chat connector can complete the OAuth discovery + dynamic client
        # registration flow it requires (it has no field to paste a bearer token
        # into). The two fixed bearer keys still authenticate exactly as before --
        # CreditSenseOAuthProvider.load_access_token checks ApiKeyTokenVerifier first.
        # required_scopes=None because enforcement here is per-tool
        # (auth.require_scope), not global.
        auth_server_provider=CreditSenseOAuthProvider(),
        auth=AuthSettings(
            issuer_url=base_url,
            resource_server_url=base_url,
            required_scopes=None,
            client_registration_options=ClientRegistrationOptions(
                enabled=True,
                valid_scopes=[SCOPE_READ, SCOPE_WRITE],
                default_scopes=[SCOPE_READ, SCOPE_WRITE],
            ),
            revocation_options=RevocationOptions(enabled=True),
        ),
    )

    @server.tool()
    def get_applicant_financials(applicant_id: str) -> dict:
        """Look up one applicant's model-input financial fields (18 fields: current
        ratio, debt-to-equity ratio, ECIB score, documentation tier, sector, and
        others) plus their account number and CNIC, both masked. Never returns the
        applicant's own risk score or recommended credit limit -- those are the
        model's own prediction targets, not inputs, and are withheld so no caller can
        read the answer instead of computing it."""
        return _run_tool(
            "get_applicant_financials", tools.get_applicant_financials,
            {"applicant_id": applicant_id},
        )

    @server.tool()
    def search_sbp_regulations(topic: str, regulation_number: str | None = None) -> list[dict]:
        """Search the SBP prudential regulations / internal credit policy / loan
        structuring template corpus by topic, using hybrid keyword + meaning search
        with an LLM reranker. Pass regulation_number (e.g. 'R-5', 'P-7', 'T-2')
        instead of relying on the ranked search to fetch a specific clause directly
        and completely, including any other chunks sharing that same number."""
        return _run_tool(
            "search_sbp_regulations", tools.search_sbp_regulations,
            {"topic": topic, "regulation_number": regulation_number},
        )

    @server.tool()
    def get_portfolio_exposure(sector: str | None = None) -> list[dict]:
        """Get the bank's aggregate exposure by sector, as a percentage of the whole
        SME book -- what internal policy P-1's 25% single-sector concentration limit
        is checked against. Omit sector to get every sector's exposure at once,
        ordered by total exposure."""
        return _run_tool(
            "get_portfolio_exposure", tools.get_portfolio_exposure, {"sector": sector},
        )

    @server.tool()
    def submit_underwriting_decision(
        applicant_id: str,
        decision: str,
        rationale: str,
        requested_amount_pkr: float | None = None,
        is_clean_facility: bool = False,
    ) -> dict:
        """Record an underwriting decision for an applicant: 'APPROVE', 'DECLINE', or
        'ESCALATE_TO_HUMAN', with a plain-English rationale. An APPROVE is
        re-verified against a live compliance check before being accepted -- if it
        conflicts with an open regulatory breach (for example exceeding the R-5
        per-party exposure ceiling), it is REFUSED, not silently recorded, and the
        refused attempt is still logged. requested_amount_pkr is the proposed
        facility size (omit it to check against the model's own recommended limit);
        is_clean_facility marks an unsecured, personal-guarantee-only facility, which
        is checked against the R-9 clean-exposure ceiling."""
        return _run_tool(
            "submit_underwriting_decision", tools.submit_underwriting_decision,
            {
                "applicant_id": applicant_id,
                "decision": decision,
                "rationale": rationale,
                "requested_amount_pkr": requested_amount_pkr,
                "is_clean_facility": is_clean_facility,
                "actor": caller_identity(),
            },
        )

    return server
