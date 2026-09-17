"""P7.4: the hand-rolled token bucket (ratelimit.py) plus its two wired-in
chokepoints -- the MCP write tool only (server.py) and the underwrite endpoint
(covered separately in test_underwrite_endpoint.py's dedicated 429 test).
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from mcp.client import Client

from creditsense.mcp_server.server import build_server
from creditsense.ratelimit import RateLimitExceeded, TokenBucketLimiter


def test_burst_allows_up_to_capacity_then_raises():
    limiter = TokenBucketLimiter(rate_per_minute=60, burst=3)

    limiter.check("caller-a")
    limiter.check("caller-a")
    limiter.check("caller-a")

    with pytest.raises(RateLimitExceeded):
        limiter.check("caller-a")


def test_keys_are_independent():
    limiter = TokenBucketLimiter(rate_per_minute=60, burst=1)

    limiter.check("caller-a")
    limiter.check("caller-b")  # separate bucket -- must not be affected by caller-a


def test_tokens_refill_over_time():
    limiter = TokenBucketLimiter(rate_per_minute=60, burst=1)  # 1 token/second
    limiter.check("caller-a")

    with pytest.raises(RateLimitExceeded):
        limiter.check("caller-a")

    time.sleep(1.05)
    limiter.check("caller-a")  # should have refilled by now


def test_retry_after_is_positive_and_reasonable():
    limiter = TokenBucketLimiter(rate_per_minute=60, burst=1)
    limiter.check("caller-a")

    with pytest.raises(RateLimitExceeded) as exc_info:
        limiter.check("caller-a")

    assert 0 < exc_info.value.retry_after_seconds <= 1.0


def _fake_scope(session):
    class _Scope:
        def __enter__(self):
            return session

        def __exit__(self, *exc):
            return False

    return _Scope()


def test_write_tool_trips_the_limit_but_read_tools_do_not():
    session = SimpleNamespace(add=lambda obj: None, commit=lambda: None)
    fresh_limiter = TokenBucketLimiter(rate_per_minute=60, burst=1)

    async def run():
        server = build_server()
        with patch("creditsense.mcp_server.server.session_scope", return_value=_fake_scope(session)), \
             patch("creditsense.mcp_server.server._write_limiter", fresh_limiter), \
             patch("creditsense.mcp_server.tools._fetch_applicant", return_value=None), \
             patch("creditsense.mcp_server.tools.hybrid_search", return_value=[]):
            async with Client(server, raise_exceptions=False) as client:
                # First write call consumes the single token.
                first = await client.call_tool(
                    "submit_underwriting_decision",
                    {"applicant_id": "SME-999999", "decision": "DECLINE", "rationale": "x"},
                )
                # Second write call in the same burst window must be rejected.
                second = await client.call_tool(
                    "submit_underwriting_decision",
                    {"applicant_id": "SME-999999", "decision": "DECLINE", "rationale": "x"},
                )
                # A read tool must be entirely unaffected by the write bucket being empty.
                read_result = await client.call_tool(
                    "get_applicant_financials", {"applicant_id": "SME-999999"}
                )
                return first, second, read_result

    first, second, read_result = asyncio.run(run())

    assert first.is_error is False
    assert second.is_error is True
    assert read_result.is_error is False
