"""build_server() end-to-end, via the SDK's in-process Client -- no subprocess, no
live DB (session_scope and the DB-touching seams in tools.py are patched), matching
the rest of this suite's offline convention.

Plain sync test functions driving asyncio.run() rather than pytest-asyncio -- that
dependency isn't installed anywhere else in this project, and adding a test-only
dependency for three tests isn't worth it.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

from mcp.client import Client

from creditsense.mcp_server.server import build_server


class _FakeScope:
    def __init__(self, session):
        self._session = session

    def __enter__(self):
        return self._session

    def __exit__(self, *exc):
        return False


def _fake_session_scope():
    written = []
    session = SimpleNamespace(add=lambda o: written.append(o), commit=lambda: None)
    return _FakeScope(session), written


def test_all_four_tools_are_registered():
    async def run():
        server = build_server()
        async with Client(server, raise_exceptions=True) as client:
            return await client.list_tools()

    result = asyncio.run(run())

    names = {t.name for t in result.tools}
    assert names == {
        "get_applicant_financials",
        "search_sbp_regulations",
        "get_portfolio_exposure",
        "submit_underwriting_decision",
    }


def test_calling_a_tool_writes_a_generic_audit_row():
    scope, written = _fake_session_scope()

    async def run():
        server = build_server()
        with patch("creditsense.mcp_server.server.session_scope", return_value=scope), \
             patch("creditsense.mcp_server.tools._fetch_applicant", return_value=None):
            async with Client(server, raise_exceptions=True) as client:
                return await client.call_tool(
                    "get_applicant_financials", {"applicant_id": "SME-999999"}
                )

    result = asyncio.run(run())

    body = json.loads(result.content[0].text)
    assert body == {"found": False, "applicant_id": "SME-999999"}

    assert len(written) == 1
    assert written[0].action == "mcp_call:get_applicant_financials"
    assert written[0].status == "success"
    assert written[0].resource_id == "SME-999999"


def test_search_tool_call_reaches_hybrid_search():
    scope, written = _fake_session_scope()

    async def run():
        server = build_server()
        with patch("creditsense.mcp_server.server.session_scope", return_value=scope), \
             patch("creditsense.mcp_server.tools.hybrid_search", return_value=[]) as mock_search:
            async with Client(server, raise_exceptions=True) as client:
                result = await client.call_tool(
                    "search_sbp_regulations", {"topic": "clean facility limit"}
                )
            mock_search.assert_called_once()
            return result

    result = asyncio.run(run())

    # An empty list result produces no content blocks at all (nothing to render as
    # text) -- the meaningful assertion is that hybrid_search was actually reached
    # and the call completed without error, which mock_search.assert_called_once()
    # inside run() already confirmed.
    assert result.content == []
    assert written[0].action == "mcp_call:search_sbp_regulations"
