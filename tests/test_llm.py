"""P8.1 gap: agents/llm.py had zero direct tests -- it appeared in the suite only as a
patch target. It is the single place every agent talks to the model, and its contract
("degrade, never raise") is what turns an LLM outage into an escalation rather than a
stack trace, so it deserves to be proven rather than assumed.
"""

from __future__ import annotations

from unittest.mock import patch

from pydantic import BaseModel

from creditsense.agents.llm import complete_structured


class _Answer(BaseModel):
    text: str


class _Other(BaseModel):
    value: int = 0


class _FakeStructuredClient:
    def __init__(self, result=None, raises: Exception | None = None):
        self._result = result
        self._raises = raises
        self.invoked_with = None

    def invoke(self, messages):
        self.invoked_with = messages
        if self._raises is not None:
            raise self._raises
        return self._result


class _FakeClient:
    def __init__(self, structured_client):
        self._structured_client = structured_client
        self.schema_requested = None

    def with_structured_output(self, schema):
        self.schema_requested = schema
        return self._structured_client


def test_returns_the_parsed_result_on_success():
    client = _FakeClient(_FakeStructuredClient(result=_Answer(text="ok")))

    with patch("creditsense.agents.llm._get_client", return_value=client):
        result = complete_structured("system prompt", "user prompt", _Answer)

    assert result is not None
    assert result.text == "ok"
    assert client.schema_requested is _Answer


def test_sends_the_system_and_user_prompts_in_order():
    structured = _FakeStructuredClient(result=_Answer(text="ok"))

    with patch("creditsense.agents.llm._get_client", return_value=_FakeClient(structured)):
        complete_structured("SYSTEM", "USER", _Answer)

    assert [m.content for m in structured.invoked_with] == ["SYSTEM", "USER"]


def test_an_api_failure_returns_the_fallback_and_never_raises():
    """The contract the whole agents layer depends on."""
    client = _FakeClient(_FakeStructuredClient(raises=RuntimeError("API is down")))
    fallback = _Answer(text="fallback used")

    with patch("creditsense.agents.llm._get_client", return_value=client):
        result = complete_structured("s", "u", _Answer, fallback=fallback)

    assert result is fallback


def test_a_failure_with_no_fallback_returns_none():
    client = _FakeClient(_FakeStructuredClient(raises=RuntimeError("API is down")))

    with patch("creditsense.agents.llm._get_client", return_value=client):
        result = complete_structured("s", "u", _Answer)

    assert result is None


def test_a_wrong_schema_type_is_treated_as_a_failure():
    """A model that returns something of the wrong shape must be rejected, not passed
    through -- otherwise a caller gets an object whose fields it cannot trust."""
    client = _FakeClient(_FakeStructuredClient(result=_Other(value=3)))

    with patch("creditsense.agents.llm._get_client", return_value=client):
        result = complete_structured("s", "u", _Answer)

    assert result is None


def test_a_none_result_is_treated_as_a_failure():
    client = _FakeClient(_FakeStructuredClient(result=None))

    with patch("creditsense.agents.llm._get_client", return_value=client):
        result = complete_structured("s", "u", _Answer, fallback=_Answer(text="fb"))

    assert result.text == "fb"


def test_a_client_construction_failure_also_degrades():
    """Even a missing API key (which blows up inside _get_client) must not raise."""
    with patch("creditsense.agents.llm._get_client", side_effect=RuntimeError("no api key")):
        result = complete_structured("s", "u", _Answer)

    assert result is None
