"""P7.2/P7.3 additions to agents/pipeline.py: a stage that raises still leaves an
audit record (written via a fresh session_scope, since the caller's session may no
longer be safe to commit to), and the exception still propagates to the caller.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from creditsense.agents.pipeline import run_underwriting
from creditsense.agents.schemas import (
    ComplianceReport,
    LoanApplication,
    ParsedFinancials,
    RiskAssessment,
    UnderwritingDecision,
)


def _application() -> LoanApplication:
    return LoanApplication(
        applicant_id="SME-000001",
        raw_fields={"sector_risk_code": "Retail/Trade"},
        requested_amount_pkr=1_000_000.0,
    )


def _caller_session():
    return SimpleNamespace(add=lambda obj: None, commit=lambda: None)


class _FakeFailureScope:
    def __init__(self, written: list):
        self._written = written

    def __enter__(self):
        return SimpleNamespace(add=lambda obj: self._written.append(obj), commit=lambda: None)

    def __exit__(self, *exc):
        return False


def test_a_raising_stage_still_writes_a_failure_audit_row_and_reraises():
    failure_rows: list = []

    def raise_in_analyze(application, session):
        raise RuntimeError("simulated DB outage during analyze")

    with patch("creditsense.agents.financial_analyst.analyze", side_effect=raise_in_analyze), \
         patch(
             "creditsense.agents.pipeline.session_scope",
             return_value=_FakeFailureScope(failure_rows),
         ):
        with pytest.raises(RuntimeError, match="simulated DB outage"):
            run_underwriting(_application(), _caller_session())

    assert len(failure_rows) == 1
    assert failure_rows[0].status == "failed"
    assert failure_rows[0].resource_id == "SME-000001"
    assert failure_rows[0].payload_json["error_type"] == "RuntimeError"


def test_a_failure_while_recording_the_failure_does_not_mask_the_original_exception():
    def raise_in_analyze(application, session):
        raise RuntimeError("original failure")

    def raise_in_session_scope():
        raise ConnectionError("audit DB also unreachable")

    with patch("creditsense.agents.financial_analyst.analyze", side_effect=raise_in_analyze), \
         patch("creditsense.agents.pipeline.session_scope", side_effect=raise_in_session_scope):
        with pytest.raises(RuntimeError, match="original failure"):
            run_underwriting(_application(), _caller_session())


def _parsed() -> ParsedFinancials:
    return ParsedFinancials(
        applicant_id="SME-000001",
        fields={"sector_risk_code": "Retail/Trade"},
        field_sources={"sector_risk_code": "document"},
    )


def _risk() -> RiskAssessment:
    return RiskAssessment(
        default_probability=0.05, decision_cutoff=0.17, model_decision="REFER_FOR_LIMIT",
        recommended_credit_limit_pkr=1_000_000.0, shap_explanation_raw="n/a", narrative="Low risk.",
    )


def _compliance_report() -> ComplianceReport:
    return ComplianceReport(flags=[], checks_performed=["R-5"], insufficient_data=False)


def _final_decision() -> UnderwritingDecision:
    return UnderwritingDecision(
        applicant_id="SME-000001", decision="APPROVE",
        approved_amount_pkr=1_000_000.0, rationale="Clean approval.",
    )


def test_on_stage_callback_fires_once_per_stage_in_order():
    """P9.1: api/stream.py's SSE trace consumes this callback -- it must fire with the
    same stage name and fields as the existing structlog event, once per stage, in
    pipeline order, and must not change behavior when left as the default None (the
    other tests in this file call run_underwriting without it at all)."""
    events: list[tuple[str, dict]] = []

    with patch("creditsense.agents.financial_analyst.analyze", return_value=_parsed()), \
         patch("creditsense.agents.risk_scoring.score", return_value=_risk()), \
         patch("creditsense.agents.compliance.check", return_value=_compliance_report()), \
         patch("creditsense.agents.supervisor.decide", return_value=_final_decision()):
        decision = run_underwriting(
            _application(), _caller_session(), on_stage=lambda stage, fields: events.append((stage, dict(fields)))
        )

    assert [e[0] for e in events] == ["analyze", "score", "check", "decide"]
    assert events[0][1] == {"unresolved_fields": 0}
    assert events[1][1] == {"default_probability": 0.05}
    assert events[2][1] == {"breach_count": 0, "advisory_count": 0}
    assert events[3][1] == {"decision": "APPROVE"}
    # The callback fields match what actually got logged/returned -- e.g. the decision
    # in the "decide" event is the real outcome, not a placeholder.
    assert decision.decision == "APPROVE"


def test_parsed_financials_are_attached_to_the_final_decision():
    """Frontend gap 1 (FRONTEND_REQUIREMENTS.md §2.2): the analyst's ParsedFinancials
    must survive onto the returned UnderwritingDecision, not just be consumed and
    discarded by risk_scoring."""
    parsed = _parsed()

    with patch("creditsense.agents.financial_analyst.analyze", return_value=parsed), \
         patch("creditsense.agents.risk_scoring.score", return_value=_risk()), \
         patch("creditsense.agents.compliance.check", return_value=_compliance_report()), \
         patch("creditsense.agents.supervisor.decide", return_value=_final_decision()):
        decision = run_underwriting(_application(), _caller_session())

    assert decision.parsed is not None
    assert decision.parsed.fields == parsed.fields
