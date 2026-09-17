from types import SimpleNamespace
from unittest.mock import patch

from creditsense.agents.compliance import check
from creditsense.agents.schemas import LoanApplication, ParsedFinancials, RiskAssessment
from creditsense.rag.retrieval import RetrievedChunk


def _chunk(chunk_id: str, regulation_number: str, score: float, source: str = "reranked") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, regulation_number=regulation_number,
        clause_title=f"Title for {regulation_number}", section_path="Part II",
        content=f"Clause text for {regulation_number}.", source_type="sbp_regulation",
        cross_references=[], score=score, source=source,  # type: ignore[arg-type]
    )


def _clean_parsed() -> ParsedFinancials:
    # Chosen so no deterministic rule fires -- isolates the advisory pass in tests
    # that don't care about it.
    fields = {
        "sbp_enterprise_tier": "SE", "existing_loan_exposure_pkr": 0.0,
        "group_associate_exposure_pkr": 0.0, "documentation_tier": "Audited Financials",
        "sector_risk_code": "Retail/Trade",
    }
    return ParsedFinancials(
        applicant_id="SME-000001", fields=fields,
        field_sources={k: "document" for k in fields}, unresolved_fields=[],
    )


def _risk() -> RiskAssessment:
    return RiskAssessment(
        default_probability=0.05, decision_cutoff=0.17, model_decision="REFER_FOR_LIMIT",
        recommended_credit_limit_pkr=2_000_000.0, shap_explanation_raw="x", narrative="x",
    )


def _application(**overrides) -> LoanApplication:
    return LoanApplication(applicant_id="SME-000001", raw_fields={}, **overrides)


def test_deterministic_breach_carries_a_real_citation_from_direct_lookup():
    application = _application(requested_amount_pkr=200_000_000.0)  # breaches R-5 for SE
    citing_chunk = _chunk("chunk-r5", "R-5", score=1.0, source="bm25")

    with patch("creditsense.agents.compliance._sector_pct_of_book", return_value=None), \
         patch("creditsense.agents.compliance.lookup_by_regulation_number", return_value=[citing_chunk]), \
         patch("creditsense.agents.compliance.hybrid_search", return_value=[]):
        report = check(_clean_parsed(), _risk(), application, session=object())

    # 200,000,000 also crosses the P-2 approval-authority threshold, so more than one
    # deterministic flag legitimately fires here -- this test only asserts about R-5.
    r5_flags = [f for f in report.flags if f.origin == "deterministic" and f.rule_id == "R-5"]
    assert len(r5_flags) == 1
    assert r5_flags[0].citations[0].chunk_id == "chunk-r5"


def test_deterministic_flag_is_dropped_if_no_chunk_can_be_found_to_cite():
    application = _application(requested_amount_pkr=200_000_000.0)

    with patch("creditsense.agents.compliance._sector_pct_of_book", return_value=None), \
         patch("creditsense.agents.compliance.lookup_by_regulation_number", return_value=[]), \
         patch("creditsense.agents.compliance.hybrid_search", return_value=[]):
        report = check(_clean_parsed(), _risk(), application, session=object())

    assert not any(f.rule_id == "R-5" for f in report.flags)


def test_advisory_flag_citing_a_retrieved_chunk_is_kept():
    application = _application(requested_amount_pkr=1_000_000.0)  # nothing deterministic fires
    retrieved = [_chunk("chunk-1", "P-21", score=8.0)]
    finding = SimpleNamespace(rule_id="P-21", summary="Sector overlay applies.", chunk_id="chunk-1")
    llm_result = SimpleNamespace(findings=[finding])

    with patch("creditsense.agents.compliance._sector_pct_of_book", return_value=None), \
         patch("creditsense.agents.compliance.hybrid_search", return_value=retrieved), \
         patch("creditsense.agents.compliance.complete_structured", return_value=llm_result):
        report = check(_clean_parsed(), _risk(), application, session=object())

    retrieved_flags = [f for f in report.flags if f.origin == "retrieved"]
    assert len(retrieved_flags) == 1
    assert retrieved_flags[0].citations[0].chunk_id == "chunk-1"
    assert report.insufficient_data is False


def test_advisory_flag_citing_an_unretrieved_chunk_is_dropped():
    application = _application(requested_amount_pkr=1_000_000.0)
    retrieved = [_chunk("chunk-1", "P-21", score=8.0)]
    # The LLM cites a chunk_id that was never in the retrieved set -- must be discarded.
    finding = SimpleNamespace(rule_id="P-99", summary="Made up.", chunk_id="chunk-does-not-exist")
    llm_result = SimpleNamespace(findings=[finding])

    with patch("creditsense.agents.compliance._sector_pct_of_book", return_value=None), \
         patch("creditsense.agents.compliance.hybrid_search", return_value=retrieved), \
         patch("creditsense.agents.compliance.complete_structured", return_value=llm_result):
        report = check(_clean_parsed(), _risk(), application, session=object())

    assert not any(f.origin == "retrieved" for f in report.flags)
    assert any("not in the retrieved candidate set" in r for r in report.escalation_reasons)


def test_degraded_llm_call_marks_insufficient_data():
    application = _application(requested_amount_pkr=1_000_000.0)
    retrieved = [_chunk("chunk-1", "P-21", score=8.0)]

    with patch("creditsense.agents.compliance._sector_pct_of_book", return_value=None), \
         patch("creditsense.agents.compliance.hybrid_search", return_value=retrieved), \
         patch("creditsense.agents.compliance.complete_structured", return_value=None):
        report = check(_clean_parsed(), _risk(), application, session=object())

    assert report.insufficient_data is True


def test_low_confidence_retrieval_marks_insufficient_data_without_calling_the_llm():
    application = _application(requested_amount_pkr=1_000_000.0)
    # Scores below Settings.min_rerank_score_for_citation (default 5.0).
    retrieved = [_chunk("chunk-1", "P-21", score=1.0)]

    with patch("creditsense.agents.compliance._sector_pct_of_book", return_value=None), \
         patch("creditsense.agents.compliance.hybrid_search", return_value=retrieved), \
         patch("creditsense.agents.compliance.complete_structured") as mock_complete:
        report = check(_clean_parsed(), _risk(), application, session=object())

    assert report.insufficient_data is True
    mock_complete.assert_not_called()


def test_no_amount_to_check_against_is_insufficient_data():
    application = _application(requested_amount_pkr=None)
    risk = RiskAssessment(
        default_probability=0.05, decision_cutoff=0.17, model_decision="DECLINE",
        recommended_credit_limit_pkr=None, shap_explanation_raw="x", narrative="x",
    )
    report = check(_clean_parsed(), risk, application, session=object())
    assert report.insufficient_data is True
    assert report.flags == []


def test_a_db_error_degrades_to_insufficient_data_instead_of_raising():
    """P7.2: a DB/retrieval failure inside check() must never look like a clean
    result with zero flags -- supervisor.decide() only ever escalates (never
    approves) when insufficient_data is True, so degrading here is safe."""
    application = _application(requested_amount_pkr=1_000_000.0)

    with patch(
        "creditsense.agents.compliance._sector_pct_of_book",
        side_effect=RuntimeError("database connection lost"),
    ):
        report = check(_clean_parsed(), _risk(), application, session=object())

    assert report.insufficient_data is True
    assert report.flags == []
    assert report.escalation_reasons == ["Compliance check failed due to an internal error."]
