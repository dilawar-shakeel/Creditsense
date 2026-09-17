from types import SimpleNamespace
from unittest.mock import patch

from creditsense.mcp_server.tools import (
    get_applicant_financials,
    get_portfolio_exposure,
    search_sbp_regulations,
)
from creditsense.rag.retrieval import RetrievedChunk


class FakeApplicantRecord(SimpleNamespace):
    """Stand-in for creditsense.db.models.Applicant with the fields tools.py reads."""


def _chunk(chunk_id: str, regulation_number: str, score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, regulation_number=regulation_number,
        clause_title=f"Title {regulation_number}", section_path="Part II",
        content=f"Clause text for {regulation_number}.", source_type="sbp_regulation",
        cross_references=["R-1"], score=score, source="fused",  # type: ignore[arg-type]
    )


# --- get_applicant_financials ---------------------------------------------------------

def test_get_applicant_financials_returns_masked_identifiers_and_allowlisted_fields():
    applicant = FakeApplicantRecord(
        applicant_id="SME-000001", sector="Retail/Trade", documentation_tier="Audited Financials",
        years_in_business=8.0, account_number="PK36SCBL0000001123456702", cnic="35202-1234567-1",
        raw_profile_json={
            "ecib_score": 720.0, "risk_index_score": 55, "recommend_credit_limit_pkr": 9_000_000.0,
        },
    )
    with patch("creditsense.mcp_server.tools._fetch_applicant", return_value=applicant):
        result = get_applicant_financials("SME-000001", session=object())

    assert result["found"] is True
    assert result["account_number"] == "PK36****************6702"
    assert result["cnic"] == "35202-*******-1"
    assert result["financials"]["ecib_score"] == 720.0
    assert "risk_index_score" not in result["financials"]
    assert "recommend_credit_limit_pkr" not in result["financials"]


def test_get_applicant_financials_reports_not_found_cleanly():
    with patch("creditsense.mcp_server.tools._fetch_applicant", return_value=None):
        result = get_applicant_financials("SME-999999", session=object())

    assert result == {"found": False, "applicant_id": "SME-999999"}


# --- search_sbp_regulations ------------------------------------------------------------

def test_search_sbp_regulations_passes_through_hybrid_search_results():
    chunks = [_chunk("c1", "R-5"), _chunk("c2", "R-9")]
    with patch("creditsense.mcp_server.tools.hybrid_search", return_value=chunks) as mock_search:
        result = search_sbp_regulations("exposure limit", session=object())

    mock_search.assert_called_once()
    call_args = mock_search.call_args
    assert call_args.args[1] == "exposure limit"
    assert call_args.kwargs["regulation_number"] is None
    assert len(result) == 2
    assert result[0]["chunk_id"] == "c1"
    assert result[0]["regulation_number"] == "R-5"
    assert result[0]["cross_references"] == ["R-1"]


def test_search_sbp_regulations_forwards_a_regulation_number_filter():
    with patch("creditsense.mcp_server.tools.hybrid_search", return_value=[]) as mock_search:
        search_sbp_regulations("anything", regulation_number="R-5", session=object())

    assert mock_search.call_args.kwargs["regulation_number"] == "R-5"


def test_search_sbp_regulations_returns_an_empty_list_for_no_hits():
    with patch("creditsense.mcp_server.tools.hybrid_search", return_value=[]):
        result = search_sbp_regulations("something out of scope", session=object())
    assert result == []


# --- get_portfolio_exposure -------------------------------------------------------------

def _exposure_row(sector, count, total, pct):
    return SimpleNamespace(
        sector=sector, applicant_count=count, total_exposure_pkr=total, pct_of_book=pct,
    )


def test_get_portfolio_exposure_returns_every_sector_when_none_given():
    rows = [_exposure_row("Retail/Trade", 20, 40_000_000.0, 21.80)]
    with patch("creditsense.mcp_server.tools._fetch_sector_exposure_rows", return_value=rows) as mock_fetch:
        result = get_portfolio_exposure(session=object())

    assert mock_fetch.call_args.args[1] is None
    assert result[0]["sector"] == "Retail/Trade"
    assert result[0]["pct_of_book"] == 21.80


def test_get_portfolio_exposure_filters_by_sector():
    rows = [_exposure_row("Textile/Garments", 15, 30_000_000.0, 18.25)]
    with patch("creditsense.mcp_server.tools._fetch_sector_exposure_rows", return_value=rows) as mock_fetch:
        get_portfolio_exposure(sector="Textile/Garments", session=object())

    assert mock_fetch.call_args.args[1] == "Textile/Garments"


def test_get_portfolio_exposure_handles_no_matching_sector():
    with patch("creditsense.mcp_server.tools._fetch_sector_exposure_rows", return_value=[]):
        result = get_portfolio_exposure(sector="Nonexistent Sector", session=object())
    assert result == []
