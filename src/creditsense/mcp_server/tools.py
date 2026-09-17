"""The four MCP tools (P6.2-P6.5), as plain Python functions independent of the MCP
SDK. `server.py` wraps each of these with `@mcp.tool()` and opens a session per call
(`session_scope()`) -- a `Session` cannot itself be an MCP tool argument (it isn't
JSON-serializable, and the SDK builds a tool's schema from its type hints), so every
function here takes `session` as an explicit, keyword-only parameter rather than
opening its own. This also matches the rest of this codebase's convention
(`agents/financial_analyst.analyze(application, session)`,
`agents/compliance.check(parsed, risk, application, session)`) and keeps these
functions testable in-process with a fake session, with no MCP server involved.

Each tool has its own small private DB helper (`_fetch_applicant`,
`_fetch_sector_exposure_rows`, `_write_audit_log`) rather than one shared generic
query layer -- this is the established seam in this repo for DB-touching code
(`agents/financial_analyst.py`'s `_fetch_applicant` docstring explains why: tests
patch the seam function directly instead of faking SQLAlchemy's
`execute(select(...))` machinery).
"""

from __future__ import annotations

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from creditsense.agents import compliance, financial_analyst, risk_scoring
from creditsense.agents.schemas import LoanApplication
from creditsense.db.models import Applicant, AuditLog
from creditsense.mcp_server.masking import allowed_profile_fields, mask_account_number, mask_cnic
from creditsense.rag.retrieval import hybrid_search

_VALID_DECISIONS = frozenset({"APPROVE", "DECLINE", "ESCALATE_TO_HUMAN"})

# The 6 fields a document is expected to carry -- see financial_analyst.py's
# DOCUMENT_FIELDS. submit_underwriting_decision has no uploaded document to parse (the
# caller only ever supplies applicant_id/decision/rationale), so it builds an
# equivalent "document" straight from the applicant's own stored profile -- the same
# fallback worker.py uses for any applicant outside the messy-fixture cohort.
_DOCUMENT_FIELD_NAMES = (
    "current_ratio", "debt_to_equity_ratio", "existing_loan_exposure_pkr",
    "collateral_coverage_ratio", "annual_bank_turnover_pkr", "years_in_business",
    "sector_risk_code",
)

_SECTOR_EXPOSURE_ALL_SQL = text(
    "SELECT sector, applicant_count, total_exposure_pkr, pct_of_book "
    "FROM portfolio_sector_exposure ORDER BY total_exposure_pkr DESC"
)
_SECTOR_EXPOSURE_ONE_SQL = text(
    "SELECT sector, applicant_count, total_exposure_pkr, pct_of_book "
    "FROM portfolio_sector_exposure WHERE sector = :sector"
)


def _fetch_applicant(session: Session, applicant_id: str) -> Applicant | None:
    return session.execute(
        select(Applicant).where(Applicant.applicant_id == applicant_id)
    ).scalar_one_or_none()


def _fetch_sector_exposure_rows(session: Session, sector: str | None):
    if sector:
        return session.execute(_SECTOR_EXPOSURE_ONE_SQL, {"sector": sector}).all()
    return session.execute(_SECTOR_EXPOSURE_ALL_SQL).all()


def _write_audit_log(
    session: Session, *, actor: str, action: str, resource_id: str, status: str, payload: dict,
) -> None:
    session.add(
        AuditLog(
            actor=actor, action=action, resource_type="applicant",
            resource_id=resource_id, payload_json=payload, status=status,
        )
    )
    session.commit()


def get_applicant_financials(applicant_id: str, *, session: Session) -> dict:
    """Scoped read (P6.2): the 18 model-input features for one applicant, plus their
    masked account number and CNIC. Never the raw generator row -- see masking.py for
    why that would leak the model's own answer key."""
    applicant = _fetch_applicant(session, applicant_id)
    if applicant is None:
        return {"found": False, "applicant_id": applicant_id}

    return {
        "found": True,
        "applicant_id": applicant.applicant_id,
        "sector": applicant.sector,
        "documentation_tier": applicant.documentation_tier,
        "years_in_business": applicant.years_in_business,
        "account_number": mask_account_number(applicant.account_number),
        "cnic": mask_cnic(applicant.cnic),
        "financials": allowed_profile_fields(applicant.raw_profile_json),
    }


def search_sbp_regulations(
    topic: str, regulation_number: str | None = None, *, session: Session, top_k: int = 5,
) -> list[dict]:
    """Wraps hybrid RAG retrieval (P6.3). `regulation_number` short-circuits to a
    direct lookup of that clause and every chunk sharing its number -- `topic` is
    ignored in that case, matching `hybrid_search`'s own dispatch (retrieval.py:199)."""
    chunks = hybrid_search(session, topic, regulation_number=regulation_number, top_k=top_k)
    return [
        {
            "chunk_id": c.chunk_id,
            "regulation_number": c.regulation_number,
            "clause_title": c.clause_title,
            "section_path": c.section_path,
            "content": c.content,
            "source_type": c.source_type,
            "cross_references": c.cross_references,
            "score": c.score,
            "source": c.source,
        }
        for c in chunks
    ]


def get_portfolio_exposure(sector: str | None = None, *, session: Session) -> list[dict]:
    """Sector exposure aggregation (P6.4) from the portfolio_sector_exposure view
    (P2.4) -- what a ComplianceAgent checks the P-1 25% concentration limit against.
    `sector=None` returns the whole view, one row per sector."""
    rows = _fetch_sector_exposure_rows(session, sector)
    return [
        {
            "sector": r.sector,
            "applicant_count": r.applicant_count,
            "total_exposure_pkr": float(r.total_exposure_pkr),
            "pct_of_book": float(r.pct_of_book),
        }
        for r in rows
    ]


def _build_application_from_record(
    applicant: Applicant, *, requested_amount_pkr: float | None, is_clean_facility: bool,
) -> LoanApplication:
    profile = applicant.raw_profile_json or {}
    raw_fields = {name: profile.get(name) for name in _DOCUMENT_FIELD_NAMES}
    return LoanApplication(
        applicant_id=applicant.applicant_id,
        raw_fields=raw_fields,
        requested_amount_pkr=requested_amount_pkr,
        is_clean_facility=is_clean_facility,
    )


def submit_underwriting_decision(
    applicant_id: str,
    decision: str,
    rationale: str,
    requested_amount_pkr: float | None = None,
    is_clean_facility: bool = False,
    *,
    session: Session,
    actor: str = "mcp-caller",
) -> dict:
    """Write-scoped, audited (P6.5). This is the tool that could otherwise bypass
    `supervisor.decide()` -- the hard guardrail this project is pitched on -- by
    simply recording whatever a caller says. It cannot: an `APPROVE` is re-verified
    against a real, freshly-run compliance check before being accepted, and REFUSED if
    it conflicts with an open breach. The refused attempt is still written to
    audit_logs as evidence rather than vanishing -- there is no door into this system,
    API, CLI, or MCP, through which an approval can be recorded over an open breach.
    """
    if decision not in _VALID_DECISIONS:
        return {
            "accepted": False,
            "reason": f"Unknown decision {decision!r}; must be one of {sorted(_VALID_DECISIONS)}.",
        }

    applicant = _fetch_applicant(session, applicant_id)
    if applicant is None:
        return {"accepted": False, "reason": f"No applicant record for {applicant_id!r}."}

    application = _build_application_from_record(
        applicant, requested_amount_pkr=requested_amount_pkr, is_clean_facility=is_clean_facility,
    )
    parsed = financial_analyst.analyze(application, session)
    risk = risk_scoring.score(parsed) if not parsed.unresolved_fields else None

    if risk is None:
        _write_audit_log(
            session, actor=actor, action="submit_underwriting_decision",
            resource_id=applicant_id, status="blocked_verification_unavailable",
            payload={
                "requested_decision": decision, "rationale": rationale,
                "reason": "Could not verify against compliance rules -- risk scoring "
                          "unavailable or applicant data incomplete.",
                "unresolved_fields": parsed.unresolved_fields,
            },
        )
        return {
            "accepted": False,
            "reason": "Cannot verify this decision right now (risk scoring unavailable "
                      "or applicant data incomplete). The attempt has been logged.",
        }

    compliance_report = compliance.check(parsed, risk, application, session)
    breaches = [f for f in compliance_report.flags if f.severity == "BREACH"]

    if decision == "APPROVE" and breaches:
        conflicting = [{"rule_id": b.rule_id, "summary": b.summary} for b in breaches]
        _write_audit_log(
            session, actor=actor, action="submit_underwriting_decision",
            resource_id=applicant_id, status="blocked_guardrail_conflict",
            payload={
                "requested_decision": decision, "rationale": rationale,
                "conflicting_flags": conflicting,
            },
        )
        return {
            "accepted": False,
            "reason": "Refused: this would approve over an open compliance breach.",
            "conflicting_flags": conflicting,
        }

    _write_audit_log(
        session, actor=actor, action="submit_underwriting_decision",
        resource_id=applicant_id,
        status="escalated" if decision == "ESCALATE_TO_HUMAN" else "success",
        payload={
            "decision": decision, "rationale": rationale,
            "compliance_flags": [f.model_dump(mode="json") for f in compliance_report.flags],
        },
    )
    return {"accepted": True, "decision": decision}
