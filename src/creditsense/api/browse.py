"""Read/query routes the frontend needs (FRONTEND_REQUIREMENTS.md §5) that nothing in
the codebase exposed before: browsing applicants, listing/reading/annotating stored
decisions, searching regulations, and portfolio exposure. All sync `def`, all
`Depends(get_db)`, matching `applications.py`'s conventions.

Each route has its own small private DB helper rather than a shared query layer --
the established seam in this repo (see `mcp_server/tools.py`'s module docstring for
why: tests patch the seam function directly instead of faking SQLAlchemy's
`execute(select(...))` machinery).

Decision storage needs no new write path for the read side: `pipeline.py`'s success
path already writes the full `UnderwritingDecision` into `AuditLog.payload_json` on
every run, so `GET /api/decisions*` just reads what's already there.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import or_, select, text
from sqlalchemy.orm import Session

from creditsense.agents.compliance_rules import P1_SECTOR_CONCENTRATION_CEILING_PCT
from creditsense.api.schemas import (
    ApplicantCreateRequest,
    ApplicantDetail,
    ApplicantSummary,
    DecisionDetail,
    DecisionSummary,
    OverrideRequest,
    PortfolioExposure,
    RegulationChunk,
    RegulationDetail,
    SectorExposureRow,
)
from creditsense.db.models import Applicant, AuditLog
from creditsense.db.session import get_db
from creditsense.mcp_server.masking import allowed_profile_fields, mask_account_number, mask_cnic
from creditsense.rag.retrieval import hybrid_search, lookup_by_regulation_number

router = APIRouter(prefix="/api")

_FIXTURE_PATH = Path("tests/fixtures/messy_applications.json")


def _provenance(regulation_number: str) -> str:
    """R-* is real SBP regulation text; P-*/T-* is simulated internal policy -- the
    corpus manifest's own convention (FRONTEND_REQUIREMENTS.md §3.3). Non-negotiable
    that these are never presented identically in the UI."""
    return "real_sbp_regulation" if regulation_number.startswith("R-") else "simulated_policy"


def _fetch_applicant(session: Session, applicant_id: str) -> Applicant | None:
    return session.execute(
        select(Applicant).where(Applicant.applicant_id == applicant_id)
    ).scalar_one_or_none()


def _fetch_fixture_document(applicant_id: str) -> dict[str, Any] | None:
    if not _FIXTURE_PATH.exists():
        return None
    records = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    record = next((r for r in records if r["applicant_id"] == applicant_id), None)
    if record is None:
        return None
    return {"raw_fields": record["raw_fields"], "notes": record.get("notes")}


def _fetch_audit_log(session: Session, decision_id: str) -> AuditLog | None:
    return session.get(AuditLog, decision_id)


def _search_applicants(session: Session, search: str, limit: int) -> list[Applicant]:
    query = select(Applicant)
    if search:
        pattern = f"%{search}%"
        query = query.where(
            or_(Applicant.applicant_id.ilike(pattern), Applicant.sector.ilike(pattern))
        )
    return list(session.execute(query.limit(limit)).scalars().all())


def _list_audit_logs(session: Session, status: str | None, limit: int) -> list[AuditLog]:
    query = select(AuditLog).where(
        AuditLog.action.in_(("underwriting_decision", "decision_accepted", "decision_overridden"))
    )
    if status:
        query = query.where(AuditLog.status == status)
    query = query.order_by(AuditLog.created_at.desc()).limit(limit)
    return list(session.execute(query).scalars().all())


def _fetch_portfolio_rows(session: Session):
    return session.execute(
        text(
            "SELECT sector, applicant_count, total_exposure_pkr, pct_of_book "
            "FROM portfolio_sector_exposure ORDER BY total_exposure_pkr DESC"
        )
    ).all()


def _decision_summary(row: AuditLog) -> DecisionSummary:
    payload = row.payload_json or {}
    return DecisionSummary(
        id=row.id,
        applicant_id=row.resource_id,
        action=row.action,
        status=row.status,
        decision=payload.get("decision"),
        approved_amount_pkr=payload.get("approved_amount_pkr"),
        rationale=payload.get("rationale"),
        created_at=row.created_at.isoformat(),
    )


@router.get("/applicants", response_model=list[ApplicantSummary])
def search_applicants(search: str = "", limit: int = 25, session: Session = Depends(get_db)):
    limit = max(1, min(limit, 100))
    applicants = _search_applicants(session, search, limit)
    return [
        ApplicantSummary(
            applicant_id=a.applicant_id,
            sector=a.sector,
            years_in_business=a.years_in_business,
            documentation_tier=a.documentation_tier,
        )
        for a in applicants
    ]


@router.post("/applicants", response_model=ApplicantSummary)
def create_applicant(request: ApplicantCreateRequest, session: Session = Depends(get_db)):
    """Onboards a new applicant (new-application.html) -- upserts on applicant_id so a
    corrected resubmission updates the same row rather than erroring or duplicating.
    Called before underwriting runs, so financial_analyst.analyze()'s existing,
    unmodified DB-lookup path finds the 10 bank-record fields naturally."""
    applicant = _fetch_applicant(session, request.applicant_id)
    if applicant is None:
        applicant = Applicant(applicant_id=request.applicant_id)
        session.add(applicant)

    applicant.sector = request.sector
    applicant.years_in_business = request.years_in_business
    # documentation_tier is duplicated onto its own column (read by ApplicantSummary/
    # ApplicantDetail) and inside raw_profile_json (read by financial_analyst.py) --
    # the same redundancy db/seed.py already writes for every seeded applicant.
    applicant.documentation_tier = request.raw_profile_json.get("documentation_tier")
    applicant.raw_profile_json = request.raw_profile_json
    session.commit()

    return ApplicantSummary(
        applicant_id=applicant.applicant_id,
        sector=applicant.sector,
        years_in_business=applicant.years_in_business,
        documentation_tier=applicant.documentation_tier,
    )


@router.get("/applicants/{applicant_id}", response_model=ApplicantDetail)
def get_applicant(applicant_id: str, session: Session = Depends(get_db)):
    applicant = _fetch_applicant(session, applicant_id)
    if applicant is None:
        raise HTTPException(status_code=404, detail=f"No applicant record for {applicant_id!r}.")
    return ApplicantDetail(
        applicant_id=applicant.applicant_id,
        sector=applicant.sector,
        years_in_business=applicant.years_in_business,
        documentation_tier=applicant.documentation_tier,
        account_number=mask_account_number(applicant.account_number),
        cnic=mask_cnic(applicant.cnic),
        financials=allowed_profile_fields(applicant.raw_profile_json),
        raw_document=_fetch_fixture_document(applicant_id),
    )


@router.get("/decisions", response_model=list[DecisionSummary])
def list_decisions(status: str | None = None, limit: int = 50, session: Session = Depends(get_db)):
    limit = max(1, min(limit, 200))
    rows = _list_audit_logs(session, status, limit)
    return [_decision_summary(r) for r in rows]


@router.get("/decisions/{decision_id}", response_model=DecisionDetail)
def get_decision(decision_id: str, session: Session = Depends(get_db)):
    row = _fetch_audit_log(session, decision_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No decision with id {decision_id!r}.")
    return DecisionDetail(
        id=row.id,
        applicant_id=row.resource_id,
        actor=row.actor,
        action=row.action,
        status=row.status,
        payload=row.payload_json,
        created_at=row.created_at.isoformat(),
    )


@router.post("/decisions/{decision_id}/accept", response_model=DecisionDetail)
def accept_decision(decision_id: str, session: Session = Depends(get_db)):
    """Never mutates the original decision row -- appends a new one, same rule as
    `submit_underwriting_decision`'s audit trail elsewhere in this codebase."""
    target = _fetch_audit_log(session, decision_id)
    if target is None:
        raise HTTPException(status_code=404, detail=f"No decision with id {decision_id!r}.")
    row = AuditLog(
        actor="credit-officer",
        action="decision_accepted",
        resource_type="applicant",
        resource_id=target.resource_id,
        payload_json={"accepted_decision_id": decision_id},
        status="success",
    )
    session.add(row)
    session.commit()
    return DecisionDetail(
        id=row.id,
        applicant_id=row.resource_id,
        actor=row.actor,
        action=row.action,
        status=row.status,
        payload=row.payload_json,
        created_at=row.created_at.isoformat(),
    )


@router.post("/decisions/{decision_id}/override", response_model=DecisionDetail)
def override_decision(
    decision_id: str, request: OverrideRequest, session: Session = Depends(get_db)
):
    target = _fetch_audit_log(session, decision_id)
    if target is None:
        raise HTTPException(status_code=404, detail=f"No decision with id {decision_id!r}.")
    row = AuditLog(
        actor="credit-officer",
        action="decision_overridden",
        resource_type="applicant",
        resource_id=target.resource_id,
        payload_json={"overridden_decision_id": decision_id, "reason": request.reason},
        status="success",
    )
    session.add(row)
    session.commit()
    return DecisionDetail(
        id=row.id,
        applicant_id=row.resource_id,
        actor=row.actor,
        action=row.action,
        status=row.status,
        payload=row.payload_json,
        created_at=row.created_at.isoformat(),
    )


@router.get("/regulations/search", response_model=list[RegulationChunk])
def search_regulations(q: str, session: Session = Depends(get_db)):
    chunks = hybrid_search(session, q, top_k=10)
    return [
        RegulationChunk(
            chunk_id=c.chunk_id,
            regulation_number=c.regulation_number,
            clause_title=c.clause_title,
            section_path=c.section_path,
            content=c.content,
            source_type=c.source_type,
            cross_references=c.cross_references,
            score=c.score,
            provenance=_provenance(c.regulation_number),
        )
        for c in chunks
    ]


@router.get("/regulations/{regulation_number}", response_model=RegulationDetail)
def get_regulation(regulation_number: str, session: Session = Depends(get_db)):
    chunks = lookup_by_regulation_number(session, regulation_number)
    if not chunks:
        raise HTTPException(
            status_code=404, detail=f"No regulation found for {regulation_number!r}."
        )
    cross_refs: list[str] = []
    for c in chunks:
        for ref in c.cross_references:
            if ref not in cross_refs:
                cross_refs.append(ref)
    return RegulationDetail(
        regulation_number=regulation_number,
        clause_title=chunks[0].clause_title,
        content="\n\n".join(c.content for c in chunks),
        source_type=chunks[0].source_type,
        cross_references=cross_refs,
        provenance=_provenance(regulation_number),
    )


@router.get("/portfolio/exposure", response_model=PortfolioExposure)
def get_portfolio_exposure(session: Session = Depends(get_db)):
    rows = _fetch_portfolio_rows(session)
    return PortfolioExposure(
        rows=[
            SectorExposureRow(
                sector=r.sector,
                applicant_count=r.applicant_count,
                total_exposure_pkr=float(r.total_exposure_pkr),
                pct_of_book=float(r.pct_of_book),
            )
            for r in rows
        ],
        sector_concentration_limit_pct=P1_SECTOR_CONCENTRATION_CEILING_PCT,
    )
