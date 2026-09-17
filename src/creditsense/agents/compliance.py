"""ComplianceAgent (P5.4): deterministic numeric limits + a RAG/LLM advisory pass for
everything not hardcoded, with every advisory citation verified in code.

Two passes:
  1. Deterministic (compliance_rules.py) -- the PKR ceilings. Cannot depend on retrieval
     landing the right clause in the top-k, so these are plain Python. Each flag's
     citation is fetched with `lookup_by_regulation_number`, a direct lookup that never
     depends on retrieval luck.
  2. Advisory (this module) -- retrieves candidate clauses for the proposed structure
     and asks the LLM what else might apply. This is where §2.3's rule lives:
     "must not state a duty/limit without citing a retrieved source chunk." The
     structural half is `ComplianceFlag.citations: Field(min_length=1)` in schemas.py --
     a flag literally cannot be constructed without one. The enforcement half is here:
     any citation naming a chunk_id that wasn't actually in this call's retrieved set is
     dropped before it ever reaches a caller, and the drop is logged as an escalation
     reason rather than silently discarded.

Confidence-floor scale warning (read before touching the threshold): rag/eval.py's
OUT_OF_SCOPE_SCORE_FLOOR (0.01) is calibrated for RRF-scale scores (~0.02) and is used
with use_reranker=False configurations. This agent always calls hybrid_search with
use_reranker=True, whose top score is an LLM score on a 0-10 scale -- a 0.01 floor would
never trip and would silently treat every retrieval as confident. Settings.
min_rerank_score_for_citation (default 5.0) is the separate, scale-correct floor for
this agent. Do not import or reuse OUT_OF_SCOPE_SCORE_FLOOR here.
"""

from __future__ import annotations

import logging

from sqlalchemy import text
from sqlalchemy.orm import Session

from creditsense.agents import compliance_rules
from creditsense.agents.llm import complete_structured
from creditsense.agents.schemas import (
    Citation,
    ComplianceFlag,
    ComplianceReport,
    LoanApplication,
    ParsedFinancials,
    RiskAssessment,
)
from creditsense.config import get_settings
from creditsense.rag.retrieval import RetrievedChunk, hybrid_search, lookup_by_regulation_number

logger = logging.getLogger(__name__)

_ADVISORY_SYSTEM_PROMPT = (
    "You are a compliance reviewer for an SME lender in Pakistan. Given a proposed "
    "loan structure and a set of retrieved regulation/policy clauses, list any "
    "additional applicable requirements not already covered by the standard exposure, "
    "clean-facility, approval-authority, or documentation-tier checks. For each "
    "requirement, you MUST cite the exact chunk_id of a clause from the candidates "
    "given to you, and set rule_id to that clause's regulation_number exactly as given "
    "(e.g. 'R-9', 'T-1', 'P-21') -- never the clause title. Never state a requirement "
    "without a chunk_id from the candidate list -- if none of the candidates support a "
    "requirement, omit it."
)


def _citation_from_chunk(chunk: RetrievedChunk) -> Citation:
    return Citation(
        chunk_id=chunk.chunk_id,
        regulation_number=chunk.regulation_number,
        clause_title=chunk.clause_title,
        excerpt=chunk.content[:500],
    )


def _resolve_proposed_amount(
    application: LoanApplication, risk: RiskAssessment
) -> float | None:
    if application.requested_amount_pkr is not None:
        return application.requested_amount_pkr
    return risk.recommended_credit_limit_pkr


def _sector_pct_of_book(session: Session, sector: str | None) -> float | None:
    if not sector:
        return None
    row = session.execute(
        text("SELECT pct_of_book FROM portfolio_sector_exposure WHERE sector = :sector"),
        {"sector": sector},
    ).first()
    return float(row.pct_of_book) if row is not None else None


def _deterministic_flags(
    application: LoanApplication,
    parsed: ParsedFinancials,
    proposed_amount: float,
    session: Session,
    sector_pct_of_book: float | None,
) -> list[ComplianceFlag]:
    rule_flags = compliance_rules.run_all(
        application, parsed, proposed_amount=proposed_amount, sector_pct_of_book=sector_pct_of_book
    )
    flags: list[ComplianceFlag] = []
    for rule_flag in rule_flags:
        chunks = lookup_by_regulation_number(session, rule_flag.rule_id)
        if not chunks:
            logger.warning(
                "Rule %s fired but no corpus chunk found to cite -- dropping flag "
                "rather than stating a limit with no source", rule_flag.rule_id,
            )
            continue
        flags.append(
            ComplianceFlag(
                rule_id=rule_flag.rule_id,
                severity=rule_flag.severity,  # type: ignore[arg-type]
                summary=rule_flag.summary,
                citations=[_citation_from_chunk(c) for c in chunks],
                origin="deterministic",
            )
        )
    return flags


def _advisory_flags(
    application: LoanApplication,
    parsed: ParsedFinancials,
    session: Session,
    escalation_reasons: list[str],
) -> tuple[list[ComplianceFlag], bool]:
    """Returns (flags, retrieval_was_insufficient). The second value is an explicit
    signal rather than something the caller has to infer from escalation_reasons text."""
    from pydantic import BaseModel, Field

    class AdvisoryFinding(BaseModel):
        rule_id: str
        summary: str
        chunk_id: str

    class AdvisoryFindings(BaseModel):
        findings: list[AdvisoryFinding] = Field(default_factory=list)

    query = (
        f"Sector: {parsed.fields.get('sector_risk_code')}. "
        f"Documentation tier: {parsed.fields.get('documentation_tier')}. "
        f"Enterprise tier: {parsed.fields.get('sbp_enterprise_tier')}. "
        f"Clean facility: {application.is_clean_facility}. "
        f"What SBP regulation or internal policy requirements apply to this loan structure?"
    )
    retrieved = hybrid_search(session, query, top_k=5, use_reranker=True)
    if not retrieved:
        escalation_reasons.append("Advisory retrieval returned no candidates.")
        return [], True

    settings = get_settings()
    confident = [c for c in retrieved if c.score >= settings.min_rerank_score_for_citation]
    if not confident:
        escalation_reasons.append(
            "Advisory retrieval's top candidate scored below the confidence floor "
            f"({settings.min_rerank_score_for_citation}) -- treating as insufficient data."
        )
        return [], True

    candidate_ids = {c.chunk_id for c in retrieved}
    by_id = {c.chunk_id: c for c in retrieved}

    candidates_text = "\n".join(
        f"- chunk_id: {c.chunk_id} | {c.regulation_number}: {c.clause_title}\n  {c.content[:500]}"
        for c in retrieved
    )
    result = complete_structured(
        _ADVISORY_SYSTEM_PROMPT,
        f"Proposed structure:\n{query}\n\nCandidates:\n{candidates_text}",
        AdvisoryFindings,
        fallback=None,
    )
    if result is None:
        escalation_reasons.append("Advisory compliance LLM call failed or was unavailable.")
        return [], True

    flags: list[ComplianceFlag] = []
    for finding in result.findings:
        # The enforcement half of "no claim without a cited source chunk": a citation
        # naming a chunk_id this call didn't actually retrieve is discarded, not trusted.
        if finding.chunk_id not in candidate_ids:
            escalation_reasons.append(
                f"Advisory finding for rule {finding.rule_id!r} cited chunk_id "
                f"{finding.chunk_id!r}, which was not in the retrieved candidate set -- "
                f"dropped."
            )
            continue
        chunk = by_id[finding.chunk_id]
        flags.append(
            ComplianceFlag(
                # rule_id comes from the verified chunk itself, not the LLM's own
                # rule_id field -- the chunk_id is checked against candidate_ids above,
                # but its regulation_number is ground truth already in hand, so there's
                # no reason to trust the model to transcribe it correctly too.
                rule_id=chunk.regulation_number,
                severity="ADVISORY",
                summary=finding.summary,
                citations=[_citation_from_chunk(chunk)],
                origin="retrieved",
            )
        )
    return flags, False


def check(
    parsed: ParsedFinancials,
    risk: RiskAssessment,
    application: LoanApplication,
    session: Session,
) -> ComplianceReport:
    proposed_amount = _resolve_proposed_amount(application, risk)
    escalation_reasons: list[str] = []
    checks_performed = ["R-5", "R-9", "P-2", "documentation-tier-cap", "P-1"]

    if proposed_amount is None:
        return ComplianceReport(
            flags=[],
            checks_performed=[],
            insufficient_data=True,
            escalation_reasons=["No requested amount and no recommended credit limit "
                                 "to check against -- cannot run deterministic rules."],
        )

    try:
        sector = parsed.fields.get("sector_risk_code")
        sector_pct = _sector_pct_of_book(session, sector)

        flags = _deterministic_flags(application, parsed, proposed_amount, session, sector_pct)
        advisory_flags, advisory_insufficient = _advisory_flags(
            application, parsed, session, escalation_reasons
        )
        flags.extend(advisory_flags)
    except Exception:
        # A DB/retrieval failure here must never look like "zero flags found" -- that
        # would let the supervisor read silence as a clean compliance check. Degrade
        # to the same insufficient_data=True signal an inconclusive advisory pass
        # already uses (P7.2), which supervisor.decide() always escalates, never
        # approves, on.
        logger.exception(
            "Compliance check failed for %s -- treating as insufficient data",
            application.applicant_id,
        )
        return ComplianceReport(
            flags=[],
            checks_performed=[],
            insufficient_data=True,
            escalation_reasons=["Compliance check failed due to an internal error."],
        )

    return ComplianceReport(
        flags=flags,
        checks_performed=checks_performed,
        insufficient_data=advisory_insufficient,
        escalation_reasons=escalation_reasons,
    )
