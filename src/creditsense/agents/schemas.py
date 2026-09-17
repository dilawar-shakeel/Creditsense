"""Structured types for the agent pipeline (P5.7).

Every agent below reads/returns one of these — built first because everything else
imports from here. Same idiom as `api/schemas.py`: `BaseModel`, `ConfigDict(extra="forbid")`,
`Field(ge=..., le=...)` constraints on boundary types.

`ComplianceFlag.citations` uses `Field(min_length=1)` — this is the structural half of
§2.3's "must not state a duty/limit without citing a retrieved source chunk": a flag with
no citation cannot be constructed at all. The other half (verifying the citation was
actually retrieved, not invented) is enforced in code in `compliance.py`.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class LoanApplication(BaseModel):
    """Pipeline input: one applicant's uploaded document plus the proposed structure."""

    model_config = ConfigDict(extra="forbid")

    applicant_id: str
    raw_fields: dict[str, Any]  # as uploaded -- messy, unparsed (see parsing.py)
    notes: str | None = None
    requested_amount_pkr: float | None = None  # None -> use the model's recommended limit
    is_clean_facility: bool = False  # unsecured / personal-guarantee only (R-9)
    tenor_months: int | None = None


class NoteInsights(BaseModel):
    """The only LLM output inside FinancialAnalystAgent -- free-text notes only, never
    numbers. See parsing.py for why numeric fields are never handed to an LLM."""

    model_config = ConfigDict(extra="forbid")

    seasonality: bool = False
    seasonal_peak: str | None = None
    stated_concerns: list[str] = Field(default_factory=list)
    summary_english: str = ""


class ParsedFinancials(BaseModel):
    """FinancialAnalystAgent output. `field_sources` records where every value in
    `fields` came from, so a downstream reader can tell a bank record apart from a
    figure copied off the uploaded document."""

    model_config = ConfigDict(extra="forbid")

    applicant_id: str
    fields: dict[str, float | int | str]
    field_sources: dict[str, Literal["document", "applicant_record", "derived"]]
    unresolved_fields: list[str] = Field(default_factory=list)
    note_insights: NoteInsights | None = None
    parse_warnings: list[str] = Field(default_factory=list)


class RiskAssessment(BaseModel):
    """RiskScoringAgent output. `model_decision` mirrors the ML API's own vocabulary --
    it is never "APPROVE"; only the supervisor can say that."""

    model_config = ConfigDict(extra="forbid")

    default_probability: float = Field(ge=0, le=1)
    decision_cutoff: float = Field(ge=0, le=1)
    model_decision: Literal["DECLINE", "REFER_FOR_LIMIT"]
    recommended_credit_limit_pkr: float | None = Field(default=None, ge=0)
    shap_explanation_raw: str  # verbatim from the API's `explanation` field
    narrative: str  # the agent's plain-English rewrite of shap_explanation_raw


class Citation(BaseModel):
    """A single retrieved (or directly looked-up) regulation/policy clause."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    regulation_number: str
    clause_title: str
    excerpt: str


class ComplianceFlag(BaseModel):
    """One compliance finding. `origin="deterministic"` means a hardcoded numeric rule
    fired (compliance_rules.py); `origin="retrieved"` means the RAG/LLM advisory pass
    raised it, and by the time it reaches here its citation has already been verified
    against what hybrid_search actually returned (see compliance.py)."""

    model_config = ConfigDict(extra="forbid")

    rule_id: str  # "R-5", "P-7", ...
    severity: Literal["BREACH", "ADVISORY"]
    summary: str
    citations: list[Citation] = Field(min_length=1)
    origin: Literal["deterministic", "retrieved"]


class ComplianceReport(BaseModel):
    """ComplianceAgent output."""

    model_config = ConfigDict(extra="forbid")

    flags: list[ComplianceFlag] = Field(default_factory=list)
    checks_performed: list[str] = Field(default_factory=list)
    insufficient_data: bool = False
    escalation_reasons: list[str] = Field(default_factory=list)


class UnderwritingDecision(BaseModel):
    """UnderwritingSupervisorAgent output -- the pipeline's final answer."""

    model_config = ConfigDict(extra="forbid")

    applicant_id: str
    decision: Literal["APPROVE", "DECLINE", "ESCALATE_TO_HUMAN"]
    approved_amount_pkr: float | None = Field(default=None, ge=0)
    rationale: str
    risk: RiskAssessment | None = None
    compliance: ComplianceReport | None = None
    escalation_reasons: list[str] = Field(default_factory=list)
    # Frontend gap 1 (FRONTEND_REQUIREMENTS.md §2.2): FinancialAnalystAgent's output was
    # computed by every run but discarded after being handed to risk_scoring -- without
    # it there is nothing for a UI to show for the "messy document became clean data"
    # stage. None only when analysis itself never ran (there is no such path today, but
    # the field stays optional rather than widening every other caller's contract).
    parsed: ParsedFinancials | None = None
