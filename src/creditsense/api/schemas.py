from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class CreditRiskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current_ratio: float = Field(ge=0)
    debt_to_equity_ratio: float = Field(ge=0)
    revenue_growth_yoy_pct: float
    sector_risk_code: str
    years_in_business: float = Field(ge=0)
    existing_loan_exposure_pkr: float = Field(ge=0)
    late_payment_count_12m: int = Field(ge=0)
    collateral_coverage_ratio: float = Field(ge=0)
    bank_statement_volatility: float = Field(ge=0)
    ecib_score: float | None = Field(default=None, ge=0)
    kibor_sensitivity_pct: float = Field(ge=0)
    documentation_tier: str
    working_capital_cycle_days: float = Field(ge=0)
    utility_default_count_12m: int = Field(ge=0)
    annual_bank_turnover_pkr: float = Field(ge=0)
    guarantor_net_worth_pkr: float = Field(ge=0)
    group_associate_exposure_pkr: float = Field(ge=0)
    sbp_enterprise_tier: str


class CreditRiskResponse(BaseModel):
    default_probability: float = Field(ge=0, le=1)
    decision_cutoff: float = Field(ge=0, le=1)
    decision: str
    recommended_credit_limit_pkr: float | None = Field(default=None, ge=0)
    explanation: str


class UnderwritingRequest(BaseModel):
    """P7.1: request body for POST /applications/underwrite. `raw_fields` is
    optional -- omit it and the server loads the applicant's document itself (the
    messy fixture if one exists, else their own stored profile), the same lookup
    agents/worker.py's CLI has always used. Supply it to submit an uploaded document
    directly, e.g. from a real credit officer's intake flow."""

    model_config = ConfigDict(extra="forbid")

    applicant_id: str
    raw_fields: dict[str, Any] | None = None
    notes: str | None = None
    requested_amount_pkr: float | None = Field(default=None, ge=0)
    is_clean_facility: bool = False
    tenor_months: int | None = Field(default=None, ge=0)


class ApplicantSummary(BaseModel):
    """One row of GET /api/applicants -- deliberately thin (no financials, no masked
    identifiers) since this is a search-result list, not a detail view."""

    model_config = ConfigDict(extra="forbid")

    applicant_id: str
    sector: str | None = None
    years_in_business: float | None = None
    documentation_tier: str | None = None


class ApplicantDetail(BaseModel):
    """GET /api/applicants/{id}. `financials` is already allow-listed by
    mcp_server.masking.allowed_profile_fields -- the same boundary the MCP tool uses,
    reused rather than re-derived. `raw_document` is the matching messy-fixture entry's
    `raw_fields`/`notes` if one exists, for the "As received" panel -- None means this
    applicant has no fixture document, only their own stored profile."""

    model_config = ConfigDict(extra="forbid")

    applicant_id: str
    sector: str | None = None
    years_in_business: float | None = None
    documentation_tier: str | None = None
    account_number: str | None = None
    cnic: str | None = None
    financials: dict[str, Any]
    raw_document: dict[str, Any] | None = None


class DecisionSummary(BaseModel):
    """One row of GET /api/decisions -- the Work Queue's data source, read straight
    off audit_logs.payload_json (already the full UnderwritingDecision on every
    successful/escalated run -- see pipeline.py)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    applicant_id: str | None = None
    action: str
    status: str
    decision: str | None = None
    approved_amount_pkr: float | None = None
    rationale: str | None = None
    created_at: str


class DecisionDetail(BaseModel):
    """GET /api/decisions/{id} -- the full stored audit row."""

    model_config = ConfigDict(extra="forbid")

    id: str
    applicant_id: str | None = None
    actor: str | None = None
    action: str
    status: str
    payload: dict[str, Any] | None = None
    created_at: str


class OverrideRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(min_length=10)


class RegulationChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    regulation_number: str
    clause_title: str | None = None
    section_path: str | None = None
    content: str
    source_type: str | None = None
    cross_references: list[str] = Field(default_factory=list)
    score: float
    provenance: Literal["real_sbp_regulation", "simulated_policy"]


class RegulationDetail(BaseModel):
    """GET /api/regulations/{regulation_number} -- every chunk of the clause,
    reassembled in chunk_index order into one full text, never the 500-char excerpt a
    ComplianceFlag.citations entry carries (FRONTEND_REQUIREMENTS.md §3.3)."""

    model_config = ConfigDict(extra="forbid")

    regulation_number: str
    clause_title: str | None = None
    content: str
    source_type: str | None = None
    cross_references: list[str] = Field(default_factory=list)
    provenance: Literal["real_sbp_regulation", "simulated_policy"]


class SectorExposureRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sector: str
    applicant_count: int
    total_exposure_pkr: float
    pct_of_book: float


class PortfolioExposure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[SectorExposureRow]
    sector_concentration_limit_pct: float


class ApplicantCreateRequest(BaseModel):
    """POST /api/applicants -- onboards a brand-new applicant from the PDF-intake flow
    (new-application.html) as a real `Applicant` row, *before* underwriting runs.
    `raw_profile_json` carries the 10 "bank record" fields (ECIB score, late-payment
    count, etc.) `financial_analyst.analyze()` has always read exclusively from an
    existing Applicant row -- writing a real row here means that logic needs no
    change at all; it just finds one now. Upserts on `applicant_id`, so re-running
    intake for the same id (e.g. a corrected resubmission) updates it in place."""

    model_config = ConfigDict(extra="forbid")

    applicant_id: str
    sector: str | None = None
    years_in_business: float | None = None
    raw_profile_json: dict[str, Any] = Field(default_factory=dict)


class ParsedDocumentFields(BaseModel):
    """POST /api/applicants/parse-document's response -- see agents/document_intake.py.
    `fields` values are deliberately left as raw strings (possibly "Rs. 2,130,819"),
    never numeric-parsed here; parsing.py owns that, later, when the application
    actually runs."""

    model_config = ConfigDict(extra="forbid")

    fields: dict[str, str]
    unresolved_fields: list[str]
    warnings: list[str]
    extraction_method: Literal["form_fields", "text_scan"]