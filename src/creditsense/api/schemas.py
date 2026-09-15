from __future__ import annotations

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