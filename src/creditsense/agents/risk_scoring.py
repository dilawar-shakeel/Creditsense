"""RiskScoringAgent (P5.3): call the ML API, turn the score into a plain-English
narrative.

Goes over HTTP to `POST /predict/credit-risk` rather than calling
`CreditSenseBundle.score_applicant` in-process, deliberately -- it exercises the real
deployed path, and it makes "the ML service timed out" (interview question #4) a
genuine scenario the pipeline has to survive, not a hypothetical.

`decision` from the API is always "DECLINE" or "REFER_FOR_LIMIT" -- the model never
says "APPROVE"; only UnderwritingSupervisorAgent can say that, and only after the
compliance check also passes.
"""

from __future__ import annotations

import logging
import time

import httpx
from pydantic import BaseModel

from creditsense.agents.llm import complete_structured
from creditsense.agents.schemas import ParsedFinancials, RiskAssessment
from creditsense.config import get_settings
from creditsense.data.generators.portfolio import SECTOR_NAMES

logger = logging.getLogger(__name__)

DOCUMENTATION_TIERS = frozenset(
    {"Bank-Statement-Only", "Unaudited Financials", "Audited Financials"}
)
SBP_ENTERPRISE_TIERS = frozenset({"Micro", "SE", "ME"})
SECTOR_CODES = frozenset(SECTOR_NAMES)

# The exact 18 fields CreditRiskRequest requires (extra="forbid" -- anything else is a
# 422). Order doesn't matter; completeness does.
_REQUIRED_FIELDS = (
    "current_ratio",
    "debt_to_equity_ratio",
    "revenue_growth_yoy_pct",
    "sector_risk_code",
    "years_in_business",
    "existing_loan_exposure_pkr",
    "late_payment_count_12m",
    "collateral_coverage_ratio",
    "bank_statement_volatility",
    "ecib_score",
    "kibor_sensitivity_pct",
    "documentation_tier",
    "working_capital_cycle_days",
    "utility_default_count_12m",
    "annual_bank_turnover_pkr",
    "guarantor_net_worth_pkr",
    "group_associate_exposure_pkr",
    "sbp_enterprise_tier",
)

MAX_RETRIES = 2
INITIAL_BACKOFF_SECONDS = 1.0

_http_client: httpx.Client | None = None


def _get_http_client() -> httpx.Client:
    global _http_client
    if _http_client is None:
        settings = get_settings()
        _http_client = httpx.Client(
            base_url=settings.ml_api_base_url, timeout=settings.ml_api_timeout_seconds
        )
    return _http_client


class _Narrative(BaseModel):
    narrative: str


def _build_narrative(explanation: str) -> str:
    result = complete_structured(
        "Rewrite this SHAP-style credit risk explanation in plain English a "
        "non-technical loan officer can read in one glance. Keep it to 2-3 sentences. "
        "Do not invent numbers or reasons not present in the input.",
        explanation,
        _Narrative,
        fallback=None,
    )
    return result.narrative if result is not None else explanation


def score(parsed: ParsedFinancials) -> RiskAssessment | None:
    missing = [f for f in _REQUIRED_FIELDS if f not in parsed.fields]
    if missing:
        logger.warning("Cannot score %s -- missing fields: %s", parsed.applicant_id, missing)
        return None

    doc_tier = parsed.fields["documentation_tier"]
    tier = parsed.fields["sbp_enterprise_tier"]
    sector = parsed.fields["sector_risk_code"]
    if doc_tier not in DOCUMENTATION_TIERS:
        logger.warning("Unknown documentation_tier %r for %s", doc_tier, parsed.applicant_id)
        return None
    if tier not in SBP_ENTERPRISE_TIERS:
        logger.warning("Unknown sbp_enterprise_tier %r for %s", tier, parsed.applicant_id)
        return None
    if sector not in SECTOR_CODES:
        logger.warning("Unknown sector_risk_code %r for %s", sector, parsed.applicant_id)
        return None

    payload = {name: parsed.fields[name] for name in _REQUIRED_FIELDS}
    # Defense-in-depth: ParsedFinancials.fields' type forbids None values, so this
    # can't fire via the normal financial_analyst.py -> risk_scoring.py path today
    # (an unknown ecib_score is simply absent, caught by the `missing` check above).
    # Kept in case a future caller constructs ParsedFinancials another way -- ecib_score
    # feeds score_adjusted_risk = ecib_score - financial_stress_count*50 on the server,
    # and a null there should never be silently sent.
    if payload.get("ecib_score") is None:
        logger.warning("ecib_score missing/None for %s -- refusing to score with a "
                        "null score-adjusted-risk input", parsed.applicant_id)
        return None

    client = _get_http_client()
    backoff = INITIAL_BACKOFF_SECONDS
    response: httpx.Response | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.post("/predict/credit-risk", json=payload)
            if response.status_code >= 500:
                raise httpx.HTTPStatusError(
                    "server error", request=response.request, response=response
                )
            break
        except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.ConnectError):
            # A 4xx (our request was malformed) never lands here -- HTTPStatusError is
            # only raised above for 5xx, and TimeoutException/ConnectError are always
            # retryable. Mirrors rag/embed.py's retryable-vs-fatal split.
            if attempt >= MAX_RETRIES:
                logger.warning(
                    "ML API call failed after %d attempt(s) for %s",
                    attempt + 1, parsed.applicant_id, exc_info=True,
                )
                return None
            time.sleep(backoff)
            backoff *= 2
            continue

    if response is None or response.status_code != 200:
        logger.warning(
            "ML API returned %s for %s",
            response.status_code if response is not None else "no response",
            parsed.applicant_id,
        )
        return None

    body = response.json()
    return RiskAssessment(
        default_probability=body["default_probability"],
        decision_cutoff=body["decision_cutoff"],
        model_decision=body["decision"],
        recommended_credit_limit_pkr=body.get("recommended_credit_limit_pkr"),
        shap_explanation_raw=body["explanation"],
        narrative=_build_narrative(body["explanation"]),
    )
