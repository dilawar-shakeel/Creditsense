from __future__ import annotations

from functools import lru_cache

import pandas as pd
from fastapi import APIRouter, HTTPException

from creditsense.api.schemas import CreditRiskRequest, CreditRiskResponse
from creditsense.config import get_settings
from creditsense.ml.models import CreditSenseBundle

router = APIRouter()


@lru_cache
def get_model_bundle() -> CreditSenseBundle:
    return CreditSenseBundle.load(get_settings().model_bundle_path)


@router.post("/predict/credit-risk", response_model=CreditRiskResponse)
def predict_credit_risk(request: CreditRiskRequest) -> CreditRiskResponse:
    try:
        bundle = get_model_bundle()
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail="The trained CreditSense model is not available.",
        ) from exc

    try:
        result = bundle.score_applicant(pd.DataFrame([request.model_dump()]))
    except (KeyError, RuntimeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Applicant data could not be scored: {exc}",
        ) from exc
    return CreditRiskResponse.model_validate(result)