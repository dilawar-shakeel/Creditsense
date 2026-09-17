"""P7.1: the HTTP door into the agent pipeline. `run_underwriting()`
(agents/pipeline.py) existed since Phase 5 but was only reachable from the worker CLI
and one MCP tool -- this is the first API route.

Deliberately a plain `def`, not `async def`: risk_scoring.py's ML call is a *sync*
httpx request to this same FastAPI app's own /predict/credit-risk endpoint, with a
blocking time.sleep() backoff. An async route would block the event loop on its own
self-call. A sync route runs on Starlette's threadpool, so the self-call is served by
a different thread and there's no deadlock.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from creditsense.agents.application_loader import load_application
from creditsense.agents.pipeline import run_underwriting
from creditsense.agents.schemas import LoanApplication, UnderwritingDecision
from creditsense.api.schemas import UnderwritingRequest
from creditsense.config import get_settings
from creditsense.db.session import get_db
from creditsense.ratelimit import RateLimitExceeded, TokenBucketLimiter

router = APIRouter()

_settings = get_settings()
_limiter = TokenBucketLimiter(
    rate_per_minute=_settings.write_rate_limit_per_minute,
    burst=_settings.write_rate_limit_burst,
)


@router.post("/applications/underwrite", response_model=UnderwritingDecision)
def underwrite_application(
    request: UnderwritingRequest, http_request: Request, session: Session = Depends(get_db)
) -> UnderwritingDecision:
    client_key = http_request.client.host if http_request.client else "unknown"
    try:
        _limiter.check(client_key)
    except RateLimitExceeded as exc:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded for underwriting submissions.",
            headers={"Retry-After": str(int(exc.retry_after_seconds) + 1)},
        ) from exc

    if request.raw_fields is not None:
        application = LoanApplication(
            applicant_id=request.applicant_id,
            raw_fields=request.raw_fields,
            notes=request.notes,
            requested_amount_pkr=request.requested_amount_pkr,
            is_clean_facility=request.is_clean_facility,
            tenor_months=request.tenor_months,
        )
    else:
        application = load_application(
            request.applicant_id,
            session,
            requested_amount_pkr=request.requested_amount_pkr,
            is_clean_facility=request.is_clean_facility,
            tenor_months=request.tenor_months,
        )
        if application is None:
            raise HTTPException(
                status_code=404,
                detail=f"No document supplied and no record found for applicant "
                       f"{request.applicant_id!r}.",
            )

    try:
        return run_underwriting(application, session)
    except (KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Application could not be processed: {exc}",
        ) from exc
