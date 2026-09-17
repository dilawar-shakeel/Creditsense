"""SSE trace endpoint (FRONTEND_REQUIREMENTS.md §2.3): the same underwriting run as
`POST /applications/underwrite`, but streamed as one `stage.completed` event per
pipeline stage instead of a single response at the end -- the data source for the
Application Review screen's live agent trace. `EventSource` can't send a JSON body, so
this route takes the same fields `POST /applications/underwrite` takes, but as query
parameters. An optional `raw_fields` query param (a URL-encoded JSON string -- small,
~7 keys, comfortably fits a query string) mirrors that route's own `raw_fields`-supplied
branch, for a brand-new applicant that has never been seeded/uploaded before (the PDF
intake flow, `new-application.html`); when it's absent, this route falls back to
loading the applicant's own existing document, same as before.

`run_underwriting()` is still sync/blocking (same self-HTTP-call reason as
`applications.py`), so it runs on a plain background thread rather than the request
handler itself; a `queue.Queue` carries each stage event from that thread to the async
generator that actually writes the SSE response, using `anyio.to_thread.run_sync` on
the blocking `queue.get()` so the event loop is never blocked waiting for the next
event.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from typing import Any

import anyio
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from creditsense.agents.application_loader import load_application
from creditsense.agents.pipeline import run_underwriting
from creditsense.agents.schemas import LoanApplication
from creditsense.config import get_settings
from creditsense.db.session import get_db, get_session_factory
from creditsense.ratelimit import RateLimitExceeded, TokenBucketLimiter

logger = logging.getLogger(__name__)
router = APIRouter()

_settings = get_settings()
_limiter = TokenBucketLimiter(
    rate_per_minute=_settings.write_rate_limit_per_minute,
    burst=_settings.write_rate_limit_burst,
)

_SENTINEL = object()


def _format_event(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.get("/applications/underwrite/stream")
def stream_underwriting(
    applicant_id: str,
    http_request: Request,
    raw_fields: str | None = None,
    requested_amount_pkr: float | None = None,
    notes: str | None = None,
    is_clean_facility: bool = False,
    tenor_months: int | None = None,
    session: Session = Depends(get_db),
) -> StreamingResponse:
    client_key = http_request.client.host if http_request.client else "unknown"
    try:
        _limiter.check(client_key)
    except RateLimitExceeded as exc:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded for underwriting submissions.",
            headers={"Retry-After": str(int(exc.retry_after_seconds) + 1)},
        ) from exc

    if raw_fields is not None:
        try:
            parsed_raw_fields = json.loads(raw_fields)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=422, detail=f"raw_fields is not valid JSON: {exc}"
            ) from exc
        application = LoanApplication(
            applicant_id=applicant_id,
            raw_fields=parsed_raw_fields,
            notes=notes,
            requested_amount_pkr=requested_amount_pkr,
            is_clean_facility=is_clean_facility,
            tenor_months=tenor_months,
        )
    else:
        application = load_application(
            applicant_id,
            session,
            requested_amount_pkr=requested_amount_pkr,
            is_clean_facility=is_clean_facility,
            tenor_months=tenor_months,
        )
        if application is None:
            raise HTTPException(
                status_code=404,
                detail=f"No document supplied and no record found for applicant {applicant_id!r}.",
            )
        if notes is not None:
            application = application.model_copy(update={"notes": notes})

    session_factory = get_session_factory()
    event_queue: queue.Queue = queue.Queue()

    def on_stage(stage: str, fields: dict) -> None:
        event_queue.put(("stage.completed", {"stage": stage, **fields}))

    def worker() -> None:
        worker_session = session_factory()
        try:
            decision = run_underwriting(application, worker_session, on_stage=on_stage)
            event_queue.put(("underwriting.decided", decision.model_dump(mode="json")))
        except (KeyError, ValueError) as exc:
            event_queue.put(("error", {"detail": f"Application could not be processed: {exc}"}))
        except Exception:
            logger.exception("Streamed underwriting run failed for %s", applicant_id)
            event_queue.put(("error", {"detail": "An internal error occurred."}))
        finally:
            worker_session.close()
            event_queue.put(_SENTINEL)

    threading.Thread(target=worker, daemon=True).start()

    async def event_generator():
        while True:
            item = await anyio.to_thread.run_sync(event_queue.get)
            if item is _SENTINEL:
                break
            event_name, data = item
            yield _format_event(event_name, data)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
