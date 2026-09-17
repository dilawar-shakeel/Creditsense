"""Pipeline (P5.5/P5.6 wiring): analyze -> score -> check -> decide, with every
stage's failure turned into an escalation reason rather than an exception, and the
final decision written to audit_logs.

Wired into an HTTP endpoint at P7.1 (`api/applications.py`). P7.2/P7.3 added here:
a stage that *raises* (rather than degrading, e.g. a DB error) still leaves a record --
the failure is written to `audit_logs` via a *fresh* `session_scope()`, because the
session passed in has already had its transaction rolled back by the exception and
cannot be committed to anymore. And every stage emits a `stage.completed` structured
event with its duration, both for operational visibility and because this is exactly
the per-agent data Phase 9's live streaming UI (`FRONTEND_REQUIREMENTS.md`) needs.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from sqlalchemy.orm import Session

from creditsense.agents import compliance, financial_analyst, risk_scoring, supervisor
from creditsense.agents.schemas import LoanApplication, UnderwritingDecision
from creditsense.config import get_settings
from creditsense.db.models import AuditLog
from creditsense.db.session import session_scope
from creditsense.logging_config import get_logger

logger = logging.getLogger(__name__)
stage_logger = get_logger(__name__)


def _log_stage(stage: str, applicant_id: str, started_at: float, **fields) -> None:
    stage_logger.info(
        "stage.completed",
        stage=stage,
        applicant_id=applicant_id,
        duration_ms=round((time.monotonic() - started_at) * 1000, 1),
        **fields,
    )


def _record_failure(applicant_id: str, exc: Exception) -> None:
    """Writes a `status="failed"` audit row in its own session -- the session the
    pipeline was using is not safe to commit on anymore (see module docstring)."""
    try:
        with session_scope() as failure_session:
            failure_session.add(
                AuditLog(
                    actor="UnderwritingSupervisorAgent",
                    action="underwriting_decision",
                    resource_type="applicant",
                    resource_id=applicant_id,
                    payload_json={"error_type": type(exc).__name__, "error": str(exc)},
                    status="failed",
                )
            )
    except Exception:
        # Recording the failure is best-effort -- never let a broken audit write mask
        # the original exception the caller is about to see.
        logger.exception("Failed to write failure audit row for %s", applicant_id)


def run_underwriting(
    application: LoanApplication,
    session: Session,
    *,
    on_stage: Callable[[str, dict], None] | None = None,
) -> UnderwritingDecision:
    """`on_stage`, if given, is called with (stage_name, fields) right after each
    stage's own `_log_stage` structlog event, with the same fields -- this is the
    frontend's SSE trace hook (api/stream.py). Optional and defaulted to None so every
    existing caller (worker.py, mcp_server/tools.py, the existing test suite) is
    unaffected."""
    settings = get_settings()
    applicant_id = application.applicant_id
    pipeline_started = time.monotonic()

    def _emit(stage: str, started_at: float, **fields) -> None:
        _log_stage(stage, applicant_id, started_at, **fields)
        if on_stage is not None:
            on_stage(stage, fields)

    try:
        stage_started = time.monotonic()
        parsed = financial_analyst.analyze(application, session)
        _emit(
            "analyze", stage_started,
            unresolved_fields=len(parsed.unresolved_fields),
        )

        # risk_scoring.score() already checks parsed.fields has every required field
        # and returns None gracefully if not -- no need to duplicate that check here
        # using parsed.unresolved_fields, which tracks a slightly different thing
        # (fields the analyst couldn't source from either the document or the
        # applicant record).
        stage_started = time.monotonic()
        risk = risk_scoring.score(parsed)
        if risk is None:
            logger.warning("Risk scoring returned no result for %s", applicant_id)
        _emit(
            "score", stage_started,
            default_probability=risk.default_probability if risk else None,
        )

        stage_started = time.monotonic()
        compliance_report = (
            compliance.check(parsed, risk, application, session) if risk is not None else None
        )
        _emit(
            "check", stage_started,
            breach_count=sum(
                1 for f in (compliance_report.flags if compliance_report else []) if f.severity == "BREACH"
            ),
            advisory_count=sum(
                1 for f in (compliance_report.flags if compliance_report else []) if f.severity == "ADVISORY"
            ),
        )

        stage_started = time.monotonic()
        decision = supervisor.decide(
            parsed,
            risk,
            compliance_report,
            band=settings.escalation_probability_band,
            requested_amount_pkr=application.requested_amount_pkr,
        )
        _emit("decide", stage_started, decision=decision.decision)
        decision = decision.model_copy(update={"parsed": parsed})
    except Exception as exc:
        _record_failure(applicant_id, exc)
        raise

    session.add(
        AuditLog(
            actor="UnderwritingSupervisorAgent",
            action="underwriting_decision",
            resource_type="applicant",
            resource_id=applicant_id,
            payload_json=decision.model_dump(mode="json"),
            status="success" if decision.decision != "ESCALATE_TO_HUMAN" else "escalated",
        )
    )
    session.commit()

    stage_logger.info(
        "underwriting.decided",
        applicant_id=applicant_id,
        decision=decision.decision,
        total_ms=round((time.monotonic() - pipeline_started) * 1000, 1),
    )

    return decision
