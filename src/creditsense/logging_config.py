"""Central structured logging (P7.3).

`structlog` has been a declared dependency since Phase 5's `pyproject.toml` but was
never actually wired up -- every module still logged through stdlib
`logging.getLogger(__name__)` with no handler ever configured, so output went to the
root logger's last-resort stderr handler with no structure at all.

`configure_logging()` bridges stdlib logging through structlog's `ProcessorFormatter`
so those existing `logger.warning(...)` call sites (rag/rerank.py, agents/compliance.py,
agents/pipeline.py, agents/llm.py, agents/risk_scoring.py, mcp_server/server.py) flow
through the same renderer without being rewritten one by one.
"""

from __future__ import annotations

import logging
import sys

import structlog

from creditsense.config import get_settings

_configured = False


def configure_logging() -> None:
    """Idempotent -- safe to call from every entry point (api/main.py,
    agents/worker.py, mcp_server/__main__.py) without double-configuring handlers."""
    global _configured
    if _configured:
        return
    _configured = True

    settings = get_settings()
    json_output = settings.environment != "development"

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=shared_processors + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    renderer = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer()
    )
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(logging.INFO)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Structured logger for new call sites (pipeline stage events). Existing stdlib
    `logging.getLogger(__name__)` call sites don't need to migrate -- they already
    flow through the same formatter once `configure_logging()` has run."""
    return structlog.get_logger(name)
