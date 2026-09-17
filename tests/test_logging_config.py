"""P7.3: configure_logging() is idempotent, picks a renderer by environment, and the
request-ID middleware round-trips X-Request-ID through response headers.
"""

from __future__ import annotations

import logging

import structlog
from fastapi.testclient import TestClient

import creditsense.logging_config as logging_config


def test_configure_logging_is_idempotent(monkeypatch):
    logging_config._configured = False

    logging_config.configure_logging()
    handlers_after_first = list(logging.getLogger().handlers)

    logging_config.configure_logging()  # second call must not add a second handler
    handlers_after_second = list(logging.getLogger().handlers)

    assert handlers_after_first == handlers_after_second
    logging_config._configured = True  # restore for the rest of the suite


def test_json_renderer_selected_outside_development(monkeypatch):
    monkeypatch.setattr(
        "creditsense.logging_config.get_settings",
        lambda: type("S", (), {"environment": "production"})(),
    )
    logging_config._configured = False

    logging_config.configure_logging()

    handler = logging.getLogger().handlers[0]
    assert isinstance(handler.formatter, structlog.stdlib.ProcessorFormatter)

    logging_config._configured = True


def test_request_id_middleware_echoes_header():
    from creditsense.api.main import app

    response = TestClient(app).get("/health", headers={"X-Request-ID": "test-request-123"})

    assert response.headers["X-Request-ID"] == "test-request-123"


def test_request_id_middleware_generates_one_when_absent():
    from creditsense.api.main import app

    response = TestClient(app).get("/health")

    assert "X-Request-ID" in response.headers
    assert len(response.headers["X-Request-ID"]) > 0
