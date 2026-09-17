"""P9.1 follow-on: POST /api/applicants/parse-document. Same TestClient(app) style as
the rest of the API test suite; the "happy path" body is a real filled copy of the
committed template (same fixture-building approach as test_document_intake.py) so this
test exercises the real multipart -> UploadFile -> parse_uploaded_pdf chain end to end.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfReader, PdfWriter

from creditsense.api.main import app

TEMPLATE_PATH = Path("src/creditsense/frontend/static/assets/creditsense_intake_template.pdf")

client = TestClient(app)


def _filled_template_bytes() -> bytes:
    reader = PdfReader(TEMPLATE_PATH)
    writer = PdfWriter()
    writer.append(reader)
    writer.update_page_form_field_values(
        writer.pages[0], {"current_ratio": "1.5", "sector_risk_code": "Retail/Trade"}
    )
    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()


@pytest.mark.skipif(not TEMPLATE_PATH.exists(), reason="intake template not generated")
def test_parse_document_happy_path():
    response = client.post(
        "/api/applicants/parse-document",
        files={"file": ("intake.pdf", _filled_template_bytes(), "application/pdf")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["extraction_method"] == "form_fields"
    assert body["fields"]["current_ratio"] == "1.5"
    assert "collateral_coverage_ratio" in body["unresolved_fields"]


def test_parse_document_rejects_non_pdf_content_type():
    response = client.post(
        "/api/applicants/parse-document",
        files={"file": ("notes.txt", b"just some text", "text/plain")},
    )

    assert response.status_code == 400


def test_parse_document_rejects_oversized_upload(monkeypatch):
    monkeypatch.setattr("creditsense.api.intake._MAX_UPLOAD_BYTES", 10)

    response = client.post(
        "/api/applicants/parse-document",
        files={"file": ("intake.pdf", b"%PDF-1.4 way more than ten bytes", "application/pdf")},
    )

    assert response.status_code == 413
