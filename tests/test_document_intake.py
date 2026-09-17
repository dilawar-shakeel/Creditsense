"""P9.1 follow-on: agents/document_intake.py. The form-field path is exercised against
the REAL committed template (`frontend/static/assets/creditsense_intake_template.pdf`)
rather than a hand-built fixture -- this is the one place in the suite that needs a
genuine PDF as input, and testing against the real asset also catches template/parser
drift (if the template's field names ever stop matching ALL_FIELD_NAMES, this fails).
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas

from creditsense.agents.document_intake import ALL_FIELD_NAMES, parse_uploaded_pdf

TEMPLATE_PATH = Path("src/creditsense/frontend/static/assets/creditsense_intake_template.pdf")


def _filled_template_bytes(values: dict[str, str]) -> bytes:
    reader = PdfReader(TEMPLATE_PATH)
    writer = PdfWriter()
    writer.append(reader)
    writer.update_page_form_field_values(writer.pages[0], values)
    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _plain_text_pdf_bytes(lines: list[str]) -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf)
    y = 750
    for line in lines:
        c.drawString(50, y, line)
        y -= 20
    c.showPage()
    c.save()
    return buf.getvalue()


@pytest.mark.skipif(not TEMPLATE_PATH.exists(), reason="intake template not generated")
def test_form_field_extraction_reads_filled_values_and_flags_the_rest_unresolved():
    file_bytes = _filled_template_bytes({
        "current_ratio": "1.5",
        "debt_to_equity_ratio": "0.8",
        "existing_loan_exposure_pkr": "Rs. 2,130,819",
        "sector_risk_code": "Retail/Trade",
    })

    result = parse_uploaded_pdf(file_bytes)

    filled = {"current_ratio", "debt_to_equity_ratio", "existing_loan_exposure_pkr", "sector_risk_code"}

    assert result.extraction_method == "form_fields"
    assert result.fields["current_ratio"] == "1.5"
    assert result.fields["existing_loan_exposure_pkr"] == "Rs. 2,130,819"
    assert set(result.unresolved_fields) == set(ALL_FIELD_NAMES) - filled


@pytest.mark.skipif(not TEMPLATE_PATH.exists(), reason="intake template not generated")
def test_form_field_extraction_all_blank_is_all_unresolved():
    result = parse_uploaded_pdf(_filled_template_bytes({}))

    assert result.extraction_method == "form_fields"
    assert result.fields == {}
    assert set(result.unresolved_fields) == set(ALL_FIELD_NAMES)


def test_text_scan_fallback_matches_known_labels():
    file_bytes = _plain_text_pdf_bytes([
        "Current Ratio: 1.4",
        "Debt to Equity Ratio: 0.6",
        "Sector: Technology",
    ])

    result = parse_uploaded_pdf(file_bytes)

    assert result.extraction_method == "text_scan"
    assert result.fields["current_ratio"] == "1.4"
    assert result.fields["debt_to_equity_ratio"] == "0.6"
    assert result.fields["sector_risk_code"] == "Technology"
    assert "existing_loan_exposure_pkr" in result.unresolved_fields


def test_text_scan_never_invents_a_value_for_an_unmatched_field():
    result = parse_uploaded_pdf(_plain_text_pdf_bytes(["Some unrelated line of text."]))

    assert result.fields == {}
    assert set(result.unresolved_fields) == set(ALL_FIELD_NAMES)


def test_garbage_input_never_raises():
    result = parse_uploaded_pdf(b"this is not a pdf at all")

    assert result.fields == {}
    assert set(result.unresolved_fields) == set(ALL_FIELD_NAMES)
    assert result.warnings
