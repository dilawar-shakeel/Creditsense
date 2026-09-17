"""Dev-only, one-off generator for the fillable PDF intake template served at
`/ui/assets/creditsense_intake_template.pdf`.

Not part of the app -- `reportlab` is NOT a project dependency (see
`src/creditsense/agents/document_intake.py`'s module docstring for why: the running
app only ever needs to *read* PDFs, via `pypdf`, which already is one). Run this only
when the template's fields need to change:

    uv run --with reportlab python scripts/generate_intake_template.py

Each field's internal PDF name is set to exactly the `raw_fields` key
`agents/document_intake.py::parse_uploaded_pdf` reads it back as -- keep the two in
sync if a field is ever added, renamed, or removed.
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "creditsense" / "frontend" / "static" / "assets"
    / "creditsense_intake_template.pdf"
)

# (pdf field name, label, format hint) -- the 7 fields a real uploaded document would
# carry itself.
DOCUMENT_FIELDS = [
    ("current_ratio", "Current Ratio", "e.g. 1.5"),
    ("debt_to_equity_ratio", "Debt-to-Equity Ratio", "e.g. 0.8"),
    ("existing_loan_exposure_pkr", "Existing Loan Exposure (PKR)", "e.g. Rs. 2,130,819 or 2130819"),
    ("collateral_coverage_ratio", "Collateral Coverage Ratio", "e.g. 0.9 -- leave blank if unknown"),
    ("annual_bank_turnover_pkr", "Annual Bank Turnover (PKR)", "e.g. PKR 40,000,000 or 40000000"),
    ("years_in_business", "Years in Business", "e.g. 8"),
    ("sector_risk_code", "Sector", "e.g. Retail/Trade, Textile, Technology"),
]

# The 10 "bank record" fields -- normally on file at the bank already, but for a
# brand-new applicant there is no file yet, so this intake form is what creates one
# (see api/browse.py::create_applicant). documentation_tier gets a real PDF dropdown
# since it's a fixed 3-value category, not free text.
RECORD_FIELDS = [
    ("ecib_score", "eCIB Score", "e.g. 720"),
    ("kibor_sensitivity_pct", "KIBOR Sensitivity (%)", "e.g. 0.3"),
    ("late_payment_count_12m", "Late Payments (last 12 months)", "e.g. 0"),
    ("bank_statement_volatility", "Bank Statement Volatility", "e.g. 0.4"),
    ("working_capital_cycle_days", "Working Capital Cycle (days)", "e.g. 35"),
    ("utility_default_count_12m", "Utility Defaults (last 12 months)", "e.g. 0"),
    ("revenue_growth_yoy_pct", "Revenue Growth YoY (%)", "e.g. 6.5 -- can be negative"),
    ("guarantor_net_worth_pkr", "Guarantor Net Worth (PKR)", "e.g. 15,000,000"),
    ("group_associate_exposure_pkr", "Group / Associate Exposure (PKR)", "e.g. 0"),
]
DOCUMENTATION_TIER_FIELD = (
    "documentation_tier", "Documentation Tier",
    # A leading blank option so the field's real default is "not chosen" -- a
    # pre-selected real tier would silently guess on the officer's behalf, exactly
    # what this whole form exists to avoid.
    ["", "Bank-Statement-Only", "Unaudited Financials", "Audited Financials"],
)

NOTES_FIELD = ("notes", "Credit Officer's Notes", "English or Urdu-English is fine")


def _text_row(c, form, x, y, name, label, hint, field_width=250):
    c.setFont("Helvetica-Bold", 11)
    c.drawString(x, y, label)
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(x, y - 12, hint)
    form.textfield(
        name=name, tooltip=label,
        x=x + 250, y=y - 16, width=field_width, height=20,
        borderStyle="underlined", forceBorder=True,
    )


def build() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUTPUT_PATH), pagesize=LETTER)
    width, height = LETTER
    form = c.acroForm

    # Page 1 -- the 7 fields a real uploaded document would carry itself.
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 60, "CreditSense — SME Application Intake (1/2)")
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 78, "Document details. Leave a field blank if it isn't available -- do not guess.")

    y = height - 120
    for name, label, hint in DOCUMENT_FIELDS:
        _text_row(c, form, 50, y, name, label, hint)
        y -= 62

    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y, NOTES_FIELD[1])
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(50, y - 12, NOTES_FIELD[2])
    form.textfield(
        name=NOTES_FIELD[0], tooltip=NOTES_FIELD[1],
        x=50, y=y - 90, width=500, height=70,
        borderStyle="underlined", forceBorder=True, fieldFlags="multiline",
    )

    c.showPage()

    # Page 2 -- the 10 "bank record" fields. For a brand-new applicant there is no
    # bank record yet; this page is what creates one (api/browse.py::create_applicant).
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 60, "CreditSense — SME Application Intake (2/2)")
    c.setFont("Helvetica", 10)
    c.drawString(
        50, height - 78,
        "Bank/credit-bureau details, if known. Leave blank if unavailable -- it will be treated as unresolved, never guessed.",
    )

    y = height - 120
    name, label, options = DOCUMENTATION_TIER_FIELD
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y, label)
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(50, y - 12, "Choose one")
    form.choice(
        # reportlab's choice() has a bug where a plain falsy value="" skips setting
        # up the widget's label metadata (AttributeError deep in acroform.py) --
        # wrapping it in a truthy single-item list ([""]) avoids that code path
        # while still selecting the blank option, so the field's real default stays
        # "not chosen" rather than silently pre-selecting a real tier.
        name=name, tooltip=label, options=options, value=[""],
        x=300, y=y - 16, width=250, height=20,
        borderStyle="underlined", forceBorder=True,
    )
    y -= 48

    for name, label, hint in RECORD_FIELDS:
        _text_row(c, form, 50, y, name, label, hint)
        y -= 48

    c.showPage()
    c.save()
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    build()
