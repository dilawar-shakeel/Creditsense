"""
Fix 1-E — build a small, messy applicant fixture.

P1.3 (OCR digit noise / currency formatting / missing collateral) and P1.4 (Urdu-English
notes) were deliberately skipped for the 50,000-row synthetic portfolio (see its
metadata's scope_decisions). That left two problems: interview question #3 ("what
data-quality problem did you simulate?") had no answer, and FinancialAnalystAgent
(P5.2) — whose whole job is tolerating messy input — would have nothing messy to
tolerate.

Rather than reopening the 50,000-row generator, this script samples 60 real applicants
from the clean dataset and deterministically injects the same four problems at small
scale, into tests/fixtures/messy_applications.json. Deterministic (fixed seed) so the
fixture is reproducible; run again after regenerating the base CSV if row content
should be refreshed.

Run: python tests/fixtures/build_messy_applications.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

SOURCE_CSV = Path("src/creditsense/data/raw/creditsense_synthetic_portfolio.csv")
OUTPUT_PATH = Path("tests/fixtures/messy_applications.json")

N_SAMPLES = 60
SEED = 42

# Raw fields as they'd realistically arrive from an uploaded bank statement / financial
# summary, before FinancialAnalystAgent normalizes them — a mix of numeric and PKR
# monetary fields.
RAW_NUMERIC_FIELDS = [
    "current_ratio",
    "debt_to_equity_ratio",
    "existing_loan_exposure_pkr",
    "collateral_coverage_ratio",
    "annual_bank_turnover_pkr",
    "years_in_business",
]
PKR_FIELDS = ["existing_loan_exposure_pkr", "annual_bank_turnover_pkr"]

CURRENCY_FORMATS = [
    lambda v: f"Rs. {v:,.0f}",
    lambda v: f"PKR {v:,.0f}",
    lambda v: f"{v:,.0f}",  # numeric-only, no currency marker
]

URDU_ENGLISH_NOTES = [
    "client ka cash flow seasonal hai, Eid se pehle spike hota hai",
    "business slow chal raha hai is saal, load-shedding ne bhi asar dala",
    "owner ne bataya keh naya order aaya hai, next quarter mein growth expected hai",
    "thoda late payment history hai but overall trustworthy client lagta hai",
    "collateral documents abhi tak submit nahi hue, follow-up required",
    "bohat purana relationship hai bank ke saath, koi issue nahi raha ab tak",
    "revenue is stable lekin buyer concentration zyada hai teen clients pe",
    "digital payments barh rahay hain, cash reliance kam ho rahi hai gradually",
    "guarantor ka net worth strong hai, risk kam samjha ja sakta hai",
]


def transpose_two_digits(value: float, rng: random.Random) -> str:
    """Simulate an OCR digit-transposition error on a whole-number PKR figure."""
    digits = list(str(int(round(value))))
    if len(digits) < 2:
        return str(int(round(value)))
    i = rng.randrange(len(digits) - 1)
    digits[i], digits[i + 1] = digits[i + 1], digits[i]
    return "".join(digits)


def build_fixture(frame: pd.DataFrame, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    sample = frame.sample(n=n, random_state=seed).reset_index(drop=True)

    ocr_noise_indices = set(rng.sample(range(n), k=max(1, round(n * 0.10))))  # ~10%
    missing_collateral_indices = set(
        rng.sample(range(n), k=max(1, round(n * 0.05)))  # ~5%
    ) - ocr_noise_indices
    notes_indices = set(rng.sample(range(n), k=len(URDU_ENGLISH_NOTES)))

    records = []
    for i, row in sample.iterrows():
        raw_fields: dict[str, object] = {}
        noise_applied: list[str] = []

        for field in RAW_NUMERIC_FIELDS:
            value = row[field]
            if field == "collateral_coverage_ratio" and i in missing_collateral_indices:
                raw_fields[field] = None
                if "missing_collateral" not in noise_applied:
                    noise_applied.append("missing_collateral")
                continue
            if field in PKR_FIELDS and i in ocr_noise_indices:
                raw_fields[field] = transpose_two_digits(value, rng)
                if "ocr_digit_transposition" not in noise_applied:
                    noise_applied.append("ocr_digit_transposition")
            elif field in PKR_FIELDS:
                fmt = CURRENCY_FORMATS[i % len(CURRENCY_FORMATS)]
                raw_fields[field] = fmt(float(value))
                if "currency_format_variation" not in noise_applied:
                    noise_applied.append("currency_format_variation")
            else:
                raw_fields[field] = value

        record = {
            "applicant_id": row["applicant_id"],
            "sector_risk_code": row["sector_risk_code"],
            "raw_fields": raw_fields,
            "notes": None,
        }
        if i in notes_indices:
            note_pool_index = list(notes_indices).index(i)
            record["notes"] = URDU_ENGLISH_NOTES[note_pool_index % len(URDU_ENGLISH_NOTES)]
            noise_applied.append("urdu_english_notes")

        record["noise_applied"] = noise_applied
        records.append(record)

    return records


def main() -> None:
    frame = pd.read_csv(SOURCE_CSV)
    records = build_fixture(frame, N_SAMPLES, SEED)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {len(records)} messy applicant records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
