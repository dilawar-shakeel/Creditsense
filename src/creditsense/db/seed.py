"""P2.3 — seed script: load the synthetic applicant portfolio into Postgres.

Before this script, the 50k applicants existed only as a CSV on disk — the
`applicants` table was empty, which meant `get_applicant_financials(applicant_id)`
(P6.2) and every agent downstream of it had no data to look up. This is what makes
that lookup real.

Loads all rows by default (that's what makes portfolio_sector_exposure (P2.4)
reflect the actual sector mix the ML model trained on, not a sampling accident from
a small subset). Use --limit for a fast dev loop when you don't want to wait for
all 50k.

--cohort makes the table BE a specific, smaller set instead of the full 50k: every
applicant_id present in a JSON file (the shape of tests/fixtures/messy_applications.json
— a list of records with an "applicant_id" key), with every other applicant_id deleted
from the table. Built for the Phase 6 MCP demo, which runs against the 60 messy-fixture
applicants only — every one of those 60 already has a messy document to parse, which is
what makes the demo interesting. Deleting rather than just adding is deliberate: a demo
should never accidentally be running against 50,060 applicants because someone forgot
the table already had data in it.
Note: portfolio_sector_exposure's percentages become small-sample noise at 60 rows —
run a full (uncohorted) seed again before relying on that view's numbers.

Idempotent: upserts on applicant_id, so re-running never duplicates rows — safe to
run again after the generator produces a new CSV.

account_number and cnic (added in migration 0004, for the MCP masking layer, P6.6) are
SYNTHETIC — deterministically derived from applicant_id, not random and never real.
They exist because the rest of the applicant record has no identifier-shaped field to
mask at all (see that migration's docstring). Re-seeding the same applicant_id always
regenerates the same synthetic values.

Run:
    python -m creditsense.db.seed
    python -m creditsense.db.seed --limit 500
    python -m creditsense.db.seed --cohort tests/fixtures/messy_applications.json
    python -m creditsense.db.seed --csv-path path/to/other.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.dialects.postgresql import insert as pg_insert

from creditsense.db.models import Applicant
from creditsense.db.session import session_scope

DEFAULT_CSV_PATH = Path("src/creditsense/data/raw/creditsense_synthetic_portfolio.csv")
DEFAULT_BATCH_SIZE = 1000

# Cosmetic only -- picked to look like real Pakistani bank codes; carries no meaning.
_BANK_CODES = ("SCBL", "HBTB", "MEZN", "ALFH", "UNIL", "BAHL")


def _digits_from_hash(digest: bytes, count: int) -> str:
    """Deterministic digit string of length `count` from a hash digest -- cycles
    through the digest's bytes mod 10 until `count` digits are produced."""
    return "".join(str(digest[i % len(digest)] % 10) for i in range(count))


def synthetic_account_number(applicant_id: str) -> str:
    """A Pakistani-IBAN-shaped (24-char) account number, deterministic from
    applicant_id. Not a real IBAN checksum -- cosmetic only, for the masking demo."""
    digest = hashlib.sha256(f"account:{applicant_id}".encode()).digest()
    check_digits = _digits_from_hash(digest, 2)
    bank_code = _BANK_CODES[digest[0] % len(_BANK_CODES)]
    account_digits = _digits_from_hash(digest[1:], 16)
    return f"PK{check_digits}{bank_code}{account_digits}"


def synthetic_cnic(applicant_id: str) -> str:
    """A CNIC-shaped (13-digit, XXXXX-XXXXXXX-X) identifier, deterministic from
    applicant_id. Not a real CNIC -- cosmetic only, for the masking demo."""
    digest = hashlib.sha256(f"cnic:{applicant_id}".encode()).digest()
    digits = _digits_from_hash(digest, 13)
    return f"{digits[:5]}-{digits[5:12]}-{digits[12]}"


def _clean_scalar(value: Any) -> Any:
    """pandas/numpy scalars (int64, float64, bool_, NaN) aren't valid JSON and would
    make raw_profile_json unparseable for the sector-exposure view's ->>::numeric
    casts. Converts to plain Python types; NaN/NaT becomes None."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, np.floating):
        return None if math.isnan(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _row_to_values(row: dict[str, Any]) -> dict[str, Any]:
    clean = {k: _clean_scalar(v) for k, v in row.items()}
    applicant_id = clean["applicant_id"]
    return {
        "applicant_id": applicant_id,
        "sector": clean.get("sector_risk_code"),
        "years_in_business": clean.get("years_in_business"),
        "documentation_tier": clean.get("documentation_tier"),
        "raw_profile_json": clean,
        "account_number": synthetic_account_number(applicant_id),
        "cnic": synthetic_cnic(applicant_id),
    }


def _load_cohort_ids(cohort_path: Path) -> set[str]:
    records = json.loads(cohort_path.read_text(encoding="utf-8"))
    return {r["applicant_id"] for r in records}


def seed(
    csv_path: Path = DEFAULT_CSV_PATH,
    limit: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    cohort_path: Path | None = None,
) -> int:
    df = pd.read_csv(csv_path)
    cohort_ids: set[str] | None = None

    if cohort_path is not None:
        cohort_ids = _load_cohort_ids(cohort_path)
        df = df[df["applicant_id"].isin(cohort_ids)]
        missing = cohort_ids - set(df["applicant_id"])
        if missing:
            print(f"  warning: {len(missing)} cohort applicant_id(s) not found in "
                  f"{csv_path}: {sorted(missing)}")
    elif limit is not None:
        df = df.head(limit)

    table = Applicant.__table__
    total = 0

    with session_scope() as session:
        connection = session.connection()

        # --cohort means "the applicants table IS this cohort", not "at least this
        # cohort" -- delete anything not in it. No FK from documents/chunks to a real
        # applicant exists in this corpus (regulation documents all carry a NULL
        # applicant_id), so this is safe. Runs before the upsert loop so a partial
        # cohort file can never leave a mix of the old 50k and the new cohort.
        if cohort_ids is not None:
            deleted = connection.execute(
                table.delete().where(table.c.applicant_id.not_in(cohort_ids))
            )
            if deleted.rowcount:
                print(f"  removed {deleted.rowcount} applicant(s) outside the cohort")

        for start in range(0, len(df), batch_size):
            batch = df.iloc[start : start + batch_size]
            rows = [_row_to_values(r) for r in batch.to_dict(orient="records")]

            stmt = pg_insert(table).values(rows)
            stmt = stmt.on_conflict_do_update(
                index_elements=["applicant_id"],
                set_={
                    "sector": stmt.excluded.sector,
                    "years_in_business": stmt.excluded.years_in_business,
                    "documentation_tier": stmt.excluded.documentation_tier,
                    "raw_profile_json": stmt.excluded.raw_profile_json,
                    "account_number": stmt.excluded.account_number,
                    "cnic": stmt.excluded.cnic,
                },
            )
            connection.execute(stmt)
            total += len(rows)
            print(f"  seeded {total}/{len(df)} applicants...")

    print(f"Done. {total} applicant(s) upserted from {csv_path}.")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Load only the first N rows (fast dev loop; omit to load all 50k)."
    )
    parser.add_argument(
        "--cohort", type=Path, default=None, dest="cohort_path",
        help="Load only the applicant_ids present in this JSON file "
             "(e.g. tests/fixtures/messy_applications.json). Overrides --limit."
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    seed(
        csv_path=args.csv_path, limit=args.limit, batch_size=args.batch_size,
        cohort_path=args.cohort_path,
    )


if __name__ == "__main__":
    main()
