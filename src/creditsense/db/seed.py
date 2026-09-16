"""P2.3 — seed script: load the synthetic applicant portfolio into Postgres.

Before this script, the 50k applicants existed only as a CSV on disk — the
`applicants` table was empty, which meant `get_applicant_financials(applicant_id)`
(P6.2) and every agent downstream of it had no data to look up. This is what makes
that lookup real.

Loads all rows by default (that's what makes portfolio_sector_exposure (P2.4)
reflect the actual sector mix the ML model trained on, not a sampling accident from
a small subset). Use --limit for a fast dev loop when you don't want to wait for
all 50k.

Idempotent: upserts on applicant_id, so re-running never duplicates rows — safe to
run again after the generator produces a new CSV.

Run:
    python -m creditsense.db.seed
    python -m creditsense.db.seed --limit 500
    python -m creditsense.db.seed --csv-path path/to/other.csv
"""

from __future__ import annotations

import argparse
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
    return {
        "applicant_id": clean["applicant_id"],
        "sector": clean.get("sector_risk_code"),
        "years_in_business": clean.get("years_in_business"),
        "documentation_tier": clean.get("documentation_tier"),
        "raw_profile_json": clean,
    }


def seed(
    csv_path: Path = DEFAULT_CSV_PATH,
    limit: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    df = pd.read_csv(csv_path)
    if limit is not None:
        df = df.head(limit)

    table = Applicant.__table__
    total = 0

    with session_scope() as session:
        connection = session.connection()
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
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    seed(csv_path=args.csv_path, limit=args.limit, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
