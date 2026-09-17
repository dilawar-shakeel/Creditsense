"""P2.4 — portfolio_sector_exposure view.

ComplianceAgent (P5.4) needs to check the internal policy's concentration limit
(P-1: "maximum single-sector concentration of 25% of the SME book") against the
actual portfolio. That requires summing existing on-book exposure across every
applicant, grouped by sector — a query over up to 50k rows. Doing that as a view
computes it once with one definition, instead of every caller re-aggregating
raw_profile_json in Python and risking three slightly different answers to
"what's our sector exposure".

existing_loan_exposure_pkr lives inside applicants.raw_profile_json (a JSON blob of
the full generator row — Applicant only promotes sector/years_in_business/
documentation_tier to real columns) rather than as a native column, so the view
pulls it out with ->> and casts to numeric. That works whether the seed script
stored it as a JSON number or string, so it's not sensitive to how seed.py
serializes scalars.

get_portfolio_exposure(sector) (P6.4) is a straight SELECT ... WHERE sector = :sector
against this view; pct_of_book is exactly what P-1's 25% check compares against.

Revision ID: 20260916_0003
Revises: 20260916_0002
Create Date: 2026-09-16
"""

from typing import Sequence, Union

from alembic import op

revision: str = "20260916_0003"
down_revision: Union[str, None] = "20260916_0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_CREATE_VIEW = """
    CREATE VIEW portfolio_sector_exposure AS
    WITH sector_totals AS (
        SELECT
            sector,
            COUNT(*) AS applicant_count,
            SUM(
                COALESCE((raw_profile_json ->> 'existing_loan_exposure_pkr')::numeric, 0)
            ) AS total_exposure_pkr
        FROM applicants
        WHERE sector IS NOT NULL
        GROUP BY sector
    )
    SELECT
        sector,
        applicant_count,
        total_exposure_pkr,
        ROUND(
            100.0 * total_exposure_pkr
            / NULLIF(SUM(total_exposure_pkr) OVER (), 0),
            2
        ) AS pct_of_book
    FROM sector_totals
    ORDER BY total_exposure_pkr DESC
"""


def upgrade() -> None:
    op.execute(_CREATE_VIEW)


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS portfolio_sector_exposure")
