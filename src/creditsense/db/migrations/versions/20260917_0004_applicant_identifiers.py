"""P6.6 — synthetic applicant identifiers, for the MCP masking layer.

The applicant record (Applicant columns + raw_profile_json) has no direct identifier
in it at all — no account number, no CNIC, no name. It's entirely financial ratios,
scores, and PKR amounts. That means the tracker's P6.6 example ("masked account
numbers") had nothing to mask.

These two columns exist to fix that honestly: synthetic, deterministically generated
from applicant_id by db/seed.py (not random, not real), added specifically so the MCP
masking layer has real identifier-shaped fields to redact. Labeled as synthetic here,
in seed.py's docstring, and in data/raw_corpus/MANIFEST.md — they will look real, and
a demo must never imply otherwise.

Revision ID: 20260917_0004
Revises: 20260916_0003
Create Date: 2026-09-17
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20260917_0004"
down_revision: Union[str, None] = "20260916_0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("applicants", sa.Column("account_number", sa.String(length=24), nullable=True))
    op.add_column("applicants", sa.Column("cnic", sa.String(length=15), nullable=True))


def downgrade() -> None:
    op.drop_column("applicants", "cnic")
    op.drop_column("applicants", "account_number")
