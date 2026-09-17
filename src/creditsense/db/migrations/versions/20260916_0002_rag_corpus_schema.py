"""Unblock the RAG corpus: nullable document.applicant_id, clause identity on chunks,
content_tsv as a generated column instead of a dead nullable one.

Fixes four problems (P4.0 in the execution plan) that would otherwise block every
Phase 4 script from storing anything:

1. documents.applicant_id was NOT NULL with an FK to applicants, but a regulation
   document belongs to no applicant. Made nullable.
2. chunks.content_tsv was a plain nullable column nothing ever populated, so keyword
   search would silently return zero rows forever. Recreated as a STORED generated
   column so Postgres maintains it and it can never drift.
3. chunks had no clause identity (regulation_number, clause_title, section_path,
   cross_references) — nothing to cite, nothing to filter on, no way to test
   multi-hop retrieval. Added.
4. Nothing stopped re-ingesting the same document from duplicating chunk rows. Added
   a unique constraint on (document_id, chunk_index).

Revision ID: 20260916_0002
Revises: 20260915_0001
Create Date: 2026-09-16
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "20260916_0002"
down_revision: Union[str, None] = "20260915_0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. documents.applicant_id -> nullable
    op.alter_column(
        "documents", "applicant_id", existing_type=sa.String(length=64), nullable=True
    )

    # 2. clause identity columns on chunks
    op.add_column("chunks", sa.Column("regulation_number", sa.String(length=32), nullable=True))
    op.add_column("chunks", sa.Column("clause_title", sa.String(length=256), nullable=True))
    op.add_column("chunks", sa.Column("section_path", sa.String(length=256), nullable=True))
    op.add_column("chunks", sa.Column("cross_references", sa.JSON(), nullable=True))
    op.add_column("chunks", sa.Column("source_type", sa.String(length=32), nullable=True))
    op.add_column("chunks", sa.Column("token_count", sa.Integer(), nullable=True))
    op.create_index("ix_chunks_regulation_number", "chunks", ["regulation_number"], unique=False)
    op.create_index("ix_chunks_source_type", "chunks", ["source_type"], unique=False)

    # 3. content_tsv: drop the dead plain column, recreate as a generated column.
    # Must use the two-argument to_tsvector('english', ...) — the one-argument form
    # depends on a session-level setting, so Postgres refuses it in a generated
    # column expression (it must be IMMUTABLE). Title and regulation number are
    # weighted 'A' (highest) so a title/number match outranks a passing mention in
    # the body ('B').
    op.drop_index("ix_chunks_content_tsv", table_name="chunks")
    op.drop_column("chunks", "content_tsv")
    op.execute(
        """
        ALTER TABLE chunks ADD COLUMN content_tsv tsvector
        GENERATED ALWAYS AS (
            setweight(to_tsvector('english', coalesce(clause_title, '')), 'A') ||
            setweight(to_tsvector('english', coalesce(regulation_number, '')), 'A') ||
            setweight(to_tsvector('english', content), 'B')
        ) STORED
        """
    )
    op.create_index(
        "ix_chunks_content_tsv", "chunks", ["content_tsv"], unique=False, postgresql_using="gin"
    )

    # 4. no duplicate chunk_index within a document on re-ingest
    op.create_unique_constraint(
        "uq_chunks_document_id_chunk_index", "chunks", ["document_id", "chunk_index"]
    )


def downgrade() -> None:
    op.drop_constraint("uq_chunks_document_id_chunk_index", "chunks", type_="unique")

    op.drop_index("ix_chunks_content_tsv", table_name="chunks")
    op.drop_column("chunks", "content_tsv")
    op.add_column("chunks", sa.Column("content_tsv", postgresql.TSVECTOR(), nullable=True))
    op.create_index(
        "ix_chunks_content_tsv", "chunks", ["content_tsv"], unique=False, postgresql_using="gin"
    )

    op.drop_index("ix_chunks_source_type", table_name="chunks")
    op.drop_index("ix_chunks_regulation_number", table_name="chunks")
    op.drop_column("chunks", "token_count")
    op.drop_column("chunks", "source_type")
    op.drop_column("chunks", "cross_references")
    op.drop_column("chunks", "section_path")
    op.drop_column("chunks", "clause_title")
    op.drop_column("chunks", "regulation_number")

    op.alter_column(
        "documents", "applicant_id", existing_type=sa.String(length=64), nullable=False
    )
