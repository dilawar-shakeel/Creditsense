"""Create the initial CreditSense database schema.

Revision ID: 20260915_0001
Revises:
Create Date: 2026-09-15
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "20260915_0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "applicants",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("applicant_id", sa.String(length=64), nullable=False),
        sa.Column("sector", sa.String(length=64), nullable=True),
        sa.Column("years_in_business", sa.Float(), nullable=True),
        sa.Column("documentation_tier", sa.String(length=64), nullable=True),
        sa.Column("raw_profile_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("applicant_id"),
    )
    op.create_index("ix_applicants_applicant_id", "applicants", ["applicant_id"], unique=False)

    op.create_table(
        "documents",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("applicant_id", sa.String(length=64), nullable=False),
        sa.Column("document_type", sa.String(length=64), nullable=False),
        sa.Column("storage_path", sa.String(length=512), nullable=True),
        sa.Column("content_text", sa.Text(), nullable=True),
        sa.Column("checksum", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["applicant_id"], ["applicants.applicant_id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_documents_applicant_id", "documents", ["applicant_id"], unique=False)

    op.create_table(
        "chunks",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("document_id", sa.String(length=64), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_tsv", postgresql.TSVECTOR(), nullable=True),
        sa.Column("embedding", Vector(1536), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"], unique=False)
    op.create_index(
        "ix_chunks_content_tsv",
        "chunks",
        ["content_tsv"],
        unique=False,
        postgresql_using="gin",
    )

    op.create_table(
        "audit_logs",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("actor", sa.String(length=128), nullable=True),
        sa.Column("action", sa.String(length=128), nullable=False),
        sa.Column("resource_type", sa.String(length=128), nullable=True),
        sa.Column("resource_id", sa.String(length=128), nullable=True),
        sa.Column("payload_json", sa.JSON(), nullable=True),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_audit_logs_action", "audit_logs", ["action"], unique=False)
    op.create_index("ix_audit_logs_resource_id", "audit_logs", ["resource_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_audit_logs_resource_id", table_name="audit_logs")
    op.drop_index("ix_audit_logs_action", table_name="audit_logs")
    op.drop_table("audit_logs")
    op.drop_index("ix_chunks_content_tsv", table_name="chunks")
    op.drop_index("ix_chunks_document_id", table_name="chunks")
    op.drop_table("chunks")
    op.drop_index("ix_documents_applicant_id", table_name="documents")
    op.drop_table("documents")
    op.drop_index("ix_applicants_applicant_id", table_name="applicants")
    op.drop_table("applicants")
    op.execute("DROP EXTENSION IF EXISTS vector")
