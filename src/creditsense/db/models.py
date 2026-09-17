from datetime import datetime
from typing import Any
from uuid import uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    DateTime,
    FetchedValue,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Applicant(Base):
    __tablename__ = "applicants"

    id: Mapped[str] = mapped_column(
        String(64), primary_key=True, default=lambda: str(uuid4())
    )
    applicant_id: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, index=True
    )
    sector: Mapped[str] = mapped_column(String(64), nullable=True)
    years_in_business: Mapped[float] = mapped_column(nullable=True)
    documentation_tier: Mapped[str] = mapped_column(String(64), nullable=True)
    raw_profile_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=True)
    # Migration 0004 (P6.6) — synthetic, deterministically generated from applicant_id
    # by db/seed.py, never real. Exist so the MCP masking layer has real identifier
    # fields to redact; the rest of the applicant record is ratios/scores/PKR amounts
    # with nothing else identifier-shaped to mask. NEVER returned unmasked by any tool.
    account_number: Mapped[str] = mapped_column(String(24), nullable=True)
    cnic: Mapped[str] = mapped_column(String(15), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    documents: Mapped[list["Document"]] = relationship(back_populates="applicant")


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(
        String(64), primary_key=True, default=lambda: str(uuid4())
    )
    # Nullable (migration 0002, fix P4.0): a regulation/policy document belongs to no
    # applicant. NOT NULL here originally made it impossible to store the RAG corpus
    # without inventing a fake applicant row — nullable is the cleaner fix. Written as
    # Mapped[str] + nullable=True (not Mapped[str | None]) to match this codebase's
    # existing convention (see Applicant.sector above) — Mapped[X | None] union syntax
    # crashes SQLAlchemy 2.0.36's declarative scan on this Python 3.14 install.
    applicant_id: Mapped[str] = mapped_column(
        ForeignKey("applicants.applicant_id"), nullable=True, index=True
    )
    document_type: Mapped[str] = mapped_column(String(64), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(512), nullable=True)
    content_text: Mapped[str] = mapped_column(Text, nullable=True)
    checksum: Mapped[str] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    applicant: Mapped[Applicant] = relationship(back_populates="documents")
    chunks: Mapped[list["Chunk"]] = relationship(back_populates="document")


class Chunk(Base):
    __tablename__ = "chunks"

    __table_args__ = (
        Index("ix_chunks_content_tsv", "content_tsv", postgresql_using="gin"),
        UniqueConstraint(
            "document_id", "chunk_index", name="uq_chunks_document_id_chunk_index"
        ),
    )

    id: Mapped[str] = mapped_column(
        String(64), primary_key=True, default=lambda: str(uuid4())
    )
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id"), nullable=False, index=True
    )
    chunk_index: Mapped[int] = mapped_column(nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Clause identity (migration 0002, fix P4.0) — without these there is nothing to
    # cite, nothing to filter search_sbp_regulations(regulation_number=...) on, and no
    # way to test multi-hop cross-reference retrieval.
    regulation_number: Mapped[str] = mapped_column(
        String(32), nullable=True, index=True
    )
    clause_title: Mapped[str] = mapped_column(String(256), nullable=True)
    section_path: Mapped[str] = mapped_column(String(256), nullable=True)
    cross_references: Mapped[list[str]] = mapped_column(JSON, nullable=True)
    source_type: Mapped[str] = mapped_column(String(32), nullable=True, index=True)
    token_count: Mapped[int] = mapped_column(Integer, nullable=True)

    # Generated column (migration 0002) — Postgres maintains this from clause_title,
    # regulation_number and content, weighted (title/number above body), so it can
    # never silently drift out of sync the way a plain nullable column did. Never
    # written from Python: server_default=FetchedValue() marks that for SQLAlchemy.
    content_tsv: Mapped[str] = mapped_column(
        TSVECTOR, nullable=True, server_default=FetchedValue()
    )
    embedding: Mapped[list[float]] = mapped_column(Vector(1536), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    document: Mapped[Document] = relationship(back_populates="chunks")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(
        String(64), primary_key=True, default=lambda: str(uuid4())
    )
    actor: Mapped[str] = mapped_column(String(128), nullable=True)
    action: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    resource_type: Mapped[str] = mapped_column(String(128), nullable=True)
    resource_id: Mapped[str] = mapped_column(String(128), nullable=True, index=True)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(64), nullable=False, default="success")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
