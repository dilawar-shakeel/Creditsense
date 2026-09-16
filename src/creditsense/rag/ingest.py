"""Corpus ingestion pipeline (P4.3): parse -> embed -> upsert.

Run: python -m creditsense.rag.ingest [--dry-run]

--dry-run parses the corpus and reports chunk counts without calling the embedding API
— this is what CI runs, so CI never needs an OpenAI key.

Idempotent: re-running upserts by (document.document_type, chunk.chunk_index) instead
of duplicating rows, using the unique constraint added in migration 0002.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from creditsense.db.models import Chunk, Document
from creditsense.db.session import session_scope
from creditsense.rag.chunking import RegulationChunk, parse_corpus_dir
from creditsense.rag.embed import EmbeddingCache, embed_texts

DEFAULT_CORPUS_DIR = Path("src/creditsense/data/raw_corpus")

# One synthetic Document per source_type, grouping every chunk of that kind. A
# regulation/policy document belongs to no applicant (migration 0002 made
# documents.applicant_id nullable for exactly this).
DOCUMENT_TYPE_BY_SOURCE = {
    "sbp_regulation": "sbp_regulation",
    "internal_policy": "internal_credit_policy",
    "template": "loan_structuring_template",
}


def _get_or_create_document(session: Session, document_type: str) -> Document:
    existing = session.scalar(
        select(Document).where(
            Document.document_type == document_type, Document.applicant_id.is_(None)
        )
    )
    if existing is not None:
        return existing
    document = Document(applicant_id=None, document_type=document_type)
    session.add(document)
    session.flush()
    return document


def _upsert_chunk(
    session: Session,
    document: Document,
    document_chunk_index: int,
    regulation_chunk: RegulationChunk,
    embedding: list[float] | None,
) -> None:
    """document_chunk_index is a position within `document` (0, 1, 2, ...) — distinct
    from regulation_chunk.chunk_index, which only counts sub-clause splits within one
    clause and can repeat across clauses (each clause's own split starts back at 0).
    The (document_id, chunk_index) unique constraint needs the former.
    """
    existing = session.scalar(
        select(Chunk).where(
            Chunk.document_id == document.id, Chunk.chunk_index == document_chunk_index
        )
    )
    if existing is None:
        existing = Chunk(document_id=document.id, chunk_index=document_chunk_index)
        session.add(existing)

    existing.content = regulation_chunk.content
    existing.regulation_number = regulation_chunk.regulation_number
    existing.clause_title = regulation_chunk.clause_title
    existing.section_path = regulation_chunk.section_path
    existing.cross_references = regulation_chunk.cross_references
    existing.source_type = regulation_chunk.source_type
    existing.token_count = regulation_chunk.token_count
    if embedding is not None:
        existing.embedding = embedding


def ingest_corpus(
    corpus_dir: str | Path = DEFAULT_CORPUS_DIR, *, dry_run: bool = False
) -> int:
    """Parse the corpus, embed every chunk (unless dry_run), and upsert to Postgres.

    Returns the number of chunks processed.
    """
    chunks = parse_corpus_dir(corpus_dir)
    counts = Counter(c.source_type for c in chunks)
    print(f"Parsed {len(chunks)} chunks: {dict(counts)}")

    if dry_run:
        print("--dry-run: skipping embedding and database writes")
        return len(chunks)

    cache = EmbeddingCache()
    texts = [c.content for c in chunks]
    embeddings = embed_texts(texts, cache=cache)

    with session_scope() as session:
        documents_by_type = {
            source_type: _get_or_create_document(session, doc_type)
            for source_type, doc_type in DOCUMENT_TYPE_BY_SOURCE.items()
        }
        position_by_document: dict[str, int] = {}
        for chunk, embedding in zip(chunks, embeddings):
            document = documents_by_type[chunk.source_type]
            position = position_by_document.get(document.id, 0)
            _upsert_chunk(session, document, position, chunk, embedding)
            position_by_document[document.id] = position + 1

    print(f"Ingested {len(chunks)} chunks into Postgres")
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the RAG corpus")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and report counts without calling the embedding API or writing to the database",
    )
    parser.add_argument("--corpus-dir", default=str(DEFAULT_CORPUS_DIR))
    args = parser.parse_args()
    ingest_corpus(args.corpus_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
