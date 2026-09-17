"""Database session factory.

Nothing in the codebase could talk to Postgres before this file existed —
db/__init__.py only re-exported the ORM models. Every RAG/ingest/retrieval script and
every FastAPI route that needs a database session goes through here.
"""

from __future__ import annotations

from collections.abc import Generator, Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from creditsense.config import get_settings

_engine = None
_SessionLocal: sessionmaker[Session] | None = None


def get_engine():
    """Lazily build a single module-level engine from Settings.database_url.

    Lazy so importing this module never requires DATABASE_URL to already be
    resolvable (e.g. during collection of tests that never touch the database).
    """
    global _engine
    if _engine is None:
        _engine = create_engine(get_settings().database_url, pool_pre_ping=True)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=get_engine(), autoflush=False, expire_on_commit=False
        )
    return _SessionLocal


@contextmanager
def session_scope() -> Iterator[Session]:
    """Use in scripts: `with session_scope() as session: ...`.

    Commits on a clean exit, rolls back and re-raises on an exception, always closes.
    """
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: `session: Session = Depends(get_db)`."""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()
