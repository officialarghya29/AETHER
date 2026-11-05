"""Database engine and session management.

SQLite is supported for local development and tests; PostgreSQL is the intended deployment
target and is reached by changing ``AETHER_DATABASE_URL`` only.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from backend.app.database.models import Base

_ENGINE: Engine | None = None
_SESSION_FACTORY: sessionmaker[Session] | None = None


def _engine_kwargs(url: str) -> dict:
    if url.startswith("sqlite"):
        return {"connect_args": {"check_same_thread": False}, "pool_pre_ping": True}
    return {"pool_pre_ping": True, "pool_size": 5, "max_overflow": 5}


def init_engine(database_url: str, *, echo: bool = False) -> Engine:
    """Create the process-wide engine and session factory."""
    global _ENGINE, _SESSION_FACTORY
    if _ENGINE is not None:
        return _ENGINE
    if database_url.startswith("sqlite:///") and "memory" not in database_url:
        from pathlib import Path

        Path(database_url.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(database_url, echo=echo, future=True, **_engine_kwargs(database_url))
    if database_url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_connection, _record):  # pragma: no cover - driver hook
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

    _ENGINE = engine
    _SESSION_FACTORY = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    return engine


def get_engine() -> Engine:
    if _ENGINE is None:
        raise RuntimeError("database engine has not been initialised")
    return _ENGINE


def get_session_factory() -> sessionmaker[Session]:
    if _SESSION_FACTORY is None:
        raise RuntimeError("database session factory has not been initialised")
    return _SESSION_FACTORY


def create_all() -> None:
    """Create tables directly (development and tests only; Alembic owns production schema)."""
    Base.metadata.create_all(get_engine())


def drop_all() -> None:
    Base.metadata.drop_all(get_engine())


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope for background jobs and scripts."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a request-scoped session."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engine() -> None:
    """Dispose the engine (used by tests to switch databases)."""
    global _ENGINE, _SESSION_FACTORY
    if _ENGINE is not None:
        _ENGINE.dispose()
    _ENGINE = None
    _SESSION_FACTORY = None
