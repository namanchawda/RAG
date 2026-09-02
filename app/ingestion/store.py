"""Storage layer for persisting embedded document chunks to pgvector.

This module creates the vector-retrieval table and inserts chunk rows in batches to
keep ingestion fast and efficient for the SEC filing RAG baseline.
"""

from __future__ import annotations

from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import Computed, Index, Integer, Text, create_engine
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from app.config import settings


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""


class DocumentChunk(Base):
    """A single document chunk stored alongside its embedding vector."""

    __tablename__ = settings.VECTOR_TABLE
    __table_args__ = (
        Index(
            f"ix_{settings.VECTOR_TABLE}_search_vector",
            "search_vector",
            postgresql_using="gin",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_file: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_id: Mapped[int] = mapped_column(Integer, nullable=False)
    chunking_strategy: Mapped[str | None] = mapped_column(Text, nullable=True, server_default="fixed")
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(settings.EMBEDDING_DIMENSION), nullable=False)
    search_vector: Mapped[str | None] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', chunk_text)", persisted=True),
        nullable=True,
    )


engine: Any = None
SessionLocal: Any = None


def _default_database_url() -> str:
    """Build the default Postgres URL from the environment settings."""
    return (
        f"postgresql+psycopg://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
        f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    )


def init_engine(database_url: str) -> None:
    """Initialize the module-level SQLAlchemy engine and session factory.

    The caller passes the full database URL directly (for example a NeonDB URL),
    which makes the app able to switch DB connection at runtime without reloading
    a module or editing environment values.
    """
    global engine, SessionLocal
    engine = create_engine(database_url, pool_pre_ping=True)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def _ensure_initialized() -> None:
    """Initialize the default engine lazily when the module is used without UI setup."""
    global engine, SessionLocal
    if engine is None or SessionLocal is None:
        init_engine(_default_database_url())


def ensure_search_vector_column() -> None:
    """Ensure the generated search_vector column and GIN index exist.

    Existing databases may already have the table without the new full-text column.
    In that case we add the migration explicitly instead of silently leaving the
    table in an inconsistent state.
    """
    _ensure_initialized()
    with engine.begin() as conn:
        result = conn.exec_driver_sql(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_name = %s AND column_name = 'search_vector'
            );
            """,
            (settings.VECTOR_TABLE,),
        )
        exists = result.scalar()

    if exists:
        return

    print(
        f"Warning: table '{settings.VECTOR_TABLE}' is missing the generated search_vector column. "
        "Existing rows may need a migration before keyword search can be used."
    )

    with engine.begin() as conn:
        conn.exec_driver_sql(
            f"""
            ALTER TABLE {settings.VECTOR_TABLE}
            ADD COLUMN search_vector tsvector
            GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED;
            """
        )
        conn.exec_driver_sql(
            f"""
            CREATE INDEX IF NOT EXISTS ix_{settings.VECTOR_TABLE}_search_vector
            ON {settings.VECTOR_TABLE} USING GIN (search_vector);
            """
        )

    print(f"Migration complete: added search_vector and GIN index for '{settings.VECTOR_TABLE}'.")


def ensure_chunking_strategy_column() -> None:
    """Ensure the chunking strategy metadata column exists and backfills old rows."""
    _ensure_initialized()

    with engine.begin() as conn:
        result = conn.exec_driver_sql(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_name = %s AND column_name = 'chunking_strategy'
            );
            """,
            (settings.VECTOR_TABLE,),
        )
        exists = result.scalar()

    if not exists:
        with engine.begin() as conn:
            conn.exec_driver_sql(
                f"ALTER TABLE {settings.VECTOR_TABLE} ADD COLUMN chunking_strategy TEXT DEFAULT 'fixed';"
            )

    with engine.begin() as conn:
        conn.exec_driver_sql(
            f"UPDATE {settings.VECTOR_TABLE} SET chunking_strategy = 'fixed' WHERE chunking_strategy IS NULL;"
        )

    print(f"Migration complete: ensured chunking_strategy column for '{settings.VECTOR_TABLE}'.")


def create_table() -> None:
    """Create the vector table and the search_vector migration if needed."""
    _ensure_initialized()
    Base.metadata.create_all(bind=engine)
    ensure_search_vector_column()
    ensure_chunking_strategy_column()


def store_chunks(
    source_file: str,
    chunks: list[dict],
    embeddings: list[list[float]],
    chunking_strategy: str = "fixed",
) -> None:
    """Insert chunk texts and embeddings into the vector table in a single batch."""
    _ensure_initialized()
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must be the same length")

    if not chunks:
        return

    with SessionLocal() as session:
        rows = [
            DocumentChunk(
                source_file=source_file,
                chunk_id=chunk["chunk_id"],
                chunking_strategy=chunking_strategy,
                chunk_text=chunk["text"],
                token_count=chunk["token_count"],
                embedding=embedding,
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]
        session.add_all(rows)
        session.commit()


if __name__ == "__main__":
    _ensure_initialized()
    create_table()
    print(f"Ensured table '{settings.VECTOR_TABLE}' exists with vector dimension {settings.EMBEDDING_DIMENSION}.")
