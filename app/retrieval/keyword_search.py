"""Keyword-based full-text search over document chunks using PostgreSQL tsquery."""

from __future__ import annotations

from sqlalchemy import func, select

from app.config import settings
from app.ingestion import store

DocumentChunk = store.DocumentChunk


def keyword_search(
    query: str,
    top_k: int = 10,
    source_file: str | None = None,
    chunking_strategy: str | None = None,
) -> list[dict]:
    """Return the most relevant chunks for a keyword query using PostgreSQL full-text search.

    Uses plainto_tsquery('english', query) to rank chunk_text by textual similarity and
    returns the same record structure used by the vector search layer so both can be merged.
    """
    if top_k is None:
        top_k = settings.QUERY_TOP_K

    ts_query = func.plainto_tsquery("english", query)
    ts_rank = func.ts_rank(DocumentChunk.search_vector, ts_query)

    stmt = (
        select(
            DocumentChunk.chunk_text.label("chunk_text"),
            DocumentChunk.source_file.label("source_file"),
            DocumentChunk.chunk_id.label("chunk_id"),
            ts_rank.label("rank_score"),
        )
        .where(DocumentChunk.search_vector.isnot(None))
        .where(DocumentChunk.search_vector.op("@@")(ts_query))
        .order_by(ts_rank.desc())
        .limit(top_k)
    )

    if source_file is not None:
        stmt = stmt.where(DocumentChunk.source_file == source_file)
    if chunking_strategy is not None:
        stmt = stmt.where(DocumentChunk.chunking_strategy == chunking_strategy)

    with store.SessionLocal() as session:
        rows = session.execute(stmt).mappings().all()

    return [
        {
            "chunk_text": row["chunk_text"],
            "source_file": row["source_file"],
            "chunk_id": row["chunk_id"],
            "rank_score": float(row["rank_score"]),
        }
        for row in rows
    ]


if __name__ == "__main__":
    results = keyword_search("Apple risk factors", top_k=5)
    for row in results:
        print(f"source_file={row['source_file']} chunk_id={row['chunk_id']} rank_score={row['rank_score']}")
        print(row["chunk_text"][:200])
        print("---")
