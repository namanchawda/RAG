"""Embedding utilities for generating vector representations with sentence-transformers.

The embedding model is loaded once at module import so we do not reload it for each
call. This keeps the local embedding workflow efficient for batch document ingestion.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

from app.config import settings

MODEL = SentenceTransformer(settings.EMBEDDING_MODEL)


def load_embedding_model(model_name: str = settings.EMBEDDING_MODEL):
    """Return the singleton sentence-transformers model used for document embeddings."""
    global MODEL
    if MODEL is None or getattr(MODEL, "_model_card_vars", {}).get("model_name") != model_name:
        MODEL = SentenceTransformer(model_name)
    return MODEL


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of text strings and return Python float lists.

    The model is called in a single batch to keep the embedding pipeline efficient.
    """
    if not texts:
        return []

    model = load_embedding_model(settings.EMBEDDING_MODEL)
    embeddings = model.encode(texts, batch_size=32, convert_to_numpy=False, normalize_embeddings=False)

    return [[float(value) for value in vector] for vector in embeddings]


if __name__ == "__main__":
    sample_texts = [
        "Apple reported strong revenue growth in the Services business.",
        "JPMorgan is focused on risk management and capital allocation.",
    ]

    vectors = embed_texts(sample_texts)
    print(f"Embedded {len(vectors)} texts.")
    print(f"Vector length: {len(vectors[0])}")
    print(f"Expected dimension: {settings.EMBEDDING_DIMENSION}")
    print(f"First vector sample: {vectors[0][:5]}")
