"""Chunking utilities for splitting SEC filing text into manageable retrieval units."""

from __future__ import annotations

import re

import tiktoken


def _token_count(text: str) -> int:
    """Return the token count for a string using the OpenAI cl100k_base encoding."""
    if not text or not text.strip():
        return 0
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def _make_chunk(text: str, chunk_id: int) -> dict:
    """Normalize a chunk payload to the shared record shape."""
    cleaned = text.strip()
    return {"text": cleaned, "chunk_id": chunk_id, "token_count": _token_count(cleaned)}


def chunk_fixed(text: str, chunk_size: int = 512, chunk_overlap: int = 50) -> list[dict]:
    """Split text into fixed-size token windows with overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    if not text or not text.strip():
        return []

    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)

    step_size = chunk_size - chunk_overlap
    chunks: list[dict] = []
    start = 0
    chunk_id = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text_value = encoding.decode(chunk_tokens)
        chunks.append(
            {
                "text": chunk_text_value.strip(),
                "chunk_id": chunk_id,
                "token_count": len(chunk_tokens),
            }
        )
        if end == len(tokens):
            break
        start += step_size
        chunk_id += 1

    return chunks


def _split_sentences(text: str) -> list[str]:
    """Split text into sentence-sized units while preserving the original wording."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"(?<=[.!?])\s+|\n\s*\n+", normalized)
    return [part.strip() for part in parts if part and part.strip()]


def chunk_sentence_aware(text: str, max_tokens: int = 512) -> list[dict]:
    """Group whole sentences into chunks without splitting any sentence across chunks."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero")

    if not text or not text.strip():
        return []

    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[dict] = []
    current = ""
    chunk_id = 0

    for sentence in sentences:
        sentence_tokens = _token_count(sentence)

        if sentence_tokens > max_tokens:
            if current:
                chunks.append(_make_chunk(current, chunk_id))
                chunk_id += 1
                current = ""

            long_sentence_chunks = chunk_fixed(sentence, chunk_size=max_tokens, chunk_overlap=0)
            for long_chunk in long_sentence_chunks:
                chunks.append(
                    {
                        "text": long_chunk["text"].strip(),
                        "chunk_id": chunk_id,
                        "token_count": long_chunk["token_count"],
                    }
                )
                chunk_id += 1
            continue

        candidate = sentence if not current else f"{current} {sentence}"
        if current and _token_count(candidate) > max_tokens:
            chunks.append(_make_chunk(current, chunk_id))
            chunk_id += 1
            current = sentence
        else:
            current = candidate

    if current:
        chunks.append(_make_chunk(current, chunk_id))

    return chunks


def chunk_paragraph_based(text: str, max_tokens: int = 512) -> list[dict]:
    """Split text by paragraph first, falling back to sentence-aware chunking for oversized paragraphs."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero")

    if not text or not text.strip():
        return []

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text.strip()) if part and part.strip()]
    if not paragraphs:
        return []

    chunks: list[dict] = []
    chunk_id = 0

    for paragraph in paragraphs:
        if _token_count(paragraph) <= max_tokens:
            chunks.append(_make_chunk(paragraph, chunk_id))
            chunk_id += 1
            continue

        for sentence_chunk in chunk_sentence_aware(paragraph, max_tokens=max_tokens):
            chunks.append(
                {
                    "text": sentence_chunk["text"],
                    "chunk_id": chunk_id,
                    "token_count": sentence_chunk["token_count"],
                }
            )
            chunk_id += 1

    return chunks


def chunk_recursive(text: str, max_tokens: int = 512) -> list[dict]:
    """Try paragraph, then sentence, then fixed-size chunking in order of priority."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero")

    if not text or not text.strip():
        return []

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text.strip()) if part and part.strip()]
    if not paragraphs:
        return []

    chunks: list[dict] = []
    chunk_id = 0

    for paragraph in paragraphs:
        if _token_count(paragraph) <= max_tokens:
            chunks.append(_make_chunk(paragraph, chunk_id))
            chunk_id += 1
            continue

        sentence_chunks = chunk_sentence_aware(paragraph, max_tokens=max_tokens)
        if sentence_chunks and all(_token_count(candidate["text"]) <= max_tokens for candidate in sentence_chunks):
            for candidate in sentence_chunks:
                chunks.append(
                    {
                        "text": candidate["text"],
                        "chunk_id": chunk_id,
                        "token_count": candidate["token_count"],
                    }
                )
                chunk_id += 1
            continue

        for fixed_chunk in chunk_fixed(paragraph, chunk_size=max_tokens, chunk_overlap=0):
            chunks.append(
                {
                    "text": fixed_chunk["text"],
                    "chunk_id": chunk_id,
                    "token_count": fixed_chunk["token_count"],
                }
            )
            chunk_id += 1

    return chunks


def chunk_text(text: str, strategy: str = "fixed", **kwargs) -> list[dict]:
    """Dispatch to the selected chunking strategy and return consistent chunk dictionaries."""
    strategy_map = {
        "fixed": chunk_fixed,
        "sentence_aware": chunk_sentence_aware,
        "paragraph_based": chunk_paragraph_based,
        "recursive": chunk_recursive,
    }

    if strategy not in strategy_map:
        valid = ", ".join(sorted(strategy_map))
        raise ValueError(f"Unknown chunking strategy '{strategy}'. Valid strategies: {valid}")

    return strategy_map[strategy](text, **kwargs)


if __name__ == "__main__":
    sample_text = (
        "Management discussed revenue growth and operating leverage across the business. "
        "The company increased investment in product development, enterprise software, and "
        "international expansion. Risk factors include inflation, supply chain constraints, "
        "cybersecurity exposure, and competition. The company also continued to focus on "
        "cash generation and disciplined capital allocation. "
        * 8
    )

    chunks = chunk_text(sample_text, strategy="fixed", chunk_size=128, chunk_overlap=20)
    print(f"Total chunks: {len(chunks)}")
    for chunk in chunks[:3]:
        print(f"Chunk {chunk['chunk_id']} | Tokens: {chunk['token_count']}")
        print(chunk["text"][:200].replace("\n", " "))
        print("-" * 60)
