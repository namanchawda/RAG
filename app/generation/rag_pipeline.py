"""Retrieval + generation orchestration for the SEC filing RAG baseline."""

from __future__ import annotations

from collections.abc import Iterator

from app.generation.llm_client import generate_answer, generate_answer_stream
from app.generation.prompt_builder import build_prompt
from app.ingestion import store
from app.retrieval.hybrid_search import hybrid_search
from app.retrieval.reranker import rerank


def prepare_answer(
    query: str,
    top_k: int | None = None,
    source_file: str | None = None,
    chunking_strategy: str | None = None,
    use_reranking: bool = True,
) -> dict:
    """Retrieve and prepare grounded context before either buffered or streamed generation."""
    store._ensure_initialized()

    if top_k is None:
        top_k = 5

    wide_candidates = hybrid_search(
        query=query,
        top_k=20,
        source_file=source_file,
        chunking_strategy=chunking_strategy,
    )
    if use_reranking:
        retrieved = rerank(query=query, candidates=wide_candidates, top_k=top_k)
    else:
        retrieved = wide_candidates[:top_k]

    return {
        "prompt": build_prompt(query=query, retrieved_chunks=retrieved),
        "retrieved": retrieved,
        "reranking_used": use_reranking,
    }


def answer_question_stream(
    query: str,
    top_k: int | None = None,
    source_file: str | None = None,
    chunking_strategy: str | None = None,
    use_reranking: bool = True,
):
    """Yield the generated answer after retrieval and reranking complete."""
    prepared = prepare_answer(
        query=query,
        top_k=top_k,
        source_file=source_file,
        chunking_strategy=chunking_strategy,
        use_reranking=use_reranking,
    )
    yield from generate_answer_stream(prepared["prompt"])


def stream_prepared_answer(prepared: dict) -> Iterator[str]:
    """Stream a previously prepared answer without repeating retrieval or reranking."""
    yield from generate_answer_stream(prepared["prompt"])


def answer_question(
    query: str,
    top_k: int | None = None,
    source_file: str | None = None,
    chunking_strategy: str | None = None,
    use_reranking: bool = True,
) -> dict:
    """Run retrieval on the query, build a grounded prompt, and generate an answer."""
    prepared = prepare_answer(
        query=query,
        top_k=top_k,
        source_file=source_file,
        chunking_strategy=chunking_strategy,
        use_reranking=use_reranking,
    )
    retrieved = prepared["retrieved"]
    answer = generate_answer(prepared["prompt"])

    return {
        "query": query,
        "answer": answer,
        "sources": [
            {
                "source_file": row["source_file"],
                "chunk_id": row["chunk_id"],
                "distance": row.get("distance", row.get("rank_score", 0.0)),
                "rerank_score": row.get("rerank_score"),
            }
            for row in retrieved
        ],
        "reranking_used": use_reranking,
    }


if __name__ == "__main__":
    questions = [
        "What are Apple's main risk factors?",
        "What is JPMorgan's approach to interest rate risk?",
        "How does Apple describe its operating segments?",
    ]

    for question in questions:
        try:
            result = answer_question(question, top_k=3)
            print(f"\nQuery: {question}")
            print(f"Answer: {result['answer']}")
            print(f"Sources: {result['sources']}")
        except Exception as exc:
            print(f"\nQuery: {question}")
            print(f"Error: {exc}")
