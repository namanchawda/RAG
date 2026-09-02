import pytest

from app.ingestion.chunker import chunk_fixed, chunk_recursive, chunk_sentence_aware, chunk_text


def test_fixed_chunk_has_expected_shape():
    chunks = chunk_fixed("alpha beta gamma delta epsilon", chunk_size=8, chunk_overlap=0)
    assert chunks
    assert set(chunks[0].keys()) == {"text", "chunk_id", "token_count"}
    assert isinstance(chunks[0]["chunk_id"], int)
    assert isinstance(chunks[0]["token_count"], int)


def test_sentence_aware_never_splits_sentence():
    text = "This is the first sentence. This is the second sentence. This is the third sentence."
    chunks = chunk_text(text, strategy="sentence_aware", max_tokens=8)
    assert chunks
    assert all(len(chunk["text"].split()) <= 10 for chunk in chunks)


def test_unknown_strategy_raises_value_error():
    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        chunk_text("hello world", strategy="not_real")


def test_recursive_chunking_falls_back_to_fixed_when_needed():
    text = "A very long paragraph with many words. " * 50
    chunks = chunk_recursive(text, max_tokens=20)
    assert chunks
    assert all(chunk["token_count"] <= 20 for chunk in chunks)
