"""Quick inspection script for the naive SEC filing chunking pipeline.

Run with:
    python -m app.ingestion.test_pipeline
"""

from __future__ import annotations

from pathlib import Path

from app.ingestion.chunker import chunk_text
from app.ingestion.loader import load_filing


def inspect_filing(filepath: str, sample_indices: list[int]) -> None:
    """Load a filing, split it into chunks, and print a concise sampling of chunk content."""
    text = load_filing(filepath)
    chunks = chunk_text(text)

    print(f"\n=== FILE: {filepath} ===")
    print(f"Total chunks produced: {len(chunks)}")

    for idx in sample_indices:
        if idx < 0 or idx >= len(chunks):
            print(f"Chunk index {idx} is out of range for this filing.")
            continue

        chunk = chunks[idx]
        print(f"\n--- Chunk index {idx} ---")
        print(f"token_count: {chunk['token_count']}")
        print(chunk["text"][:2000])
        print("-" * 80)

    print("\nToken counts by chunk index:")
    for chunk in chunks[:5]:
        print(f"chunk_id={chunk['chunk_id']}, token_count={chunk['token_count']}")


if __name__ == "__main__":
    jpm_path = "data/raw/jpm_10k_2025.html"
    aapl_path = "data/raw/aapl_10k_2025.html"

    for filing_path in [jpm_path, aapl_path]:
        raw_path = Path(filing_path)
        if not raw_path.exists():
            print(f"Skipping {filing_path}: file not found. Download it first.")
            continue

        # Sample a few chunks spread across the document to inspect different portions.
        sample_indices = [50, 150, 300]
        inspect_filing(filing_path, sample_indices)
