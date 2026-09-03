from pathlib import Path

from app.ingestion import ingest


def test_ingest_file_reports_progress_stages(monkeypatch, tmp_path: Path):
    filepath = tmp_path / "sample.txt"
    filepath.write_text("A short filing sentence.", encoding="utf-8")
    progress = []

    monkeypatch.setattr(ingest, "load_filing", lambda _: "A short filing sentence.")
    monkeypatch.setattr(ingest, "chunk_text", lambda text, strategy: [{"text": text, "chunk_id": 0, "token_count": 4}])
    monkeypatch.setattr(
        ingest,
        "embed_texts",
        lambda texts, progress_callback=None: (
            progress_callback(len(texts), len(texts)) if progress_callback else None
        ) or [[0.1, 0.2]],
    )
    monkeypatch.setattr(ingest.store, "store_chunks", lambda *args, **kwargs: None)

    ingest.ingest_file(
        str(filepath),
        progress_callback=lambda message, value: progress.append((message, value)),
        extracted_text="A short filing sentence.",
    )

    assert [message for message, _ in progress] == [
        "Loading document... (5%)",
        "Chunking... (20%)",
        "Embedding: 1/1 chunks (100%)",
        "Storing... (95%)",
        "Complete (100%)",
    ]
    assert progress[-1][1] == 1.0