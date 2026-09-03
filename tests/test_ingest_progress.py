from pathlib import Path

from app.ingestion import ingest


def test_ingest_file_reports_progress_stages(monkeypatch, tmp_path: Path):
    filepath = tmp_path / "sample.txt"
    filepath.write_text("A short filing sentence.", encoding="utf-8")
    progress = []

    monkeypatch.setattr(ingest, "load_filing", lambda _: "A short filing sentence.")
    monkeypatch.setattr(ingest, "chunk_text", lambda text, strategy: [{"text": text, "chunk_id": 0, "token_count": 4}])
    monkeypatch.setattr(ingest, "embed_texts", lambda texts: [[0.1, 0.2]])
    monkeypatch.setattr(ingest.store, "store_chunks", lambda *args, **kwargs: None)

    ingest.ingest_file(str(filepath), progress_callback=lambda message, value: progress.append((message, value)))

    assert [message for message, _ in progress] == [
        "Loading document...",
        "Chunking...",
        "Embedding 1/1 chunks...",
        "Storing...",
        "Complete",
    ]
    assert progress[-1][1] == 1.0