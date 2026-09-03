from pathlib import Path

import pytest

from app.ingestion import ingest
from app.ingestion import job_status


def test_ingest_file_labels_extraction_failures(monkeypatch, tmp_path: Path):
    filepath = tmp_path / "broken.pdf"
    filepath.write_bytes(b"%PDF-1.7")
    monkeypatch.setattr(ingest, "load_filing", lambda _: (_ for _ in ()).throw(ValueError("parser failed")))

    with pytest.raises(RuntimeError, match="Failed to extract text from broken.pdf: parser failed"):
        ingest.ingest_file(str(filepath))


def test_dead_process_status_is_marked_failed(tmp_path, monkeypatch):
    status_file = tmp_path / ".ingestion_status.json"
    monkeypatch.setattr(job_status, "STATUS_FILE", status_file)
    job_status.write_status(in_progress=True, process_pid=99999999, stage="embedding")

    import os

    try:
        os.kill(99999999, 0)
    except OSError:
        job_status.write_status(
            in_progress=False,
            stage="failed",
            error="Ingestion process terminated unexpectedly",
        )

    status = job_status.read_status()
    assert status["in_progress"] is False
    assert status["error"] == "Ingestion process terminated unexpectedly"