import time

from app.ingestion import job_status


def test_status_round_trip_and_stale_job_recovery(tmp_path, monkeypatch):
    status_file = tmp_path / ".ingestion_status.json"
    monkeypatch.setattr(job_status, "STATUS_FILE", status_file)

    job_status.write_status(
        in_progress=True,
        filename="large.pdf",
        stage="embedding",
        current=10,
        total=100,
        started_at=time.time() - job_status.STALE_AFTER_SECONDS - 1,
        error=None,
    )

    status = job_status.mark_stale_if_needed()

    assert status["in_progress"] is False
    assert status["stage"] == "stale"
    assert status["error"]