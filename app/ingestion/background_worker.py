"""Standalone multiprocessing target for durable document ingestion."""

from __future__ import annotations

import re

from app.ingestion import store
from app.ingestion.ingest import ingest_file
from app.ingestion.job_status import read_status, write_status


def run_ingestion_job(
    filepath: str,
    filename: str,
    chunking_strategy: str,
    database_url: str,
) -> None:
    """Run ingestion in a child process without importing the Streamlit app."""
    store.init_engine(database_url)

    def update_progress(message: str, progress: float) -> None:
        stage = re.sub(r"\s*\(\d+%\)\s*$", "", message.split(":", 1)[0]).strip().rstrip(".")
        current = 0
        total = 0
        if stage == "Embedding" and "/" in message:
            counts = message.split(":", 1)[1].split("(", 1)[0].strip().split("/")
            current = int(counts[0])
            total = int(counts[1].split()[0])
        write_status(stage=stage.lower(), current=current, total=total)

    try:
        chunk_count = ingest_file(
            filepath,
            chunking_strategy=chunking_strategy,
            progress_callback=update_progress,
        )
        write_status(
            in_progress=False,
            filename=filename,
            stage="complete",
            current=chunk_count,
            total=chunk_count,
            chunk_count=chunk_count,
            error=None,
        )
    except Exception as exc:  # pragma: no cover - rendered by the Streamlit monitor
        write_status(in_progress=False, filename=filename, stage="failed", error=str(exc))
