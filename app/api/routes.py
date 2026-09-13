"""API endpoints for ingestion and querying the SEC filing knowledge base."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel

from app.generation.rag_pipeline import answer_question
from app.ingestion import store
from app.ingestion.ingest import ingest_file

DocumentChunk = store.DocumentChunk

router = APIRouter(prefix="/api", tags=["rag"])

SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm", ".txt", ".md", ".rtf"}
MAX_UPLOAD_SIZE_BYTES = 25 * 1024 * 1024


class QueryRequest(BaseModel):
    """Request model for question answering via the RAG pipeline."""

    question: str
    source_file: str | None = None
    use_reranking: bool = False
    top_k: int = 5


@router.post("/ingest", status_code=status.HTTP_200_OK)
def ingest_documents(file: UploadFile = File(...)) -> dict:
    """Ingest an uploaded file into the vector database without persisting it in the repo.

    The file is written to a temporary location (outside data/raw), used only for
    the duration of the ingestion pipeline, and deleted immediately afterward
    regardless of success or failure.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="A file upload is required.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: {file.filename}. "
                f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            ),
        )

    try:
        file.file.seek(0, os.SEEK_END)
        upload_size = file.file.tell()
        file.file.seek(0)
    except Exception as exc:
        file.file.close()
        raise HTTPException(status_code=500, detail=f"Failed to inspect uploaded file: {exc}") from exc

    if upload_size > MAX_UPLOAD_SIZE_BYTES:
        file.file.close()
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File is too large. Please upload a document smaller than 25 MB.",
        )

    try:
        contents = file.file.read()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read uploaded file: {exc}") from exc
    finally:
        file.file.close()

    # NamedTemporaryFile(delete=False) so we control exactly when it's removed;
    # we always clean it up in the finally block below.
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        temp_path = tmp.name
    contents = None

    try:
        store.create_table()
        # ingest_file / load_filing use the file's path and extension, not its
        # original name, so a temp path with the same suffix works the same way.
        # source_file is recorded using the original uploaded filename so it still
        # displays correctly and is retrievable later.
        chunk_count = ingest_file(temp_path, source_file=file.filename)
        return {
            "filename": file.filename,
            "chunks_created": chunk_count,
            "status": "success",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


@router.post("/query", status_code=status.HTTP_200_OK)
def query_documents(payload: QueryRequest) -> dict:
    """Answer a question using hybrid search and optional reranking."""
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    result = answer_question(
        query=payload.question,
        top_k=payload.top_k,
        source_file=payload.source_file,
        use_reranking=payload.use_reranking,
    )

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "reranking_used": result["reranking_used"],
    }


@router.get("/documents", status_code=status.HTTP_200_OK)
def list_documents() -> list[str]:
    """Return the distinct source_file values currently stored in the vector database."""
    with store.SessionLocal() as session:
        rows = session.query(DocumentChunk.source_file).distinct().order_by(DocumentChunk.source_file).all()

    return [row[0] for row in rows]
