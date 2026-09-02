"""API endpoints for ingestion and querying the SEC filing knowledge base."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel

from app.generation.rag_pipeline import answer_question
from app.ingestion import store
from app.ingestion.chunker import chunk_text
from app.ingestion.ingest import ingest_file
from app.ingestion.loader import load_filing

DocumentChunk = store.DocumentChunk

router = APIRouter(prefix="/api", tags=["rag"])

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)
SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm", ".txt", ".md", ".rtf"}


class QueryRequest(BaseModel):
    """Request model for question answering via the RAG pipeline."""

    question: str
    source_file: str | None = None
    use_reranking: bool = True
    top_k: int = 5


@router.post("/ingest", status_code=status.HTTP_200_OK)
def ingest_documents(file: UploadFile = File(...)) -> dict:
    """Upload a file to data/raw and ingest it into the vector database."""
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

    destination = RAW_DIR / file.filename
    try:
        contents = file.file.read()
        destination.write_bytes(contents)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {exc}") from exc
    finally:
        file.file.close()

    try:
        store.create_table()
        ingest_file(str(destination))
        chunk_count = len(chunk_text(load_filing(str(destination))))
        return {
            "filename": file.filename,
            "chunks_created": chunk_count,
            "status": "success",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc


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
