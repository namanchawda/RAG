"""Pydantic request and response schemas for the API."""

from typing import Any

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    """Schema for a document ingestion request."""

    source_path: str = Field(..., description="Filesystem path or document source identifier.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional metadata for the source document.")


class ChunkDocument(BaseModel):
    """Single chunk representation after parsing and chunking a document."""

    id: int | None = Field(default=None, description="Database identifier for the chunk record.")
    text: str = Field(..., description="Chunk text content.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata attached to the chunk.")


class QueryRequest(BaseModel):
    """Schema for a retrieval and generation question request."""

    question: str = Field(..., min_length=1, description="User question to answer using the indexed filings.")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of relevant chunks to retrieve.")
    filters: dict[str, Any] = Field(default_factory=dict, description="Optional metadata filters for retrieval.")


class RetrievalResult(BaseModel):
    """A single retrieved chunk and its metadata."""

    chunk_id: int | None = Field(default=None, description="Identifier of the chunk in the vector store.")
    text: str = Field(..., description="Retrieved chunk text.")
    score: float | None = Field(default=None, description="Similarity score returned by the vector search.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Chunk metadata as stored in the database.")


class QueryResponse(BaseModel):
    """Schema for the answer returned to the client."""

    answer: str = Field(..., description="Generated answer based on retrieved context.")
    sources: list[RetrievalResult] = Field(default_factory=list, description="Retrieved chunks used as evidence.")


class IngestResponse(BaseModel):
    """Schema for ingestion status responses."""

    status: str = Field(default="success", description="Operation status message.")
    documents_processed: int = Field(default=0, description="Number of documents processed.")
    chunks_stored: int = Field(default=0, description="Number of chunks stored in the vector database.")
    message: str = Field(default="Ingestion complete.", description="Human-readable completion message.")
