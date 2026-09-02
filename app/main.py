"""FastAPI application entrypoint for the SEC filing RAG service."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router

app = FastAPI(
    title="SEC Filing RAG",
    description="Naive RAG baseline for retrieving and answering questions from SEC 10-K filings.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return a simple health check response for the service."""
    return {"status": "ok"}
