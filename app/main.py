"""FastAPI application entrypoint for the SEC filing RAG service."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.ingestion import store

app = FastAPI(
    title="RAG Chatbot",
    description="Naive RAG baseline for retrieving and answering questions from an uploaded file only"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        # TODO: add your Vercel URL here once deployed, e.g.
        # "https://rag-single-doc-chat.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router)


@app.on_event("startup")
def initialize_application() -> None:
    """Initialize the single environment-backed database connection at startup."""
    store.init_engine()
    store.create_table()


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return a simple health check response for the service."""
    return {"status": "ok"}
