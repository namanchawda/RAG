"""Streamlit UI for the SEC filing RAG project."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.config import settings
from app.generation.rag_pipeline import prepare_answer, stream_prepared_answer
from app.ingestion import store
from app.ingestion.ingest import ingest_file

DocumentChunk = store.DocumentChunk

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)
CONNECTION_FILE = Path(__file__).resolve().parents[2] / ".streamlit_connection.json"
AVAILABLE_GROQ_MODELS = ["openai/gpt-oss-120b"]
CHUNKING_STRATEGY_LABELS = {
    "fixed": "Fixed-size",
    "sentence_aware": "Sentence-aware",
    "paragraph_based": "Paragraph-based",
    "recursive": "Recursive (paragraph → sentence → fixed)",
}


def format_strategy_label(strategy: str) -> str:
    """Return a human-friendly label for a chunking strategy key."""
    return CHUNKING_STRATEGY_LABELS.get(strategy, strategy)


def save_connection_details(connection_string: str, groq_api_key: str, groq_model: str) -> None:
    """Persist the active DB and Groq settings to a local JSON file."""
    payload = {
        "connection_string": connection_string,
        "groq_api_key": groq_api_key,
        "groq_model": groq_model,
    }
    CONNECTION_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def clear_connection_details() -> None:
    """Remove any persisted DB/Groq connection cache."""
    if CONNECTION_FILE.exists():
        CONNECTION_FILE.unlink()


def get_available_documents() -> list[dict]:
    """Return distinct (source_file, chunking_strategy) combinations currently ingested in the DB."""
    if store.SessionLocal is None:
        return []
    with store.SessionLocal() as session:
        rows = (
            session.query(DocumentChunk.source_file, DocumentChunk.chunking_strategy)
            .distinct()
            .order_by(DocumentChunk.source_file, DocumentChunk.chunking_strategy)
            .all()
        )

    return [
        {
            "source_file": source_file,
            "chunking_strategy": chunking_strategy or "fixed",
            "label": f"{source_file} — {format_strategy_label(chunking_strategy or 'fixed')}",
        }
        for source_file, chunking_strategy in rows
    ]


st.set_page_config(page_title="RAG", page_icon="📄", layout="wide")

if "db_connected" not in st.session_state:
    st.session_state["db_connected"] = False

if not st.session_state.get("db_connected") and CONNECTION_FILE.exists():
    try:
        payload = json.loads(CONNECTION_FILE.read_text(encoding="utf-8"))
        connection_string = payload.get("connection_string", "").strip()
        groq_api_key = payload.get("groq_api_key", "").strip()
        groq_model = payload.get("groq_model", AVAILABLE_GROQ_MODELS[0]).strip()

        if connection_string and groq_api_key:
            normalized_url = connection_string
            if normalized_url.startswith("postgresql://"):
                normalized_url = "postgresql+psycopg://" + normalized_url[len("postgresql://") :]
            else:
                normalized_url = normalized_url.replace("postgresql://", "postgresql+psycopg://", 1)

            store.init_engine(normalized_url)
            store.create_table()
            settings.GROQ_API_KEY = groq_api_key
            settings.GROQ_MODEL = groq_model
            st.session_state["db_connected"] = True
    except Exception as exc:
        clear_connection_details()
        st.error(f"Saved connection failed: {exc}")


if not st.session_state.get("db_connected"):
    st.title("Connect to your database")
    connection_string = st.text_input(
        "NeonDB or Postgres connection string",
        type="password",
        placeholder="postgresql://user:pass@host/dbname?sslmode=require&channel_binding=require",
    )
    groq_api_key = st.text_input("GROQ_API_KEY", type="password", placeholder="Enter your Groq API key")
    groq_model = st.selectbox("GROQ_MODEL", options=AVAILABLE_GROQ_MODELS, index=0)

    if st.button("Connect"):
        missing = []
        if not connection_string.strip():
            missing.append("Postgres connection string")
        if not groq_api_key.strip():
            missing.append("GROQ_API_KEY")

        if missing:
            st.error(f"Missing required value(s): {', '.join(missing)}")
        else:
            try:
                normalized_url = connection_string.strip()
                if normalized_url.startswith("postgresql://"):
                    normalized_url = "postgresql+psycopg://" + normalized_url[len("postgresql://") :]
                else:
                    normalized_url = normalized_url.replace("postgresql://", "postgresql+psycopg://", 1)

                store.init_engine(normalized_url)
                store.create_table()
                settings.GROQ_API_KEY = groq_api_key.strip()
                settings.GROQ_MODEL = groq_model
                save_connection_details(normalized_url, settings.GROQ_API_KEY, settings.GROQ_MODEL)
                st.session_state["db_connected"] = True
                st.rerun()
            except Exception as exc:
                message = str(exc)
                if 'type "vector" does not exist' in message:
                    st.error(
                        'Your database doesn\'t have the pgvector extension enabled yet. Go to your '
                        'Neon dashboard\'s SQL Editor and run: CREATE EXTENSION IF NOT EXISTS vector; '
                        '— then try connecting again.'
                    )
                else:
                    st.error(f"Connection failed: {message}")

    with st.expander("New here? Setup instructions", expanded=False):
        st.markdown(
            """
            ## Setting up NeonDB (free Postgres + pgvector)
            1. Go to https://neon.tech and sign up (no credit card required)
            2. Click "Create a project", give it any name, choose a nearby region
            3. Open the SQL Editor tab and run: `CREATE EXTENSION IF NOT EXISTS vector;`
            4. Go to Connection Details, select "Pooled connection", and copy the full connection string (starts with `postgresql://`)

            ## Getting a Groq API key (free)
            1. Go to https://console.groq.com and sign up (no credit card required)
            2. Go to "API Keys" in the sidebar, click "Create API Key"
            3. Copy the key immediately — it's only shown once
            """
        )
    st.stop()


st.title("RAG")

selected_source = None

with st.sidebar:
    if st.button("Disconnect"):
        st.session_state["db_connected"] = False
        clear_connection_details()
        st.rerun()

    st.header("Ingest a document")
    uploaded_file = st.file_uploader(
        "Upload a PDF or HTML filing",
        type=["pdf", "html", "htm"],
        accept_multiple_files=False,
    )
    chunking_strategy = st.selectbox(
        "Chunking strategy",
        options=list(CHUNKING_STRATEGY_LABELS.keys()),
        format_func=lambda strategy: CHUNKING_STRATEGY_LABELS[strategy],
        index=0,
    )

    if st.button("Ingest"):
        if uploaded_file is None:
            st.warning("Please upload a PDF or HTML file before ingesting.")
        else:
            destination = RAW_DIR / uploaded_file.name
            try:
                with st.spinner("Saving and ingesting document..."):
                    destination.write_bytes(uploaded_file.getvalue())
                    ingest_file(str(destination), chunking_strategy=chunking_strategy)
                st.success(
                    f"Ingested {uploaded_file.name} successfully using '{CHUNKING_STRATEGY_LABELS[chunking_strategy]}'."
                )
            except Exception as exc:
                st.error(f"Ingestion failed: {exc}")

    st.header("Document filter")
    documents = get_available_documents()
    placeholder_label = "-- Select a document --"
    document_options = [{"source_file": "", "chunking_strategy": "", "label": placeholder_label}] + documents
    selected_doc = st.selectbox(
        "Choose a document",
        options=document_options,
        format_func=lambda option: option["label"],
        index=0,
    )
    selected_source = selected_doc.get("source_file") if selected_doc and selected_doc.get("source_file") else None
    selected_chunking_strategy = (
        selected_doc.get("chunking_strategy") if selected_doc and selected_doc.get("chunking_strategy") else None
    )

    st.caption(f"Active Groq model: {settings.GROQ_MODEL}")

if selected_source is None:
    st.info("Select a document from the sidebar to start asking questions.")
else:
    st.header("Ask a question")
    use_reranking = st.toggle("Use reranking", value=True)
    question = st.text_input("Question", placeholder="Example: What are Apple's main risk factors?")

    if st.button("Ask"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving context..."):
                prepared = prepare_answer(
                    query=question,
                    top_k=5,
                    source_file=selected_source,
                    chunking_strategy=selected_chunking_strategy,
                    use_reranking=use_reranking,
                )

            st.markdown("### Answer")
            st.write_stream(stream_prepared_answer(prepared))

            mode_label = "reranking enabled" if prepared["reranking_used"] else "reranking disabled"
            st.caption(f"Mode: {mode_label}")

            with st.expander("Sources used"):
                if not prepared["retrieved"]:
                    st.write("No sources retrieved.")
                else:
                    for idx, source in enumerate(prepared["retrieved"], start=1):
                        score = source.get("rerank_score")
                        if score is None:
                            score = source.get("distance")
                        st.markdown(
                            f"**{idx}.** {source.get('source_file')} | strategy={source.get('chunking_strategy', 'fixed')} | chunk_id={source.get('chunk_id')} | score={score}"
                        )
                        st.caption(source.get("chunk_text", ""))
