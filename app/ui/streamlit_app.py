"""Streamlit UI for the SEC filing RAG project."""

from __future__ import annotations

import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.config import settings
from app.generation.rag_pipeline import answer_question
from app.ingestion import store
from app.ingestion.background_worker import run_ingestion_job
from app.ingestion.job_status import mark_stale_if_needed, read_status, write_status

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
TEXT_WARNING_LIMIT = 200_000
TEXT_HARD_LIMIT = 1_000_000
ingestion_status = mark_stale_if_needed()


def format_strategy_label(strategy: str) -> str:
    """Return a human-friendly label for a chunking strategy key."""
    return CHUNKING_STRATEGY_LABELS.get(strategy, strategy)


def format_document_label(document: dict) -> str:
    """Display a compact document label while retaining the full source value."""
    filename = document["source_file"]
    if len(filename) > 40:
        filename = f"{filename[:37]}..."
    return f"{filename} ({document['chunking_strategy']})"


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


def launch_ingestion_worker(
    filepath: str,
    filename: str,
    chunking_strategy: str,
    database_url: str,
) -> None:
    """Launch a standalone daemon process after the uploaded file is on disk."""
    worker = multiprocessing.Process(
        target=run_ingestion_job,
        args=(filepath, filename, chunking_strategy, database_url),
        daemon=True,
        name="ingestion-worker",
    )
    worker.start()


def render_ingestion_stage_ui(status: dict) -> None:
    """Render stage-based ingestion progress, using real percentage only for embeddings."""
    raw_stage = str(status.get("stage", "starting")).lower().strip()
    stage_key = raw_stage.replace("-", "_").replace(" ", "_")
    current = status.get("current", 0)
    total = status.get("total", 0)

    # Normalize possible worker stage names into a small, user-friendly pipeline.
    stage_aliases = {
        "starting": "starting",
        "start": "starting",
        "reading": "reading",
        "read": "reading",
        "extracting": "extracting",
        "extract_text": "extracting",
        "extracting_text": "extracting",
        "parsing": "extracting",
        "chunking": "chunking",
        "chunk": "chunking",
        "creating_chunks": "chunking",
        "embedding": "embedding",
        "embeddings": "embedding",
        "storing": "storing",
        "saving": "storing",
        "database": "storing",
        "complete": "complete",
    }
    current_stage = stage_aliases.get(stage_key, stage_key)

    stages = [
        ("starting", "Preparing document"),
        ("reading", "Reading document"),
        ("extracting", "Extracting text"),
        ("chunking", "Creating chunks"),
        ("embedding", "Generating embeddings"),
        ("storing", "Saving to vector database"),
    ]

    # If the worker reports an unknown stage, show it without breaking the UI.
    known_keys = {key for key, _ in stages}
    if current_stage not in known_keys and current_stage != "complete":
        stages.insert(-1, (current_stage, current_stage.replace("_", " ").title()))

    current_index = next(
        (index for index, (key, _) in enumerate(stages) if key == current_stage),
        0,
    )

    filename = status.get("filename", "document")
    stage_label = dict(stages).get(
        current_stage,
        current_stage.replace("_", " ").title(),
    )

    st.info(f"📄 Processing **{filename}**")

    # Show a compact pipeline so the user sees meaningful activity before
    # percentage-based embedding progress becomes available.
    pipeline_html = ["<div style='margin: 0.4rem 0 0.8rem 0;'>"]
    for index, (key, label) in enumerate(stages):
        if index < current_index:
            icon = "✓"
            weight = "normal"
        elif index == current_index:
            icon = "●"
            weight = "600"
        else:
            icon = "○"
            weight = "normal"

        pipeline_html.append(
            f"<div style='line-height: 1.8; font-weight: {weight};'>"
            f"<span style='display:inline-block; width:24px;'>{icon}</span>{label}"
            f"</div>"
        )
    pipeline_html.append("</div>")
    st.markdown("".join(pipeline_html), unsafe_allow_html=True)

    if current_stage == "embedding" and total > 0:
        percent = round(current / total * 100)
        st.progress(
            min(max(percent / 100, 0.0), 1.0),
            text=f"🧠 Generating embeddings: {current}/{total} chunks ({percent}%)",
        )
    elif current_stage not in {"complete", "failed"}:
        # Indeterminate animation avoids showing a misleading 0% progress bar.
        st.markdown(
            """
            <div style="
                width: 100%;
                height: 8px;
                border-radius: 999px;
                overflow: hidden;
                background: rgba(128,128,128,0.20);
                margin: 0.4rem 0 0.7rem 0;
            ">
                <div style="
                    width: 35%;
                    height: 100%;
                    border-radius: 999px;
                    background: rgba(128,128,128,0.65);
                    animation: ingestion-slide 1.4s ease-in-out infinite;
                "></div>
            </div>
            <style>
            @keyframes ingestion-slide {
                0%   { transform: translateX(-120%); }
                50%  { transform: translateX(180%); }
                100% { transform: translateX(300%); }
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    if current_stage == "embedding" and total > 0:
        st.caption(f"Current stage: {stage_label}")
    elif current_stage not in {"complete", "failed"}:
        st.caption(f"⏳ {stage_label}... Please wait.")


def render_ingestion_monitor() -> None:
    """Poll durable worker status so progress survives Streamlit reruns and refreshes."""

    @st.fragment(run_every="2s")
    def poll_main_status() -> None:
        status = mark_stale_if_needed()
        if status.get("in_progress"):
            render_ingestion_stage_ui(status)
        elif status.get("stage") == "failed":
            st.error(f"Ingestion failed: {status.get('error', 'Unknown error')}")

    @st.fragment(run_every="2s")
    def poll_sidebar_status() -> None:
        status = mark_stale_if_needed()
        if status.get("in_progress"):
            render_ingestion_stage_ui(status)
        elif status.get("stage") == "failed":
            st.error(f"Ingestion failed: {status.get('error', 'Unknown error')}")
        elif status.get("stage") == "complete" and not st.session_state.get("ingestion_success_pending"):
            st.session_state["ingestion_success_pending"] = {
                "filename": status.get("filename", "document"),
                "chunk_count": status.get("chunk_count", status.get("total", 0)),
            }
            st.rerun()

    poll_main_status()
    with st.sidebar:
        poll_sidebar_status()


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
st.session_state.setdefault("asking", False)

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
ingestion_in_progress = bool(ingestion_status.get("in_progress"))
if ingestion_in_progress:
    render_ingestion_monitor()
success_pending = st.session_state.pop("ingestion_success_pending", None)
if success_pending:
    st.success(
        f"{success_pending['filename']} successfully ingested "
        f"({success_pending['chunk_count']} chunks)."
    )

selected_source = None

with st.sidebar:
    if st.button("Disconnect", disabled=ingestion_in_progress):
        st.session_state["db_connected"] = False
        clear_connection_details()
        st.rerun()

    st.header("Ingest a document")
    uploaded_file = st.file_uploader(
        "Upload a PDF, HTML, or text filing",
        type=["pdf", "html", "htm", "txt"],
        accept_multiple_files=False,
        disabled=ingestion_in_progress,
    )
    if not ingestion_in_progress and uploaded_file is not None and uploaded_file.size > 5 * 1024 * 1024:
        st.warning("Large files may take several minutes to process on this hosted environment.")

    chunking_strategy = st.selectbox(
        "Chunking strategy",
        options=list(CHUNKING_STRATEGY_LABELS.keys()),
        format_func=lambda strategy: CHUNKING_STRATEGY_LABELS[strategy],
        index=0,
        disabled=ingestion_in_progress,
    )

    if st.button("Ingest", disabled=ingestion_in_progress):
        if uploaded_file is None:
            st.warning("Please upload a PDF, HTML, or TXT file before ingesting.")
        elif Path(uploaded_file.name).suffix.lower() not in {".pdf", ".html", ".htm", ".txt"}:
            st.error("Unsupported file type. Please upload a .pdf, .html, .htm, or .txt file.")
        else:
            destination = RAW_DIR / uploaded_file.name
            try:
                write_status(
                    in_progress=True,
                    filename=uploaded_file.name,
                    stage="starting",
                    current=0,
                    total=0,
                    started_at=time.time(),
                    error=None,
                )
                destination.write_bytes(uploaded_file.getvalue())
                database_url = store.engine.url.render_as_string(hide_password=False)
                launch_ingestion_worker(
                    str(destination),
                    uploaded_file.name,
                    chunking_strategy,
                    database_url,
                )
                st.success(f"Ingestion started for {uploaded_file.name}.")
                st.rerun()
            except Exception as exc:
                write_status(in_progress=False, stage="failed", error=str(exc))
                st.error(f"Ingestion failed: {exc}")

    st.header("Document filter")
    documents = get_available_documents()
    placeholder_label = "-- Select a document --"
    document_options = [{"source_file": "", "chunking_strategy": "", "label": placeholder_label}] + documents
    selected_doc = st.selectbox(
        "Choose a document",
        options=document_options,
        format_func=lambda option: placeholder_label if not option["source_file"] else format_document_label(option),
        index=0,
        disabled=ingestion_in_progress,
    )
    selected_source = selected_doc.get("source_file") if selected_doc and selected_doc.get("source_file") else None
    selected_chunking_strategy = (
        selected_doc.get("chunking_strategy") if selected_doc and selected_doc.get("chunking_strategy") else None
    )

    selected_document_key = (selected_source, selected_chunking_strategy)
    if st.session_state.get("selected_document_key") != selected_document_key:
        st.session_state["selected_document_key"] = selected_document_key
        st.session_state["use_reranking"] = True

    st.caption(f"Active Groq model: {settings.GROQ_MODEL}")

if selected_source is None:
    st.info("Select a document from the sidebar to start asking questions.")
else:
    st.header("Ask a question")
    use_reranking = st.toggle("Use reranking", key="use_reranking")
    st.caption("Reranking improves answer accuracy but increases response time. Turn off for faster replies.")
    question = st.text_input("Question", placeholder="Example: What are Apple's main risk factors?")
    if st.button("Ask", disabled=st.session_state["asking"]):
        if not question.strip():
            st.warning("Please enter a question.")
        elif not st.session_state["asking"]:
            st.session_state["asking"] = True
            st.session_state["ask_request"] = {
                "query": question,
                "source_file": selected_source,
                "chunking_strategy": selected_chunking_strategy,
                "use_reranking": use_reranking,
            }
            st.session_state.pop("ask_result", None)
            st.rerun()

    if st.session_state["asking"]:
        request = st.session_state["ask_request"]
        try:
            with st.spinner("Retrieving context and generating answer..."):
                st.session_state["ask_result"] = answer_question(
                    query=request["query"],
                    top_k=5,
                    source_file=request["source_file"],
                    chunking_strategy=request["chunking_strategy"],
                    use_reranking=request["use_reranking"],
                )
        except Exception as exc:
            st.session_state["ask_result"] = {"error": str(exc)}
        finally:
            st.session_state["asking"] = False
        st.rerun()

    result = st.session_state.get("ask_result")
    if result:
        if result.get("error"):
            st.error(f"Question failed: {result['error']}")
        else:
            st.markdown("### Answer")
            st.markdown(result["answer"])

            mode_label = "reranking enabled" if result.get("reranking_used") else "reranking disabled"
            st.caption(f"Mode: {mode_label}")

            with st.expander("Sources used"):
                if not result.get("sources"):
                    st.write("No sources retrieved.")
                else:
                    for idx, source in enumerate(result["sources"], start=1):
                        score = source.get("rerank_score")
                        if score is None:
                            score = source.get("distance")
                        st.markdown(
                            f"**{idx}.** {source.get('source_file')} | chunk_id={source.get('chunk_id')} | score={score}"
                        )
                        st.caption(source.get("chunk_text", ""))