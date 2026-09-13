"use client";

import { ChangeEvent, DragEvent, FormEvent, useEffect, useRef, useState } from "react";
import {
  AlertCircle,
  Check,
  FileText,
  Loader2,
  RotateCcw,
  Sparkles,
  Upload,
  UploadCloud,
  ArrowUp,
} from "lucide-react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
const ACCEPTED_EXTENSIONS = [".pdf", ".html", ".htm", ".txt"];
// The backend's /api/ingest call is synchronous (it blocks until the whole
// pipeline finishes) and there's no status endpoint, so we can't get real
// stage/percentage data. This just advances a cosmetic checklist on a timer
// while we wait for the one synchronous response, purely for perceived progress.
const SIMULATED_STAGE_MS = 1400;

// ---- Chunking strategies (mirrors the Streamlit CHUNKING_STRATEGY_LABELS) ----
const CHUNKING_STRATEGIES: { value: string; label: string }[] = [
  { value: "fixed", label: "Fixed-size" },
  { value: "sentence_aware", label: "Sentence-aware" },
  { value: "paragraph_based", label: "Paragraph-based" },
  { value: "recursive", label: "Recursive (paragraph → sentence → fixed)" },
];

function strategyLabel(value: string): string {
  return CHUNKING_STRATEGIES.find((s) => s.value === value)?.label || value;
}

// ---- Ingestion pipeline stages (cosmetic only — see SIMULATED_STAGE_MS note above) ----
const STAGE_PIPELINE: { label: string }[] = [
  { label: "Preparing document" },
  { label: "Reading document" },
  { label: "Extracting text" },
  { label: "Creating chunks" },
  { label: "Generating embeddings" },
  { label: "Saving to vector database" },
];

type Source = Record<string, unknown>;
type Message = {
  id: number;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  loading?: boolean;
};

// The overall screen the app is showing. "choosing" is used both for the very
// first upload and for replacing an existing document — the screen only ever
// shows a file picker + chunking-strategy select, per the requested flow.
type Stage = "checking" | "choosing" | "confirm_replace" | "processing" | "chat";

function errorText(value: unknown, fallback: string) {
  if (typeof value === "string") return value;
  if (value && typeof value === "object" && "detail" in value) return String(value.detail);
  return fallback;
}

/** Timeline-style progress: a real sequence, so a connected vertical marker
 * treatment earns its place here (unlike a generic numbered list). */
function IngestionProgress({ filename, stageIndex }: { filename: string; stageIndex: number }) {
  const clampedIndex = Math.min(stageIndex, STAGE_PIPELINE.length - 1);

  return (
    <div className="w-full max-w-md rounded-2xl border border-[#E5E1D6] bg-white p-7 shadow-[0_1px_2px_rgba(20,23,31,0.04),0_8px_24px_-8px_rgba(20,23,31,0.10)]">
      <div className="mb-6 flex items-center gap-3">
        <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-[#0E7C66]/10 text-[#0E7C66]">
          <FileText size={17} />
        </span>
        <div className="min-w-0">
          <p className="text-xs uppercase tracking-wide text-[#6B7280]">Processing</p>
          <p className="truncate font-mono text-sm text-[#14171F]">{filename}</p>
        </div>
      </div>

      <ol className="relative ml-[15px] space-y-5 border-l border-[#E5E1D6] pl-6">
        {STAGE_PIPELINE.map((stage, index) => {
          const done = index < clampedIndex;
          const active = index === clampedIndex;
          return (
            <li key={stage.label} className="relative">
              <span
                className={`absolute -left-[31px] flex h-6 w-6 items-center justify-center rounded-full border text-[11px] transition-colors ${
                  done
                    ? "border-[#0E7C66] bg-[#0E7C66] text-white"
                    : active
                    ? "border-[#0E7C66] bg-white text-[#0E7C66]"
                    : "border-[#E5E1D6] bg-white text-[#C8C4B8]"
                }`}
              >
                {done ? <Check size={13} /> : active ? <Loader2 size={13} className="animate-spin" /> : index + 1}
              </span>
              <p className={`text-sm ${active ? "font-medium text-[#14171F]" : done ? "text-[#6B7280]" : "text-[#B7B3A6]"}`}>
                {stage.label}
              </p>
            </li>
          );
        })}
      </ol>
    </div>
  );
}

/** Advances a fake stage index on a timer while `active` is true, stopping just
 * short of the final stage until the real request resolves. */
function useSimulatedStages(active: boolean): number {
  const [stageIndex, setStageIndex] = useState(0);

  useEffect(() => {
    if (!active) {
      setStageIndex(0);
      return;
    }
    const maxIndex = STAGE_PIPELINE.length - 1;
    const timer = setInterval(() => {
      setStageIndex((current) => (current < maxIndex ? current + 1 : current));
    }, SIMULATED_STAGE_MS);
    return () => clearInterval(timer);
  }, [active]);

  return stageIndex;
}

/** Full-screen "choose a file + chunking strategy" screen. Used both for the very
 * first upload and for replacing the current document. */
function ChoosingScreen({
  onConfirm,
  onCancel,
  error,
}: {
  onConfirm: (file: File, chunkingStrategy: string) => void;
  onCancel?: () => void;
  error?: string;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [chunkingStrategy, setChunkingStrategy] = useState(CHUNKING_STRATEGIES[0].value);
  const [localError, setLocalError] = useState("");

  function acceptFile(candidate?: File) {
    if (!candidate) return;
    const extension = `.${candidate.name.split(".").pop()?.toLowerCase()}`;
    if (!ACCEPTED_EXTENSIONS.includes(extension)) {
      setLocalError("Unsupported file type. Please choose a PDF, HTML, or TXT file.");
      return;
    }
    setLocalError("");
    setFile(candidate);
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    acceptFile(event.dataTransfer.files[0]);
  }

  return (
    <main className="flex min-h-screen items-center justify-center bg-[#F7F5F0] px-6 text-[#14171F]">
      <section className="w-full max-w-md">
        <div className="rounded-2xl border border-[#E5E1D6] bg-white p-8 shadow-[0_1px_2px_rgba(20,23,31,0.04),0_8px_24px_-8px_rgba(20,23,31,0.10)]">
          <span className="mb-4 flex h-10 w-10 items-center justify-center rounded-full bg-[#14171F] text-white">
            <Sparkles size={17} />
          </span>
          <h1 className="font-serif text-2xl tracking-tight">Choose a document</h1>
          <p className="mt-1.5 text-sm leading-6 text-[#6B7280]">Pick a file and a chunking strategy, then confirm to start processing.</p>

          <div
            onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            className={`mt-6 rounded-xl border-2 border-dashed px-4 py-8 text-center transition-colors ${
              dragging ? "border-[#0E7C66] bg-[#0E7C66]/5" : "border-[#E5E1D6]"
            }`}
          >
            <button
              type="button"
              onClick={() => inputRef.current?.click()}
              className="inline-flex items-center gap-2 rounded-lg bg-[#14171F] px-4 py-2.5 text-sm font-medium text-white transition-colors hover:bg-[#272B36]"
            >
              <Upload size={15} />
              {file ? "Change file" : "Choose file"}
            </button>
            <input
              ref={inputRef}
              type="file"
              accept={ACCEPTED_EXTENSIONS.join(",")}
              className="hidden"
              onChange={(event: ChangeEvent<HTMLInputElement>) => acceptFile(event.target.files?.[0])}
            />
            <p className="mt-3 truncate font-mono text-sm text-[#14171F]">{file ? file.name : "or drag a file here"}</p>
            <p className="mt-1 text-xs text-[#9C9887]">PDF · HTML · HTM · TXT</p>
          </div>

          <label className="mt-6 block text-left">
            <span className="mb-1.5 block text-xs font-medium text-[#6B7280]">Chunking strategy</span>
            <div className="relative">
              <select
                value={chunkingStrategy}
                onChange={(event) => setChunkingStrategy(event.target.value)}
                className="w-full appearance-none rounded-lg border border-[#E5E1D6] bg-white px-3 py-2.5 pr-9 text-sm text-[#14171F] outline-none transition-colors focus:border-[#0E7C66]"
              >
                {CHUNKING_STRATEGIES.map((strategy) => (
                  <option key={strategy.value} value={strategy.value}>{strategy.label}</option>
                ))}
              </select>
              <svg viewBox="0 0 20 20" fill="none" className="pointer-events-none absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 text-[#9C9887]">
                <path d="M5 7.5L10 12.5L15 7.5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </div>
          </label>

          {(localError || error) && (
            <p role="alert" className="mt-4 flex items-start gap-2 rounded-lg bg-red-50 px-3.5 py-2.5 text-sm text-red-700">
              <AlertCircle size={16} className="mt-0.5 shrink-0" />
              {localError || error}
            </p>
          )}

          <div className="mt-7 flex justify-end gap-2">
            {onCancel && <button type="button" onClick={onCancel} className="rounded-lg px-4 py-2 text-sm text-[#6B7280] transition-colors hover:bg-[#F7F5F0]">Cancel</button>}
            <button
              type="button"
              disabled={!file}
              onClick={() => file && onConfirm(file, chunkingStrategy)}
              className="rounded-lg bg-[#0E7C66] px-5 py-2.5 text-sm font-medium text-white transition-colors hover:bg-[#0B6553] disabled:cursor-not-allowed disabled:bg-[#E5E1D6] disabled:text-[#9C9887]"
            >
              OK
            </button>
          </div>
        </div>
      </section>
    </main>
  );
}

/** Full-screen "are you sure" step shown before replacing an existing document. */
function ConfirmReplaceScreen({ documentName, onConfirm, onCancel }: { documentName: string; onConfirm: () => void; onCancel: () => void }) {
  return (
    <main className="flex min-h-screen items-center justify-center bg-[#F7F5F0] px-6 text-[#14171F]">
      <section className="w-full max-w-sm rounded-2xl border border-[#E5E1D6] bg-white p-8 text-center shadow-[0_1px_2px_rgba(20,23,31,0.04),0_8px_24px_-8px_rgba(20,23,31,0.10)]">
        <span className="mx-auto mb-4 flex h-10 w-10 items-center justify-center rounded-full bg-amber-50 text-amber-600">
          <RotateCcw size={17} />
        </span>
        <h1 className="font-serif text-xl tracking-tight">Replace the current document?</h1>
        <p className="mt-2 text-sm leading-6 text-[#6B7280]">
          You're currently working with <span className="font-mono text-[#14171F]">{documentName}</span>. Uploading a new document will start a fresh conversation.
        </p>
        <div className="mt-6 flex justify-center gap-2">
          <button type="button" onClick={onCancel} className="rounded-lg px-4 py-2 text-sm text-[#6B7280] transition-colors hover:bg-[#F7F5F0]">Cancel</button>
          <button type="button" onClick={onConfirm} className="rounded-lg bg-[#14171F] px-5 py-2.5 text-sm font-medium text-white transition-colors hover:bg-[#272B36]">OK, continue</button>
        </div>
      </section>
    </main>
  );
}

/** Full-screen processing view shown while an upload is being ingested. */
function ProcessingScreen({ filename, stageIndex }: { filename: string; stageIndex: number }) {
  return (
    <main className="flex min-h-screen items-center justify-center bg-[#F7F5F0] px-6">
      <IngestionProgress filename={filename} stageIndex={stageIndex} />
    </main>
  );
}

function ChatMessage({ message }: { message: Message }) {
  const isUser = message.role === "user";
  return (
    <div className={`flex items-start gap-2.5 ${isUser ? "flex-row-reverse" : ""}`}>
      <span
        className={`mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-[11px] font-medium ${
          isUser ? "bg-[#14171F] text-white" : "bg-[#0E7C66]/10 text-[#0E7C66]"
        }`}
      >
        {isUser ? "You" : <Sparkles size={13} />}
      </span>
      <div className={`max-w-[min(680px,84%)] rounded-2xl px-4 py-3 text-sm leading-6 ${
        isUser
          ? "rounded-tr-sm bg-[#14171F] text-white"
          : "rounded-tl-sm border border-[#E5E1D6] border-l-2 border-l-[#0E7C66] bg-white text-[#2A2E38] shadow-[0_1px_2px_rgba(20,23,31,0.04)]"
      }`}>
        {message.loading ? (
          <span className="inline-flex items-center gap-1.5 text-[#9C9887]">
            <Loader2 size={14} className="animate-spin" /> Thinking…
          </span>
        ) : (
          <p className="whitespace-pre-wrap">{message.content}</p>
        )}
        {!isUser && message.sources && message.sources.length > 0 && (
          <details className="mt-3 border-t border-[#E5E1D6] pt-2 text-xs text-[#6B7280]">
            <summary className="cursor-pointer select-none font-medium text-[#2A2E38]">Sources ({message.sources.length})</summary>
            <div className="mt-2 space-y-2">
              {message.sources.map((source, index) => (
                <div key={index} className="rounded-lg bg-[#F7F5F0] p-2 font-mono text-[11px]">
                  {Object.entries(source).map(([key, value]) => (
                    <div key={key}><span className="font-medium">{key.replaceAll("_", " ")}: </span>{typeof value === "object" ? JSON.stringify(value) : String(value)}</div>
                  ))}
                </div>
              ))}
            </div>
          </details>
        )}
      </div>
    </div>
  );
}

export default function Page() {
  const [stage, setStage] = useState<Stage>("checking");
  const [documentName, setDocumentName] = useState<string | null>(null);
  const [currentStrategy, setCurrentStrategy] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [reranking, setReranking] = useState(true);
  const [querying, setQuerying] = useState(false);
  const [error, setError] = useState("");

  const [uploadingName, setUploadingName] = useState("document");
  const messageId = useRef(0);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const stageIndex = useSimulatedStages(stage === "processing");

  useEffect(() => {
    fetch(`${API_URL}/api/documents`).then(async (response) => {
      if (!response.ok) throw new Error("Could not check stored documents.");
      const documents: string[] = await response.json();
      if (documents.length) {
        setDocumentName(documents[0]);
        setStage("chat");
      } else {
        setStage("choosing");
      }
    }).catch((caught) => {
      setError(caught instanceof Error ? caught.message : "Could not connect to the backend.");
      setStage("choosing");
    });
  }, []);

  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  async function startIngestion(file: File, chunkingStrategy: string) {
    setError("");
    setUploadingName(file.name);
    setStage("processing");
    const body = new FormData();
    body.append("file", file);
    body.append("chunking_strategy", chunkingStrategy);
    try {
      const response = await fetch(`${API_URL}/api/ingest`, { method: "POST", body });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(errorText(data, "Upload failed. Please try again."));
      setDocumentName(data.filename || file.name);
      setCurrentStrategy(chunkingStrategy);
      setMessages([]);
      setStage("chat");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Upload failed. Please try again.");
      setStage("choosing");
    }
  }

  async function sendQuestion(event?: FormEvent) {
    event?.preventDefault(); const text = question.trim(); if (!text || querying) return;
    const userId = ++messageId.current; const assistantId = ++messageId.current;
    setQuestion(""); setError(""); setQuerying(true);
    setMessages((current) => [...current, { id: userId, role: "user", content: text }, { id: assistantId, role: "assistant", content: "", loading: true }]);
    try {
      const response = await fetch(`${API_URL}/api/query`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: text, top_k: 5, use_reranking: reranking }) });
      const data = await response.json().catch(() => ({})); if (!response.ok) throw new Error(errorText(data, "Question failed. Please try again."));
      setMessages((current) => current.map((message) => message.id === assistantId ? { ...message, content: data.answer || "No answer returned.", sources: Array.isArray(data.sources) ? data.sources : [], loading: false } : message));
    } catch (caught) { setMessages((current) => current.filter((message) => message.id !== assistantId)); setError(caught instanceof Error ? caught.message : "Question failed. Please try again."); } finally { setQuerying(false); }
  }

  if (stage === "checking") {
    return (
      <main className="flex min-h-screen items-center justify-center bg-[#F7F5F0] text-sm text-[#6B7280]">
        <span className="inline-flex items-center gap-2">
          <Loader2 size={15} className="animate-spin" /> Loading your workspace…
        </span>
      </main>
    );
  }

  if (stage === "choosing") {
    return (
      <ChoosingScreen
        onConfirm={(file, chunkingStrategy) => void startIngestion(file, chunkingStrategy)}
        onCancel={documentName ? () => setStage("chat") : undefined}
        error={error}
      />
    );
  }

  if (stage === "confirm_replace") {
    return (
      <ConfirmReplaceScreen
        documentName={documentName || "document"}
        onConfirm={() => setStage("choosing")}
        onCancel={() => setStage("chat")}
      />
    );
  }

  if (stage === "processing") {
    return <ProcessingScreen filename={uploadingName} stageIndex={stageIndex} />;
  }

  return (
    <main className="flex min-h-screen flex-col bg-[#F7F5F0] text-[#14171F]">
      <header className="border-b border-[#E5E1D6] bg-white/90 backdrop-blur">
        <div className="mx-auto flex max-w-5xl items-center justify-between gap-4 px-5 py-4">
          <div className="flex min-w-0 items-center gap-3">
            <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-[#0E7C66]/10 text-[#0E7C66]">
              <FileText size={16} />
            </span>
            <div className="min-w-0">
              <p className="truncate font-mono text-sm text-[#14171F]">{documentName}</p>
              {currentStrategy && (
                <span className="mt-0.5 inline-block rounded-full bg-[#F7F5F0] px-2 py-0.5 font-mono text-[11px] text-[#6B7280]">
                  {strategyLabel(currentStrategy)}
                </span>
              )}
            </div>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            <button type="button" onClick={() => setStage("confirm_replace")} className="inline-flex items-center gap-1.5 rounded-lg px-3 py-2 text-sm text-[#6B7280] transition-colors hover:bg-[#F7F5F0]">
              <UploadCloud size={15} /> Upload new
            </button>
            <button type="button" onClick={() => setMessages([])} className="rounded-lg border border-[#E5E1D6] px-3 py-2 text-sm font-medium text-[#2A2E38] transition-colors hover:bg-[#F7F5F0]">New chat</button>
          </div>
        </div>
      </header>

      <section className="mx-auto flex w-full max-w-5xl flex-1 flex-col px-5">
        <div className="flex-1 space-y-5 overflow-y-auto py-8">
          {messages.length === 0 && (
            <div className="mx-auto mt-16 max-w-md text-center">
              <span className="mx-auto mb-4 flex h-11 w-11 items-center justify-center rounded-full bg-[#14171F] text-white">
                <Sparkles size={18} />
              </span>
              <h1 className="font-serif text-xl tracking-tight">What would you like to know?</h1>
              <p className="mt-2 text-sm leading-6 text-[#6B7280]">Ask a question about <span className="font-mono text-[#2A2E38]">{documentName}</span> and I'll find the relevant passages.</p>
            </div>
          )}
          {messages.map((message) => <ChatMessage key={message.id} message={message} />)}
          <div ref={messagesEndRef} />
        </div>

        {error && (
          <p role="alert" className="mb-3 flex items-start gap-2 rounded-lg bg-red-50 px-3.5 py-2.5 text-sm text-red-700">
            <AlertCircle size={16} className="mt-0.5 shrink-0" /> {error}
          </p>
        )}

        <form onSubmit={sendQuestion} className="sticky bottom-0 -mx-5 border-t border-[#E5E1D6] bg-[#F7F5F0] px-5 py-4">
          <div className="mb-3 flex items-center justify-end gap-2 text-xs text-[#6B7280]">
            <label className="flex cursor-pointer items-center gap-2">
              <span>Reranking</span>
              <span className="relative inline-flex h-5 w-9 items-center">
                <input
                  type="checkbox"
                  checked={reranking}
                  onChange={(event) => setReranking(event.target.checked)}
                  className="peer sr-only"
                />
                <span className="absolute inset-0 rounded-full bg-[#E5E1D6] transition-colors peer-checked:bg-[#0E7C66]" />
                <span className="absolute left-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform peer-checked:translate-x-4" />
              </span>
            </label>
          </div>
          <div className="flex items-center gap-2 rounded-full border border-[#E5E1D6] bg-white p-1.5 pl-4 shadow-[0_1px_2px_rgba(20,23,31,0.04)]">
            <input
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Ask anything about your document…"
              className="min-w-0 flex-1 bg-transparent text-sm outline-none placeholder:text-[#9C9887]"
              disabled={querying}
            />
            <button
              type="submit"
              disabled={querying || !question.trim()}
              className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-[#14171F] text-white transition-colors hover:bg-[#272B36] disabled:cursor-not-allowed disabled:bg-[#E5E1D6] disabled:text-[#9C9887]"
              aria-label="Send"
            >
              <ArrowUp size={16} />
            </button>
          </div>
        </form>
      </section>
    </main>
  );
}