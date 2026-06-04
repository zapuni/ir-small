"""
Student Server — FastAPI application for the Offline RAG Competition.

Exposes the two required endpoints:
    POST /upload   -> chunk + embed + index a document (<=120s budget)
    POST /ask      -> retrieve context + ask LLM -> single-letter answer (<=60s)

Plus helpers:
    GET  /health   -> readiness/status
    GET  /         -> basic info

Run (from the src/ folder):
    python app.py
    # or
    uvicorn app:app --host 0.0.0.0 --port 5000
"""
from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

import config
from embedder import warm_up
from llm import fallback_answer, get_client, trim_contexts
from retriever import INDEX
from textutils import parse_mc_question


# --------------------------------------------------------------------------- #
# Schemas                                                                      #
# --------------------------------------------------------------------------- #
class UploadRequest(BaseModel):
    doc_id: Optional[str] = "none"
    text: str = ""


class UploadResponse(BaseModel):
    status: str
    doc_id: str
    chunks: int


class AskRequest(BaseModel):
    question: str = Field(default="")


class AskResponse(BaseModel):
    answer: str
    sources: List[str] = []


# --------------------------------------------------------------------------- #
# Lifespan: warm up the model before serving traffic                          #
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[app] Config:\n  " + config.describe().replace("\n", "\n  "))
    print("[app] Warming up embedding model...")
    t0 = time.time()
    warm_up()
    try:
        get_client()  # initialise LLM client early (cheap)
    except Exception as exc:
        print(f"[app] LLM client init warning: {exc!r}")
    print(f"[app] Warm-up done in {time.time() - t0:.1f}s. Ready to serve.")
    yield
    print("[app] Shutting down.")


app = FastAPI(title="Offline RAG Student Server", version="1.0", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# Endpoints                                                                    #
# --------------------------------------------------------------------------- #
@app.get("/")
def root():
    return {
        "service": "Offline RAG Student Server",
        "student_id": config.STUDENT_ID,
        "embed_model": config.EMBED_MODEL_NAME,
        "indexed_chunks": len(INDEX.chunks),
        "ready": INDEX.ready,
    }


@app.get("/health")
def health():
    return {"status": "ok", "indexed": INDEX.ready, "chunks": len(INDEX.chunks)}


@app.post("/upload", response_model=UploadResponse)
def upload(req: UploadRequest):
    """Receive a document, chunk + embed + index it."""
    t0 = time.time()
    doc_id = req.doc_id or "none"
    n_chunks = INDEX.build(req.text or "")
    print(
        f"[upload] doc_id={doc_id} chunks={n_chunks} "
        f"chars={len(req.text or '')} took={time.time() - t0:.2f}s"
    )
    return UploadResponse(
        status="success" if n_chunks > 0 else "empty",
        doc_id=doc_id if doc_id != "none" else "uploaded_doc",
        chunks=n_chunks,
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Answer a multiple-choice question using RAG; always return one letter."""
    t0 = time.monotonic()
    deadline = t0 + config.ASK_DEADLINE
    mcq = parse_mc_question(req.question or "")

    # Build retrieval queries: the stem + each option (option-aware retrieval).
    queries: List[str] = [mcq.stem or mcq.raw]
    if mcq.has_options:
        for letter in mcq.option_letters():
            queries.append(f"{mcq.stem} {mcq.options[letter]}")

    results = INDEX.search(queries, top_k=config.TOP_K_CONTEXT)
    contexts = trim_contexts([r.chunk for r in results])

    # 1) Try the LLM, but never exceed the wall-clock deadline. Run it in a
    #    worker thread so a hung socket can't block past the budget.
    letter: str | None = None
    try:
        result_box: dict = {}

        def _call():
            try:
                result_box["letter"] = get_client().answer(mcq, contexts, deadline=deadline)
            except Exception as exc:  # pragma: no cover
                result_box["error"] = exc

        worker = threading.Thread(target=_call, daemon=True)
        worker.start()
        worker.join(timeout=max(0.1, deadline - time.monotonic()))
        if worker.is_alive():
            print("[ask] LLM exceeded deadline -> falling back")
        else:
            letter = result_box.get("letter")
    except Exception as exc:
        print(f"[ask] LLM error: {exc!r}")

    # 2) Fallback so we never return an invalid/empty answer.
    if letter not in ("A", "B", "C", "D"):
        letter = fallback_answer(mcq, contexts)
        print(f"[ask] used fallback -> {letter}")

    print(
        f"[ask] answer={letter} ctx={len(contexts)} "
        f"opts={mcq.option_letters()} took={time.monotonic() - t0:.2f}s"
    )
    return AskResponse(answer=letter, sources=contexts)


# --------------------------------------------------------------------------- #
# Entrypoint                                                                   #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=config.HOST,
        port=config.PORT,
        workers=1,        # single worker -> model loaded once, shared index
        log_level="info",
    )
