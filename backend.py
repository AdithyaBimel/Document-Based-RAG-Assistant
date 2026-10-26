"""
backend.py — FastAPI backend
Exposes two endpoints:
  POST /upload  — ingest a PDF/TXT document
  POST /query   — ask a question about the uploaded document
  GET  /status  — check pipeline status

Run with: uvicorn backend:app --reload
"""

import os
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_pipeline import RAGPipeline

# ── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Document Assistant API",
    description="Upload a document, then ask questions about it.",
    version="1.0.0"
)

# Allow Streamlit frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global Pipeline Instance ──────────────────────────────────────────────────
pipeline = RAGPipeline()

UPLOAD_DIR = Path("./documents")
UPLOAD_DIR.mkdir(exist_ok=True)


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Initialize the RAG pipeline when the server starts."""
    print("[API] Starting up — loading embeddings and LLM...")
    pipeline.initialize()

    # Auto-load existing vector store if available
    if pipeline.load_existing_vectorstore():
        print("[API] Loaded existing vector store from disk.")
    else:
        print("[API] No existing vector store found. Please upload a document.")


# ── Request/Response Models ───────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 4


class QueryResponse(BaseModel):
    answer: str
    source_chunks: list[str]
    num_chunks_retrieved: int


class StatusResponse(BaseModel):
    ready: bool
    document: Optional[str]
    llm_mode: str


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "RAG Document Assistant API is running."}


@app.get("/status", response_model=StatusResponse)
def get_status():
    """Check if the pipeline is ready and which document is loaded."""
    return StatusResponse(
        ready=pipeline.is_ready,
        document=pipeline.document_name,
        llm_mode=os.getenv("LLM_MODE", "groq")
    )


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF or TXT file.
    The document is chunked, embedded, and stored in FAISS.
    """
    # Validate file type
    allowed_types = [".pdf", ".txt", ".md"]
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_types}"
        )

    # Save uploaded file
    save_path = UPLOAD_DIR / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Ingest into RAG pipeline
    try:
        result = pipeline.ingest_document(str(save_path))
        return {
            "message": "Document ingested successfully.",
            "filename": file.filename,
            "chunks_created": result["chunks"],
            "characters_processed": result["characters"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def query_document(request: QueryRequest):
    """
    Ask a question about the uploaded document.
    Returns the answer + source chunks used to generate it.
    """
    if not pipeline.is_ready:
        raise HTTPException(
            status_code=400,
            detail="No document loaded. Please upload a document first via POST /upload"
        )

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = pipeline.query(request.question)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return QueryResponse(
        answer=result["answer"],
        source_chunks=result["source_chunks"],
        num_chunks_retrieved=result["num_chunks_retrieved"]
    )
