"""
rag_pipeline.py — Core RAG logic
Handles: document loading, chunking, embedding, vector store, and querying
Supports: Groq API, OpenAI API, or local Mistral 7B (RTX 4050)
"""

import os
import torch
import pdfplumber
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Dict, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
LLM_MODE         = os.getenv("LLM_MODE", "groq")
GROQ_API_KEY     = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
LOCAL_MODEL      = os.getenv("LOCAL_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "./vectorstore/faiss_index")


# ── Document Loading ──────────────────────────────────────────────────────────
def load_pdf(file_path: str) -> str:
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def load_document(file_path: str) -> str:
    path = Path(file_path)
    if path.suffix.lower() == ".pdf":
        return load_pdf(file_path)
    elif path.suffix.lower() in [".txt", ".md"]:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


# ── Chunking ──────────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)


# ── Embeddings ────────────────────────────────────────────────────────────────
def get_embeddings():
    print(f"[RAG] Loading embedding model: {EMBEDDING_MODEL} (CPU)")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


# ── Vector Store ──────────────────────────────────────────────────────────────
def build_vectorstore(chunks: List[str], embeddings) -> FAISS:
    print(f"[RAG] Building vector store with {len(chunks)} chunks...")
    return FAISS.from_texts(chunks, embeddings)


def save_vectorstore(vectorstore: FAISS, path: str = VECTORSTORE_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(path)
    print(f"[RAG] Vector store saved to {path}")


def load_vectorstore(embeddings, path: str = VECTORSTORE_PATH) -> FAISS:
    print(f"[RAG] Loading vector store from {path}")
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


def vectorstore_exists(path: str = VECTORSTORE_PATH) -> bool:
    return Path(path + ".faiss").exists()


# ── LLM Setup ─────────────────────────────────────────────────────────────────
def get_llm():
    if LLM_MODE == "groq":
        return _get_groq_llm()
    elif LLM_MODE == "openai":
        return _get_openai_llm()
    elif LLM_MODE == "local":
        return _get_local_llm()
    else:
        raise ValueError(f"Unknown LLM_MODE: {LLM_MODE}")


def _get_groq_llm():
    try:
        from langchain_groq import ChatGroq
        print("[RAG] Using Groq API (Llama 3 8B)")
        return ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama3-8b-8192",
            temperature=0.1,
            max_tokens=1024
        )
    except ImportError:
        raise ImportError("Run: pip install langchain-groq")


def _get_openai_llm():
    try:
        from langchain_openai import ChatOpenAI
        print("[RAG] Using OpenAI API")
        return ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model_name="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=1024
        )
    except ImportError:
        raise ImportError("Run: pip install langchain-openai")


def _get_local_llm():
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        BitsAndBytesConfig, pipeline
    )
    from langchain_community.llms import HuggingFacePipeline

    print(f"[RAG] Loading local model: {LOCAL_MODEL}")
    print("[RAG] First run downloads ~4GB. Subsequent runs are instant.")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL,
        quantization_config=bnb_config,
        device_map="cuda",
        trust_remote_code=True
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    print(f"[RAG] VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return HuggingFacePipeline(pipeline=pipe)


# ── RAG Query ─────────────────────────────────────────────────────────────────
def build_prompt(question: str, context_chunks: List[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    return f"""You are a helpful document assistant. Answer the question based ONLY on the provided context.
If the answer is not in the context, say "I couldn't find that in the document."
Be concise and indicate which part of the document supports your answer.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


def query_rag(question: str, vectorstore: FAISS, llm, top_k: int = 4) -> Dict[str, Any]:
    docs = vectorstore.similarity_search(question, k=top_k)
    context_chunks = [doc.page_content for doc in docs]
    prompt = build_prompt(question, context_chunks)

    if hasattr(llm, "invoke"):
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)
    else:
        answer = llm(prompt)

    if LLM_MODE == "local" and "ANSWER:" in answer:
        answer = answer.split("ANSWER:")[-1].strip()

    return {
        "answer": answer,
        "source_chunks": context_chunks,
        "num_chunks_retrieved": len(context_chunks)
    }


# ── Pipeline Class ────────────────────────────────────────────────────────────
class RAGPipeline:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.is_ready = False
        self.document_name = None

    def initialize(self):
        print("[RAG] Initializing pipeline...")
        self.embeddings = get_embeddings()
        self.llm = get_llm()
        print("[RAG] Pipeline ready.")

    def ingest_document(self, file_path: str) -> Dict[str, Any]:
        print(f"[RAG] Ingesting: {file_path}")
        text = load_document(file_path)
        if not text.strip():
            raise ValueError("Document appears to be empty or unreadable.")
        chunks = chunk_text(text)
        print(f"[RAG] {len(chunks)} chunks created")
        self.vectorstore = build_vectorstore(chunks, self.embeddings)
        save_vectorstore(self.vectorstore)
        self.document_name = Path(file_path).name
        self.is_ready = True
        return {
            "status": "success",
            "document": self.document_name,
            "chunks": len(chunks),
            "characters": len(text)
        }

    def query(self, question: str) -> Dict[str, Any]:
        if not self.is_ready:
            return {"error": "No document loaded. Please upload a document first."}
        return query_rag(question, self.vectorstore, self.llm)

    def load_existing_vectorstore(self):
        if vectorstore_exists():
            self.vectorstore = load_vectorstore(self.embeddings)
            self.is_ready = True
            return True
        return False
