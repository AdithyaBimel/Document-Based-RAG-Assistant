"""
app.py — Streamlit frontend
A clean chat interface for the RAG Document Assistant.
Run with: streamlit run app.py
"""

import streamlit as st
import requests
import json
from pathlib import Path

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

BACKEND_URL = "http://localhost:8000"

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark professional theme */
    .stApp { background-color: #0f1117; }

    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #8b8fa8;
        margin-bottom: 2rem;
    }

    /* Source chunk cards */
    .source-card {
        background: #1e2130;
        border-left: 3px solid #4f8ef7;
        border-radius: 4px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
        color: #b0b4c8;
        font-family: 'Courier New', monospace;
    }

    /* Status badge */
    .status-ready {
        background: #1a3a2a;
        color: #4ade80;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-not-ready {
        background: #3a1a1a;
        color: #f87171;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* Chat messages */
    .chat-user {
        background: #1e2130;
        border-radius: 12px 12px 4px 12px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        color: #e2e4f0;
    }
    .chat-assistant {
        background: #162236;
        border-left: 3px solid #4f8ef7;
        border-radius: 4px 12px 12px 12px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        color: #e2e4f0;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ──────────────────────────────────────────────────────────
def check_backend_status():
    try:
        resp = requests.get(f"{BACKEND_URL}/status", timeout=3)
        return resp.json()
    except Exception:
        return None


def upload_document(file):
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        resp = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=60)
        return resp.json(), resp.status_code
    except Exception as e:
        return {"detail": str(e)}, 500


def query_document(question: str):
    try:
        resp = requests.post(
            f"{BACKEND_URL}/query",
            json={"question": question, "top_k": 4},
            timeout=120
        )
        return resp.json(), resp.status_code
    except Exception as e:
        return {"detail": str(e)}, 500


# ── Session State ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_loaded" not in st.session_state:
    st.session_state.doc_loaded = False
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 RAG Assistant")
    st.markdown("---")

    # Backend status check
    status = check_backend_status()
    if status is None:
        st.error("❌ Backend offline\nRun: `uvicorn backend:app --reload`")
    else:
        if status.get("ready"):
            st.markdown(
                f'<span class="status-ready">● Document loaded</span>',
                unsafe_allow_html=True
            )
            st.caption(f"📁 {status.get('document', 'Unknown')}")
            st.caption(f"🤖 Mode: `{status.get('llm_mode', '?')}`")
            st.session_state.doc_loaded = True
            st.session_state.doc_name = status.get("document")
        else:
            st.markdown(
                '<span class="status-not-ready">● No document loaded</span>',
                unsafe_allow_html=True
            )
            st.caption(f"🤖 Mode: `{status.get('llm_mode', '?')}`")

    st.markdown("---")
    st.markdown("### Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF or TXT file",
        type=["pdf", "txt", "md"],
        help="The document will be chunked and embedded into the vector store."
    )

    if uploaded_file and st.button("📥 Ingest Document", use_container_width=True):
        with st.spinner(f"Processing {uploaded_file.name}..."):
            result, code = upload_document(uploaded_file)
            if code == 200:
                st.success(f"✅ Ingested successfully!\n\n"
                           f"**Chunks:** {result.get('chunks_created')}\n\n"
                           f"**Characters:** {result.get('characters_processed'):,}")
                st.session_state.doc_loaded = True
                st.session_state.doc_name = uploaded_file.name
                st.session_state.messages = []  # reset chat on new doc
                st.rerun()
            else:
                st.error(f"❌ {result.get('detail', 'Upload failed')}")

    st.markdown("---")

    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("""
    **How to use:**
    1. Start the backend: `uvicorn backend:app --reload`
    2. Upload a PDF/TXT document
    3. Ask questions in the chat

    **Stack:**
    - 🔗 LangChain + FAISS
    - 🤗 sentence-transformers
    - ⚡ FastAPI backend
    - 🖥️ Streamlit frontend
    """)


# ── Main Chat Area ────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">📄 RAG Document Assistant</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="sub-header">Ask questions about your document — answers grounded in source text</div>',
    unsafe_allow_html=True
)

# Show document context if loaded
if st.session_state.doc_loaded and st.session_state.doc_name:
    st.info(f"💬 Chatting about: **{st.session_state.doc_name}**")

# Chat history
chat_container = st.container()
with chat_container:
    if not st.session_state.messages:
        if st.session_state.doc_loaded:
            st.markdown("""
            <div style="text-align:center; color:#8b8fa8; padding: 3rem 0;">
                <div style="font-size:2rem">💬</div>
                <div>Document loaded! Ask your first question below.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center; color:#8b8fa8; padding: 3rem 0;">
                <div style="font-size:2rem">📂</div>
                <div>Upload a document in the sidebar to get started.</div>
            </div>
            """, unsafe_allow_html=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

            # Show source chunks for assistant messages
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander(f"📎 Source chunks ({len(msg['sources'])} retrieved)"):
                    for i, chunk in enumerate(msg["sources"], 1):
                        st.markdown(
                            f'<div class="source-card"><b>Chunk {i}:</b><br>{chunk[:400]}{"..." if len(chunk) > 400 else ""}</div>',
                            unsafe_allow_html=True
                        )


# ── Chat Input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input(
    "Ask a question about your document...",
    disabled=not st.session_state.doc_loaded
):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Get answer from backend
    with st.chat_message("assistant"):
        with st.spinner("Searching document and generating answer..."):
            result, code = query_document(prompt)

        if code == 200:
            answer = result.get("answer", "No answer returned.")
            sources = result.get("source_chunks", [])
            st.write(answer)

            if sources:
                with st.expander(f"📎 Source chunks ({len(sources)} retrieved)"):
                    for i, chunk in enumerate(sources, 1):
                        st.markdown(
                            f'<div class="source-card"><b>Chunk {i}:</b><br>{chunk[:400]}{"..." if len(chunk) > 400 else ""}</div>',
                            unsafe_allow_html=True
                        )

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })
        else:
            error_msg = result.get("detail", "Something went wrong.")
            st.error(f"❌ {error_msg}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Error: {error_msg}"
            })
