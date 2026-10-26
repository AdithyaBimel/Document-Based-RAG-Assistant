# RAG Document Assistant

Ask questions about any PDF or text document using local AI or free APIs.

## Stack
- **Frontend**: Streamlit
- **Backend**: FastAPI
- **RAG Core**: LangChain + FAISS
- **Embeddings**: sentence-transformers (runs on CPU)
- **LLM**: Groq API (free) / OpenAI API / Mistral 7B local (RTX 4050)

## Setup

### 1. Create virtual environment
```bash
python -m venv rag-env
rag-env\Scripts\activate        # Windows
source rag-env/bin/activate     # Mac/Linux
```

### 2. Install PyTorch (CUDA 12 for RTX 4050)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
pip install langchain-groq      # if using Groq (recommended)
```

### 4. Configure environment
```bash
cp .env.example .env
# Edit .env — add your GROQ_API_KEY (free at console.groq.com)
```

### 5. Run the project (two terminals in VSCode)

**Terminal 1 — Backend:**
```bash
uvicorn backend:app --reload
```

**Terminal 2 — Frontend:**
```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## LLM Modes

| Mode | Setup | Cost | Speed |
|------|-------|------|-------|
| `groq` | Free API key from console.groq.com | Free | Fast |
| `openai` | OpenAI API key | ~$0.01/query | Fast |
| `local` | Downloads Mistral 7B (~4GB) | Free | ~4 sec/query |

Set `LLM_MODE` in your `.env` file.

## Project Structure
```
rag-project/
├── rag_pipeline.py   # Core RAG logic
├── backend.py        # FastAPI API
├── app.py            # Streamlit UI
├── requirements.txt
├── .env.example
├── documents/        # uploaded PDFs stored here
└── vectorstore/      # FAISS index stored here
```

## Resume Bullet
> Built an end-to-end RAG pipeline using LangChain, FAISS, and Mistral 7B (4-bit quantized on RTX 4050), enabling natural language Q&A over custom documents with source-grounded citations via a FastAPI + Streamlit interface.
