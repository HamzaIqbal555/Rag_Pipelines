# RAG Mastery 🚀
## Advanced Retrieval-Augmented Generation (RAG) Implementations

This repository showcases **multiple RAG implementations from scratch and using LangChain**, designed for document analysis, research assistance, and real-time AI interactions. It processes **research papers on AI ethics, agents, smart spaces, and more** stored in `data/papers/`.

Built with **Python 3.12**, **LangChain**, **Chroma vector DB**, **HuggingFace embeddings**, and **local LLMs via Ollama**.

## ✨ Features
- **Multi-Document RAG**: Query across multiple PDFs with persistent Chroma DB.
- **Agentic RAG**: LLM agents that decide to search/retrieve from documents.
- **From-Scratch RAG**: Custom FAISS index + SentenceTransformers (no LangChain).
- **GraphRAG**: Graph-based RAG with MistralAI.
- **Realtime AI Assistant**: Streaming chat with web search tools (Tavily/DDG) + RAG.
- **Document Analyzer**: Utility for PDF/text processing.
- **Local-first**: Runs with Ollama (no cloud API costs).

## 📦 Quick Start
1. **Clone & Navigate**:
   ```bash
   git clone <repo-url>
   cd Rag_Mastery
   ```

2. **Setup Python Environment** (uses `uv` for fast installs):
   ```bash
   # Install uv if needed: pipx install uv
   uv sync  # Installs all deps from pyproject.toml
   # Or: uv venv && source .venv/bin/activate && uv pip install -e .
   ```

3. **Install & Run Ollama** (required for local LLMs):
   ```bash
   # Download: https://ollama.com
   ollama pull llama3.2  # ~2GB, lightweight
   ollama serve  # Run in background
   ```

4. **(Optional) API Keys** for tools:
   ```bash
   # Create .env
   echo "TAVILY_API_KEY=your_key_here" > .env  # For web search in Realtime_AI_Assistant.py
   # Get free key: https://tavily.com
   ```

5. **Pre-build Vector DB** (for multi-doc RAG):
   ```bash
   python Multi_document_Rag.py  # Loads papers, builds ./chroma_db/
   ```

6. **Run Examples**:
   ```bash
   # Multi-doc RAG
   python Multi_document_Rag.py

   # Agentic RAG
   python Agentic_Rag.py

   # Realtime Assistant (interactive)
   python Realtime_AI_Assistant.py

   # From scratch RAG
   python Rag_from_scratch.py
   ```

## 🛠 Detailed Setup
- **Python**: 3.12 (see `.python-version`)
- **Dependencies**: Managed via `pyproject.toml` (LangChain 0.2+, Chroma, Ollama, HF Embeddings, FAISS, etc.)
  ```bash
  uv add langchain-chroma langchain-ollama  # Add more if needed
  ```
- **Vector DB**: `./chroma_db/` - Persistent SQLite-based Chroma collections.
- **Embeddings**: `all-MiniLM-L6-v2` (local, fast).

**No GPU required** - Runs on CPU.

## 📁 Project Structure
```
Rag_Mastery/
├── data/                    # Input documents
│   └── papers/              # Research PDFs (AI ethics, agents, smart spaces)
├── chroma_db/               # Persistent vector store (auto-generated)
├── Agentic_Rag.py           # Agentic RAG with local pipeline LLM
├── Multi_document_Rag.py    # Multi-PDF RAG (main demo)
├── Rag_from_scratch.py      # FAISS + SentenceTransformers (no LangChain)
├── GraphRag_from_scratch.py # GraphRAG prototype
├── Realtime_AI_Assistant.py # Streaming chat + tools + RAG
├── Research_AI-Assistant.py # Research-focused assistant
├── Document_Analyzer.py     # PDF/text utils
├── pyproject.toml           # Deps (uv)
├── requirements.txt         # Legacy (use pyproject.toml)
├── README.md
└── ...
```

## 🚀 Usage Examples

### 1. Multi-Document RAG
Loads all PDFs, chunks, embeds, queries with Ollama.
```
$ python Multi_document_Rag.py
Enter query: What do papers say about AI agents trustworthiness?
> Retrieved context from papers... Answer: ...
```

### 2. Agentic RAG
Agent decides whether to search documents.
```
$ python Agentic_Rag.py
Query: Trust in AI agents? → Agent: SEARCH → RAG response...
```

### 3. Realtime Assistant
Interactive chat with RAG + web search.
```
$ python Realtime_AI_Assistant.py
You: Summarize latest AI ethics papers → Streams response...
```

## 🔍 Data
- **Papers**: 6+ academic PDFs on AI agents ethics, smart spaces, LLM performance, attacks on LLM robots.
- Add your docs to `data/papers/` and re-run `Multi_document_Rag.py` to rebuild DB.

## ⚠️ Notes
- **Ollama Required**: Install models like `llama3.2` or `phi3`.
- **Tavily Key**: Free tier for web search (optional).
- **Performance**: Embeddings on CPU ~1-2s/query. GPU accelerates.
- **Customization**: Edit prompts/embed models in scripts.
- **Clean DB**: `rm -rf chroma_db/` to reset.

## 🤝 Contributing
Fork, add new RAG variants (e.g., Pinecone, ensemble retrievers), PRs welcome!

## 📄 License
MIT (or add your own).

**Happy RAGging! 🎯**

