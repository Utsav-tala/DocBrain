# 🧠 DocBrain — AI-Powered Developer Documentation Assistant

> *"Stop reading docs. Start asking questions."*

DocBrain is a **RAG (Retrieval-Augmented Generation)** application that lets developers query LangChain documentation using natural language. Instead of skimming through pages of docs, just ask — and get precise, cited answers backed by real documentation.

---

## ✨ Features

- 🔍 **Semantic Search** — Finds the most relevant doc chunks using ChromaDB vector store
- 🤖 **LLM-Powered Answers** — Uses OpenAI / Gemini to synthesize concise, accurate responses
- 📎 **Source Citations** — Every answer links back to the original documentation
- 🔗 **Link Resolution** — Resolves relative doc links to full navigable URLs
- ✍️ **Output Refinement** — Post-processes LLM output for clarity and formatting
- 📊 **Evaluation Suite** — RAGAS-based evaluation pipeline for faithfulness, relevance & context recall
- 🖥️ **Streamlit UI** — Clean, interactive chat interface

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Streamlit  │────▶│  Retriever (RAG) │────▶│  LLM (OpenAI /  │
│     UI      │     │  ChromaDB + MMR  │     │     Gemini)     │
└─────────────┘     └──────────────────┘     └─────────────────┘
                            │                         │
                            ▼                         ▼
                    ┌──────────────┐         ┌───────────────────┐
                    │ Vector Store │         │  Output Refiner + │
                    │  (ChromaDB)  │         │  Link Resolver    │
                    └──────────────┘         └───────────────────┘
                            ▲
                    ┌───────┴──────┐
                    │  Ingest      │
                    │  Pipeline    │
                    │  (ingest.py) │
                    └──────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | OpenAI GPT-4o / Google Gemini |
| **Embeddings** | OpenAI text-embedding-3-small |
| **Vector Store** | ChromaDB |
| **Orchestration** | LangChain |
| **UI** | Streamlit |
| **Evaluation** | RAGAS + Datasets |
| **Language** | Python 3.10+ |

---

## 📁 Project Structure

```
docbrain/
├── app.py                  # Streamlit entry point
├── ui_components.py        # UI helper components
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template (safe — no real keys)
├── src/
│   ├── ingest.py           # Document ingestion & chunking pipeline
│   ├── retriever.py        # Retrieval logic (MMR, hybrid search)
│   ├── chain.py            # LangChain RAG chain
│   ├── prompt_templates.py # System & user prompt templates
│   ├── output_refiner.py   # Post-processing LLM output
│   ├── link_resolver.py    # Resolve doc links to full URLs
│   ├── tools.py            # LangChain tool wrappers
│   └── evaluation.py       # RAGAS evaluation pipeline
├── data/                   # ⚠️ Gitignored — regenerate via ingest.py (see below)
└── db/                     # ⚠️ Gitignored — regenerate via ingest.py (see below)
```

> **`data/` and `db/` are not included in this repository.**
> See [Why are data/ and db/ excluded?](#-why-are-data-and-db-excluded) below.

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/docbrain.git
cd docbrain
```

### 2. Set up a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your API keys:

```env
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
```

### 5. Ingest documentation (builds `data/` and `db/`)

```bash
python -m src.ingest
```

This will:
1. **Fetch** LangChain conceptual docs, API reference, and GitHub issues
2. **Chunk** documents into overlapping text segments
3. **Embed** each chunk using OpenAI `text-embedding-3-small`
4. **Store** vectors into ChromaDB under `db/chroma_langchain/`

⏱ *Expect ~10–20 min on first run depending on data source size and API throughput.*

> ⚠️ This step costs OpenAI API credits (embedding ~13k chunks ≈ a few cents).

### 6. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧪 Evaluation

DocBrain includes a RAGAS-based evaluation suite to measure:

- **Faithfulness** — Does the answer stay true to the retrieved context?
- **Answer Relevancy** — Is the answer relevant to the question?
- **Context Recall** — How much of the ground-truth is covered?

```bash
python -m src.evaluation
```

---

## 📄 Environment Variables

Copy `.env.example` → `.env` and fill in your values. **Never commit `.env`.**

| Variable | Required | Description | Where to get it |
|----------|----------|-------------|----------------|
| `OPENAI_API_KEY` | ✅ Yes | LLM responses + embeddings | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| `GOOGLE_API_KEY` | Optional | Gemini LLM fallback | [aistudio.google.com](https://aistudio.google.com/app/apikey) |
| `MODEL_NAME` | Optional | LLM to use (`gpt-4o-mini`, `gpt-4o`, `gemini-1.5-pro`) | — |
| `GITHUB_TOKEN` | Optional | Ingest GitHub Issues as data source | [github.com/settings/tokens](https://github.com/settings/tokens) (scope: `public_repo`) |

See [`.env.example`](.env.example) for the annotated template.

---

## 🗂️ Why are `data/` and `db/` excluded?

These directories are **gitignored intentionally**:

| Directory | Contents | Disk size | Why excluded |
|-----------|----------|-----------|--------------|
| `data/` | Raw scraped docs (HTML/JSON) | ~1 GB | Regeneratable; GitHub has a 100 MB per-file hard limit |
| `db/` | ChromaDB vector store (~13k chunks) | ~200 MB | Binary blobs change every re-embed; GitHub hard-rejects files >100 MB |

**To regenerate after cloning:**

```bash
# 1. Make sure your .env is configured with OPENAI_API_KEY
# 2. Run the ingest pipeline — it fetches docs + builds the vector store
python -m src.ingest
```

Both directories will be created automatically. No manual download needed.

---

## 🗺️ Roadmap

- [ ] Multi-framework support (FastAPI, Django, SQLAlchemy docs)
- [ ] Streaming responses in UI
- [ ] Persistent conversation memory
- [ ] Self-updating ingestion pipeline (detect doc changes)
- [ ] Docker deployment
- [ ] Hosted demo on Streamlit Cloud

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a PR.

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.
