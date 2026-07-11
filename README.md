# 🧠 DocBrain — AI-Powered Developer Documentation Assistant

> *"Stop reading docs. Start asking questions."*

DocBrain is a **RAG (Retrieval-Augmented Generation)** application that lets developers query LangChain documentation using natural language. Instead of skimming through pages of docs, just ask — and get precise, cited answers backed by real documentation.

---

## ✨ Features

- 🔍 **Semantic Search** — Finds the most relevant doc chunks using a ChromaDB vector store
- 🎯 **Relevance-First Re-ranking** — Ranks on semantic relevance, with document priority
  and query intent as a bounded tiebreaker (never enough to outrank a better-matching chunk)
- 🧭 **Intent Routing** — Classifies the query (concept / code / error / migration / LangGraph)
  and routes it to a matching prompt template and preferred doc types
- 🤖 **Agentic Fallback** — A LangGraph ReAct agent falls back to web search + page scraping
  when the local corpus doesn't cover the question
- 📎 **Verified Source Links** — Every link is scored against the question and HEAD-checked
  before it's shown; unverified URLs the LLM invents are stripped from the answer
- ✍️ **Output Refinement** — Groundedness and format checks on the generated answer
- 🖥️ **Streamlit UI** — Clean, interactive chat interface with token streaming

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Streamlit  │────▶│  Retriever (RAG) │────▶│  LLM (OpenAI /  │
│     UI      │     │ ChromaDB + Rerank│     │     Gemini)     │
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
| **Agent** | LangGraph (ReAct) |
| **Orchestration** | LangChain |
| **UI** | Streamlit |
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
│   ├── retriever.py        # Intent routing + relevance-first re-ranking
│   ├── chain.py            # LangGraph ReAct agent + streaming
│   ├── prompt_templates.py # System & user prompt templates
│   ├── output_refiner.py   # Groundedness / format / link sanitization
│   ├── link_resolver.py    # Relevance-scored, HEAD-verified source links
│   ├── tools.py            # Agent tools (web search, page scraping)
│   └── evaluation.py       # ⚠️ Not implemented yet — see Roadmap
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

> **Status: not built yet.** `src/evaluation.py` is currently an empty placeholder.
> There are no measured quality numbers for this project, and any you see claimed
> elsewhere would be fabricated. This is the next major piece of work.

The planned harness measures:

- **Retrieval** — recall@k and MRR against a hand-labelled golden set
- **Faithfulness** — does the answer stay true to the retrieved context?
- **Answer Relevancy** — is the answer relevant to the question?
- **Cost & Latency** — tokens and p50/p95 per query, per config

---

## ⚠️ Known Limitations

Documented honestly rather than hidden — these are real, reproducible, and open.

**1. Corpus coverage gaps are invisible to the retriever.**
The relevance score is min-max normalized *within* each candidate pool, so the
top-ranked chunk always scores `1.0` — even when all 60 candidates are irrelevant.
The system cannot currently distinguish *"I found a great answer"* from *"this is
the least-bad of 60 bad chunks."* The raw distance is carried through in metadata
(`raw_distance`) but no threshold is calibrated against it yet.

**2. `"What is LCEL?"` is a known failure case.**
LangChain's current docs de-emphasized LCEL as a concept page, so the scraped corpus
contains no LCEL explainer — only a source-code docstring in
`langchain_core/runnables/__init__.py`. Retrieval correctly surfaces that docstring
as the best available chunk, but the rest of the context is noise. Typical distance
for this query is ~1.20 vs ~0.61 for a well-covered query like an `ImportError` —
roughly **2× worse**, which is exactly the signal a calibrated threshold should use
to trigger the web-search fallback instead of leaving that call to the agent.

**3. The web-search fallback fires on the agent's judgment, not a measurement.**
See (1) and (2) — this should be driven by an absolute relevance threshold.

**4. Streaming shows unverified links briefly.**
Link sanitization can only run after the full answer is generated, so the live
stream may flash a URL that the stored version later strips. Fixing this properly
would require buffering the whole response, which defeats streaming.

**5. Session history is in-memory.**
`_history_store` is a plain dict — conversations do not survive a restart.

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

**Next up — evaluation (the current priority):**

- [ ] Golden set: ~150 questions with hand-labelled correct source documents
- [ ] Retrieval metrics: recall@k, MRR
- [ ] Answer metrics: faithfulness, answer relevancy
- [ ] Calibrate an absolute relevance threshold to trigger the web-search fallback
- [ ] Ablation study: chunking, dense vs hybrid, reranker on/off — with cost & latency

**Then — retrieval quality:**

- [ ] Hybrid search (BM25 + dense) — dense retrieval is weak on exact symbols
      like `LLMChain` or `AttributeError`, which this corpus is full of
- [ ] Cross-encoder reranker

**Later:**

- [x] Streaming responses in UI
- [ ] Persistent conversation memory
- [ ] Multi-framework support (FastAPI, Django, SQLAlchemy docs)
- [ ] Self-updating ingestion pipeline (detect doc changes)
- [ ] Docker deployment
- [ ] Hosted demo on Streamlit Cloud

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a PR.

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.
