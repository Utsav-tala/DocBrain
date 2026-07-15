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
│   └── evaluation.py       # Retrieval eval harness — hit@k, recall@k, MRR
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

Retrieval quality is **measured, not asserted**. `src/evaluation.py` scores retrieval
against a hand-labelled golden set (`eval_data/golden_set.jsonl` — 32 covered + 6
coverage-gap questions) at **source-file granularity**, so labels stay stable across
re-embeds. The metrics are pure set comparison — no LLM judge — so they're
deterministic, free, and reproducible.

```bash
python -m src.evaluation --k 5 --label v5
```

Measured results (k=5):

| Version | hit@5 | recall@5 | MRR |
|---------|:-----:|:--------:|:---:|
| v4 — rerank on metadata alone | 0.875 | 0.833 | **0.788** |
| **v5 — relevance-first rerank** | **0.906** | **0.865** | 0.745 |

v5 finds the right document more often (hit@5 and recall@5 both up). MRR dipped
slightly — the right doc is found more but ranked *first* a little less — which points
at ranking order (a cross-encoder reranker) as the next lever.

**Coverage-gap detection.** Questions the corpus can't answer are scored on a
different question — *did the system notice it couldn't answer?* — by calibrating an
absolute `raw_distance` threshold. The current fit (`raw_distance ≥ 0.80`) catches
6/6 gap questions at 12/32 false alarms (Youden's J = 0.625) and is **wired into the
retriever** to force the web-search fallback (see `retriever.py:GAP_DISTANCE_THRESHOLD`).

Still to build (these need an LLM judge — tracked in the Roadmap):

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
roughly **2× worse**. That is the signal the calibrated `raw_distance ≥ 0.80`
threshold now uses to force the web-search fallback (see Evaluation), instead of
leaving the call to the agent. Live, this query is flagged as a gap and routed to
web search rather than answered from the lone docstring.

**3. The gap threshold is calibrated on very little data.**
The web-search fallback is now driven by a measured `raw_distance ≥ 0.80` threshold
rather than the agent's judgment — but it's fit on only 6 gap questions. It catches
all of them and false-alarms on ~1/3 of covered questions (Youden's J = 0.625). It
favours recall on gaps by design (a needless web search beats a confident
hallucination), but needs a larger golden set to tighten specificity.

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

**Evaluation — retrieval done, answer-quality next:**

- [x] Retrieval metrics: hit@k, recall@k, MRR against a labelled golden set
- [x] Calibrate an absolute distance threshold + wire it to the web-search fallback
- [ ] Grow the golden set (38 → ~150) to tighten the gap threshold's specificity
- [ ] Answer metrics: faithfulness, answer relevancy (needs an LLM judge)
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
