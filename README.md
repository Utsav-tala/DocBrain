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
├── .env.example            # Environment variable template
├── src/
│   ├── ingest.py           # Document ingestion & chunking pipeline
│   ├── retriever.py        # Retrieval logic (MMR, hybrid search)
│   ├── chain.py            # LangChain RAG chain
│   ├── prompt_templates.py # System & user prompt templates
│   ├── output_refiner.py   # Post-processing LLM output
│   ├── link_resolver.py    # Resolve doc links to full URLs
│   ├── tools.py            # LangChain tool wrappers
│   └── evaluation.py       # RAGAS evaluation pipeline
├── data/                   # Raw documentation (gitignored)
└── db/                     # ChromaDB vector store (gitignored)
```

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

### 5. Ingest documentation

```bash
python -m src.ingest
```

This fetches LangChain docs, chunks them, generates embeddings, and stores them in ChromaDB.

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

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key for LLM & embeddings |
| `GOOGLE_API_KEY` | Google Gemini API key (optional fallback) |

See [`.env.example`](.env.example) for the full list.

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
