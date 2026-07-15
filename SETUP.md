# Setup & Usage Guide

Everything you need to clone, configure, and run DocBrain locally — plus how to run
the evaluation harness and (optionally) rebuild the vector index from scratch.

For what the project *is* and how it works, see the [README](README.md).

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python 3.11** | The app is pinned to the versions in `requirements.txt`. 3.10+ works, 3.11 is what it's tested on. |
| **An OpenAI API key** | Used for both the chat model (`gpt-4o-mini` by default) and embeddings (`text-embedding-3-small`). Get one at [platform.openai.com/api-keys](https://platform.openai.com/api-keys). |
| **git** | To clone the repo. |

> 💡 A query costs a fraction of a cent on `gpt-4o-mini`. If you expose the app
> publicly, set a **usage limit** in the OpenAI dashboard first.

---

## Path A — Run with the prebuilt index (recommended)

You do **not** need to ingest anything. On first run the app downloads a prebuilt
ChromaDB index (~100 MB, 13,093 chunks) from the GitHub Release and caches it locally
— see [`src/bootstrap_db.py`](src/bootstrap_db.py).

```bash
# 1. Clone
git clone https://github.com/Utsav-tala/DocBrain.git
cd DocBrain

# 2. Create + activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install the (lean, pinned) serving dependencies
pip install -r requirements.txt

# 4. Provide your OpenAI key (either a .env file or an env var)
cp .env.example .env               # then edit .env and set OPENAI_API_KEY
#   — or —
export OPENAI_API_KEY=sk-...

# 5. Run
streamlit run app.py
```

Open **http://localhost:8501**. The first launch shows a one-time
*"Downloading the knowledge base…"* spinner while the index is fetched; subsequent
launches are instant.

---

## Path B — Rebuild the index from scratch (advanced)

Only needed if you want to change the corpus, chunking, or embeddings. The ingest
pipeline ([`src/ingest.py`](src/ingest.py)) reads the raw LangChain docs and source
tree from local folders and **only auto-fetches the GitHub issues** over the API.

**1. Place the raw corpus** under `data/` in the layout `ingest.py` expects
(see `CATEGORY_CONFIG` for the exact globs):

```
data/
├── langchain_conceptual_docs/src/oss/…   # LangChain / LangGraph conceptual docs (.mdx)
└── langchain_codebase/libs/core/langchain_core/…   # langchain-core source (.py)
```

> These raw dumps are **not bundled** (≈1 GB, and they're gitignored). Populate them
> by cloning the corresponding LangChain repositories into the paths above. The exact
> doc snapshot will differ from the shipped index — that's expected.

**2. Install the dev/ingest dependencies** (adds ingest + eval tooling on top of the
serving set):

```bash
pip install -r requirements-dev.txt
```

**3. Set your keys** in `.env`:

```env
OPENAI_API_KEY=sk-...
GITHUB_TOKEN=ghp_...   # optional — only to ingest GitHub issues (scope: public_repo)
```

**4. Build the index** (embeds ~13k chunks — costs a few cents of OpenAI credit):

```bash
python -m src.ingest
```

This writes the ChromaDB store to `db/chroma_langchain/`. Now run the app as in Path A
step 5 — since the index already exists locally, no download happens.

---

## Running the evaluation harness

Retrieval quality is measured against a hand-labelled golden set
(`eval_data/golden_set.jsonl`, 38 questions) at source-file granularity — pure set
comparison, no LLM judge. Requires the index (Path A or B) and `requirements-dev.txt`.

```bash
python -m src.evaluation --k 5 --label v5
```

This prints `hit@k`, `recall@k`, `MRR`, and the coverage-gap calibration, and saves a
JSON report to `results/eval_<label>.json`. See the
[Evaluation section of the README](README.md#-evaluation) for how to read the numbers.

---

## Environment variables

| Variable | Required | Purpose |
|----------|:--------:|---------|
| `OPENAI_API_KEY` | ✅ | Chat model + embeddings |
| `GITHUB_TOKEN` | — | Ingest GitHub issues (Path B only; scope `public_repo`) |
| `DOCBRAIN_DB_URL` | — | Override the prebuilt-index download URL (cloud/self-host) |

See [`.env.example`](.env.example) for the annotated template. **Never commit `.env`.**

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'langchain'` | The virtual environment isn't active. Run `source venv/bin/activate` first. |
| `Missing credentials … set the OPENAI_API_KEY` | The key isn't visible to the app. Check `.env` (or the env var / Streamlit secret) — the name must be exactly `OPENAI_API_KEY`. |
| First run hangs on *"Downloading the knowledge base"* | The index is being fetched from the GitHub Release (~100 MB). Give it up to a minute on a slow connection. |
| Answers are empty or off-topic | The index didn't materialize. Delete `db/` and relaunch to re-trigger the download, or rebuild via Path B. |

---

Deploying your own copy (Streamlit Community Cloud + a GitHub Release for the index)
follows the same two pieces: ship the app, and make the prebuilt index reachable via
`DOCBRAIN_DB_URL` or the default Release URL in `src/bootstrap_db.py`.
