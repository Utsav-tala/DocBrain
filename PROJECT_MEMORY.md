# PROJECT MEMORY: DocBrain

## Current Status
- Project Phase: Agentic RAG Implementation (v4.2) — Output Quality & Source Link Integrity
- Last Updated: 2026-06-21

## High-Level Architecture
- Tech Stack: LangGraph, LangChain, ChromaDB, OpenAI (gpt-4o-mini), Streamlit.
- Core Flow:
  1. User Query → Rewrite Query (LLM) → Local Retrieval (ChromaDB + MMR)
  2. Evaluated by LangGraph ReAct Agent.
  3. If local context is missing/weak, Agent executes `web_search_langchain` (DuckDuckGo).
  4. Agent calls `scrape_url` on promising search results to extract full code examples.
  5. Agent generates raw answer.
  6. **`link_resolver.py`** scores retrieved docs + any web results actually used this turn
     by topical overlap with the question, verifies each URL resolves, returns top
     3-4 relevant links (NOT just "whatever ranked #1 in MMR").
  7. **`output_refiner.py`** strips the in-body `Source:` line (redundant now), strips any
     other unverified URL the LLM wrote into prose, and runs a groundedness check
     (LLM call, only if the answer has code blocks) against the retrieved context.
  8. UI renders the answer + a separate "For more understanding" button row from the
     resolved links — no inline source links in the answer body anymore.

## Task Tracker
- Completed:
  - v2 Categorical Data Ingestion (13,093 chunks, 12 metadata fields).
  - Built Web Search and URL Scraping tools to act as fallback safety nets.
  - Migrated from static LCEL chain to LangGraph ReAct Agent.
  - Implemented Streamlit UI with real-time streaming, auto-collapsing sidebar.
  - Built `output_refiner.py`: groundedness check, format compliance check, link sanitization.
  - Built `link_resolver.py`: relevance-scored, verified source link selection (fixes
    wrong/irrelevant source links — see Key Constraints below).
  - Fixed `convert_source_to_url()` to verify guessed URLs via HEAD request before
    returning them, falling back to a known-good index page on 404/mismatch.
  - Replaced pill-chip "Sources" UI section with a labeled "For more understanding"
    button row (multiple links, short topic-specific labels per button).
- In Progress: Testing refiner/link-resolver against a broader query set before
  moving to the evaluation harness.
- Next Up:
  - Build an automated Evaluation harness (`src/evaluation.py`) — now also needs to
    measure link relevance/precision, not just answer quality.
  - Expand GitHub issues fetch from 500 to 2000+.
  - Add specific keyword aliases (e.g., "output parser" → "structured_output") to classification rules.
  - Consider hybrid search (BM25 + dense) and a cross-encoder reranker — flagged as
    highest-ROI retrieval improvements, not yet implemented.

## Key Constraints & Preferences
- **Agentic RAG Philosophy**: The local database (ChromaDB) acts as the high-speed, highly-curated primary brain. Web Search acts strictly as the ultimate safety net to prevent hallucinations or "I don't know" answers when documentation is missing or outdated.
- **Response Format**: The Agent's prompts must strictly enforce a structural pattern (Definition, Problem/Solution, How it works, Code Examples, Decision Guide). The trailing `Source:` line is NO LONGER part of the body — it's stripped by `output_refiner.py` and replaced by the separate link row in the UI.
- **Source link correctness (CRITICAL — self-correction note)**: `convert_source_to_url()` originally GUESSED a live URL from the local file path with no verification and no relevance check. This caused two confirmed bugs: (1) topically wrong source links (e.g. a "Philosophy" doc linked for an "LCEL" question, just because it ranked #1 in MMR/priority — not because it was relevant), and (2) the LLM writing a plausible-looking but hallucinated "Read full documentation" URL into the answer body that was never actually verified. Fix: links are now selected by `link_resolver.py` based on keyword overlap with the actual question (not just retrieval rank), every URL is HEAD-verified before being shown, and `output_refiner.py` strips any other unverified URL the LLM writes into the answer text. Do not reintroduce un-verified or rank-only source links.
- **Embeddings/Model**: Use `text-embedding-3-small` for vector storage and `gpt-4o-mini` as the primary LLM.
- **UI/UX**: The UI must remain sleek. Code blocks rendered natively, tokens stream live. Source links are NOT inline pill chips anymore — they're a heading ("For more understanding") above a row of gradient buttons, each with a short topic-specific label, multiple shown side by side.
- **Streaming limitation (known, accepted)**: The output refiner's link-sanitization can only run AFTER the full answer is generated. For the streaming chain, the live on-screen stream may briefly show an unverified URL before the stored/re-rendered version strips it. Not fixed — would require buffering the full response before streaming, which defeats the purpose of streaming.

## Project File Structure
- `data/`
  - Raw markdown files (LangChain docs) and Python source code (LangChain codebase) used for offline data ingestion.
- `db/chroma_langchain/`
  - Local ChromaDB vector database containing 13,093 embedded chunks and rich metadata.
- `src/` (Core Backend)
  - `ingest.py`: Categorical loader. Reads raw files, applies contextual chunking (prepends headers), adds 12 metadata fields, and embeds into ChromaDB.
  - `retriever.py`: Handles vector search. Classifies query intent, runs MMR retrieval, re-ranks results to prioritize core docs over integrations. `convert_source_to_url()` now HEAD-verifies guessed URLs before returning them.
  - `prompt_templates.py`: Contains 7 strict prompt templates. Dynamically routes prompts based on intent to enforce our structural pattern.
  - `chain.py`: The brain of the application. Builds the **LangGraph ReAct Agent** (`create_react_agent`), handles memory, rewrites queries, extracts web results the agent actually used (`extract_web_results`), calls `link_resolver` + `output_refiner`, yields the live token stream.
  - `tools.py`: Defines the tools available to the Agent (`web_search_langchain` via duckduckgo and `scrape_url` via BeautifulSoup).
  - `link_resolver.py`: **(new)** Builds the relevance-scored, verified, deduped link list shown under each answer. Scores local docs + web results by keyword overlap with the actual question, not just retrieval rank.
  - `output_refiner.py`: Post-generation checks — groundedness (LLM call, code-block answers only), format compliance (regex), and link sanitization (strips the in-body `Source:` line and any unverified inline URL).
- `app.py` (Frontend)
  - Streamlit UI entry point. Manages session state (history, user preferences), handles UI layout (auto-collapsed sidebar), receives live streaming tokens from the Agent, stores resolved `links` per message.
- `ui_components.py`
  - Contains all custom CSS and UI components. `render_source_links()` (replaces old `render_citation_chips()`) renders the "For more understanding" heading + button row.

## Instructions for AI
1. Before every response, check the "Task Tracker" in this file.
2. If I ask a technical question, reference the codebase files AND this memory file.
3. If a task is completed, update the "Completed" section and shift the "In Progress" task.
4. If I give a code correction, log it under "Key Constraints & Preferences" so the mistake isn't repeated.