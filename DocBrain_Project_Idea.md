# DocBrain — AI-Powered Developer Documentation Assistant

> *"Stop reading docs. Start asking questions."*

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Solution](#3-solution)
4. [How It Works — Plain English](#4-how-it-works--plain-english)
5. [System Architecture](#5-system-architecture)
6. [Tech Stack](#6-tech-stack)
7. [Skills You Will Learn](#7-skills-you-will-learn)
8. [Data Sources](#8-data-sources)
9. [Project Structure](#9-project-structure)
10. [Features — MVP](#10-features--mvp)
11. [Features — Optional Enhancements](#11-features--optional-enhancements)
12. [Pipeline — Step by Step](#12-pipeline--step-by-step)
13. [Code Skeleton](#13-code-skeleton)
14. [Evaluation Strategy](#14-evaluation-strategy)
15. [Deployment Plan](#15-deployment-plan)
16. [Handling Documentation Updates](#16-handling-documentation-updates)
17. [Milestones and Timeline](#17-milestones-and-timeline)
18. [Success Metrics](#18-success-metrics)
19. [Resume / Portfolio Framing](#19-resume--portfolio-framing)
20. [Future Scope](#20-future-scope)

---

## 1. Project Overview

**Project Name:** DocBrain

**Tagline:** An AI chatbot that reads developer documentation so you don't have to.

**Type:** RAG (Retrieval-Augmented Generation) Application

**Difficulty:** Beginner-Intermediate

**Estimated Build Time:** 2 weeks

**Target Audience (who will use this):**
- Developers learning a new framework (LangChain, FastAPI, React, etc.)
- Developers who know what they want to do but don't know the exact syntax or API
- Anyone who finds official documentation too long, scattered, or hard to navigate

---

## 2. Problem Statement

When a developer is building something, they constantly need to refer to documentation. For example:

- "How do I load a PDF file in LangChain?"
- "What are all the types of document loaders available?"
- "How do I connect a retriever to a chat model?"

The problem is that documentation is:

- **Long** — hundreds of pages spread across many sections
- **Hard to search** — keyword search returns irrelevant results
- **Scattered** — the answer to one question lives across 3 different pages
- **Time-consuming** — you read paragraphs just to find one line of code

Developers waste significant time just finding answers inside documentation, especially when they are still learning a framework.

---

## 3. Solution

DocBrain is a chatbot that:

1. Has already **read the entire documentation** of a framework (e.g., LangChain)
2. When you ask a question, it **finds the most relevant sections** from those docs
3. Passes those sections to an AI model and **generates a clear, direct answer**
4. Shows you **which exact doc pages** the answer came from, so you can verify

Think of it as a smart documentation search engine with a conversational interface — you ask in plain English, you get a precise answer with sources.

---

## 4. How It Works — Plain English

The system works in two phases:

### Phase 1: Setup (runs once, done by the developer)

1. Download the entire documentation of the target framework (e.g., LangChain docs from their GitHub or website)
2. Break the large documentation into small paragraphs / sections called **chunks**
3. Convert each chunk into a list of numbers (called an **embedding**) that captures the *meaning* of that chunk
4. Store all these embeddings in a **vector database** (like a smart library catalog)

This phase is offline and only runs once (or when docs are updated).

### Phase 2: User Query (runs every time a user asks something)

1. User types a question: *"What are the types of document loaders in LangChain?"*
2. The system converts this question into an embedding (same process as step 3 above)
3. It searches the vector database for chunks whose embeddings are **most similar** to the question's embedding — these are the most relevant documentation sections
4. Those chunks are passed to an AI model (GPT or Gemini) along with the question
5. The AI model reads the chunks and generates a clear answer
6. The user sees the answer + the source documentation pages it came from

This is RAG — **R**etrieval **A**ugmented **G**eneration:
- **Retrieval** = finding the right chunks from the documentation
- **Augmented** = giving those chunks to the AI as extra context
- **Generation** = the AI generating a final answer using that context

---

## 5. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        PHASE 1: INGESTION                        │
│                         (One-time setup)                         │
│                                                                   │
│   LangChain Docs  ──►  Chunking  ──►  Embedding  ──►  ChromaDB  │
│   (HTML/Markdown)       (split)      (OpenAI API)   (Vector DB)  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      PHASE 2: QUERY FLOW                         │
│                      (Every user query)                           │
│                                                                   │
│   User Question                                                   │
│        │                                                          │
│        ▼                                                          │
│   Embed Question  ──►  Search ChromaDB  ──►  Top-K Chunks        │
│   (OpenAI API)         (Similarity)         (Relevant Docs)      │
│                                                  │                │
│                                                  ▼                │
│                              Prompt = Question + Chunks           │
│                                                  │                │
│                                                  ▼                │
│                              LLM (GPT-4o-mini / Gemini Flash)    │
│                                                  │                │
│                                                  ▼                │
│                              Answer + Source Citations            │
│                                                  │                │
│                                                  ▼                │
│                              Streamlit Chat UI  ──►  User        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Tech Stack

### Core Framework
| Tool | Purpose | Why This Tool |
|------|---------|---------------|
| **LangChain** | Orchestration (loaders, splitters, chains, retrievers) | Industry standard, great for learning RAG patterns |
| **Python 3.10+** | Primary language | Ecosystem support for all AI/ML tools |

### Embeddings and LLMs
| Tool | Purpose | Why This Tool |
|------|---------|---------------|
| **OpenAI `text-embedding-3-small`** | Convert text to embeddings | Cheap, high quality, widely used |
| **GPT-4o-mini** | Answer generation (primary) | Fast and cheap for development/demo |
| **Gemini 1.5 Flash** | Answer generation (comparison/fallback) | You already have the API key; good to compare |

### Vector Database
| Tool | Purpose | Why This Tool |
|------|---------|---------------|
| **ChromaDB** | Store and search embeddings | Simplest to set up locally, persists to disk, free |

### Data Ingestion
| Tool | Purpose | Why This Tool |
|------|---------|---------------|
| **LangChain Document Loaders** | Load HTML/Markdown files | Built-in, no extra code needed |
| **BeautifulSoup4** | Clean scraped HTML | Lightweight HTML parser |
| **requests / wget** | Download documentation pages | Simple scraping |

### UI
| Tool | Purpose | Why This Tool |
|------|---------|---------------|
| **Streamlit** | Chat interface | Fastest way to build a deployable web UI in Python |

### Evaluation
| Tool | Purpose | Why This Tool |
|------|---------|---------------|
| **RAGAS** | Automated RAG evaluation metrics | Industry standard for RAG evaluation |
| **pandas** | Store and display eval results | Easy tabular data handling |

### Deployment
| Tool | Purpose | Why This Tool |
|------|---------|---------------|
| **Streamlit Community Cloud** | Host the app for free | Zero config, one-click deploy from GitHub |
| **HuggingFace Spaces** | Alternative free hosting | Also supports Streamlit apps |

### Environment and Utilities
| Tool | Purpose |
|------|---------|
| `python-dotenv` | Manage API keys via `.env` file |
| `tiktoken` | Count tokens before sending to LLM |
| `tqdm` | Progress bars during ingestion |
| `loguru` | Clean logging during pipeline runs |

---

## 7. Skills You Will Learn

By building DocBrain, you will gain hands-on practice with the following:

### LangChain Syntax (the main goal)
- `WebBaseLoader`, `DirectoryLoader`, `UnstructuredHTMLLoader` — loading documents
- `RecursiveCharacterTextSplitter` — chunking strategies, chunk size, overlap
- `OpenAIEmbeddings` — generating embeddings
- `Chroma.from_documents()`, `Chroma.load()`, `as_retriever()` — vector store operations
- `similarity_search_with_score()` — understanding retrieval scores
- `ChatPromptTemplate.from_messages()` — building structured prompts
- `create_retrieval_chain()` — modern LCEL-based RAG chain (interviewers ask about this)
- `RunnablePassthrough`, `RunnableLambda` — LCEL (LangChain Expression Language) basics

### General AI Engineering Skills
- How text embeddings work conceptually
- How vector similarity search works (cosine similarity)
- How to write a system prompt that forces citation and avoids hallucination
- How to build and run an evaluation set for a RAG system
- How to measure and improve retrieval quality by tuning chunk size and k

### Deployment Skills
- Deploying a Streamlit app to the cloud
- Managing secrets and API keys in deployment environments
- Writing a clear README with architecture diagram and results

---

## 8. Data Sources

### Primary Source: LangChain Documentation

**Option A — Scrape from website (recommended for freshness):**
- URL: `https://python.langchain.com/docs/`
- Use LangChain's `SitemapLoader` or `RecursiveUrlLoader` to crawl and download all pages
- Alternatively use `wget --mirror` to download as HTML files

**Option B — Use GitHub source (easiest, most stable):**
- LangChain's docs are on GitHub as `.mdx` / `.md` files
- Repository: `https://github.com/langchain-ai/langchain/tree/master/docs/docs`
- Clone the repo and load the markdown files directly — no scraping needed

**Recommended Approach for Beginners:** Clone the GitHub repo and use `DirectoryLoader` to load all `.md` files. This avoids HTML parsing complexity while you're learning.

### Volume
- LangChain docs have approximately 300-500 pages
- After chunking (chunk size ~800 tokens), you'll have roughly 2,000-5,000 chunks
- This is a manageable size for ChromaDB and a realistic but not overwhelming corpus

### Adding More Frameworks Later (v2)
Once LangChain docs are working, you can repeat the same ingestion pipeline for:
- FastAPI docs
- HuggingFace Transformers docs
- Any documentation site or GitHub-hosted docs

Use Chroma's `collection_name` parameter to store each framework's docs in a separate collection and let the user pick which framework to query.

---

## 9. Project Structure

```
docbrain/
│
├── data/                          # Raw documentation (downloaded, not committed to git)
│   └── langchain_docs/
│       └── *.md                   # Downloaded markdown files
│
├── db/                            # ChromaDB persistent storage (not committed to git)
│   └── chroma_langchain/          # Vector store for LangChain docs
│
├── src/
│   ├── ingest.py                  # One-time script: download → chunk → embed → store
│   ├── retriever.py               # Build retriever from existing ChromaDB
│   ├── chain.py                   # RAG chain: retriever + prompt + LLM
│   ├── prompt_templates.py        # All prompt templates in one place
│   └── evaluation.py              # Eval set runner and RAGAS scoring
│
├── app.py                         # Streamlit UI — main entry point
│
├── eval_data/
│   └── test_questions.json        # Hand-written Q&A pairs for evaluation
│
├── results/
│   └── eval_results.csv           # Saved evaluation results
│
├── .env                           # API keys (never commit this)
├── .env.example                   # Template for env vars (commit this)
├── .gitignore                     # Ignore db/, data/, .env
├── requirements.txt               # All dependencies
└── README.md                      # Project documentation with results
```

---

## 10. Features — MVP

These are the features you must build for a working, demo-able project:

### 1. Documentation Ingestion Pipeline
- Download LangChain docs (one-time script)
- Chunk documents with configurable chunk size and overlap
- Generate embeddings and store in ChromaDB
- Log how many chunks were created and stored

### 2. Retrieval System
- Load existing ChromaDB on app startup
- On each query, retrieve top-4 most relevant chunks using similarity search
- Return chunks along with their source URL/filename and similarity score

### 3. Answer Generation with Citations
- Build a prompt that instructs the LLM to:
  - Answer using ONLY the provided context
  - Always cite which documentation section the answer came from
  - Say "I couldn't find this in the documentation" if context is insufficient — no hallucination
- Support both GPT-4o-mini and Gemini 1.5 Flash (configurable via environment variable)

### 4. Streamlit Chat Interface
- Clean chat UI with message history
- Each answer displays:
  - The AI's answer
  - Source documents used (filename and section heading)
  - A toggle to expand and read the raw retrieved chunks
- A sidebar showing: which model is active, which framework is loaded

### 5. Basic Evaluation Script
- A JSON file with 20-30 hand-written question-answer pairs about LangChain
- A script that runs each question through the pipeline and scores:
  - Whether the correct documentation section was retrieved (retrieval hit rate)
  - Whether key expected keywords appear in the answer

---

## 11. Features — Optional Enhancements

Build these after MVP is complete, in order of impact:

### Enhancement 1: Multi-Framework Support
- Add separate ChromaDB collections for other frameworks (FastAPI, HuggingFace, etc.)
- Add a framework selector dropdown in the Streamlit sidebar
- The retriever switches to the selected framework's collection

### Enhancement 2: Retrieval Quality Improvements
- **MMR (Maximal Marginal Relevance):** Reduces redundancy in retrieved chunks — retrieves chunks that are relevant but also different from each other
  - Enable with: `vectorstore.as_retriever(search_type="mmr")`
- **Re-ranking:** After retrieving top-10 chunks, use a HuggingFace `cross-encoder` model to re-rank and keep only the top-4
  - Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (free, runs locally)

### Enhancement 3: RAGAS Automated Evaluation
- Integrate the RAGAS library for proper metric-based evaluation
- Measure: Faithfulness, Answer Relevancy, Context Precision, Context Recall
- Output a results table and include it in the README (this is impressive on a resume)

### Enhancement 4: Conversation Memory
- Allow multi-turn follow-up questions
  - Example: user asks "what is a retriever?" then follows with "how do I customize it?"
- Use LangChain's `ConversationBufferMemory` or the newer `RunnableWithMessageHistory`

### Enhancement 5: Chunk Size Experiment Panel
- In the sidebar, allow the user to select chunk size (500 / 800 / 1200 tokens)
- Show how different chunk sizes affect the retrieved results in real time
- This turns your app into an educational tool about RAG and shows depth of understanding

### Enhancement 6: Confidence Indicator
- Show the similarity score of each retrieved chunk as a percentage
- If the highest score is below a threshold (e.g., 0.75), show a warning: "Low confidence — this answer may not be fully grounded in the docs"

---

## 12. Pipeline — Step by Step

### Step 1: Download Documentation

```bash
# Option A: Clone LangChain's GitHub repo and use the docs folder
git clone --depth=1 https://github.com/langchain-ai/langchain.git
cp -r langchain/docs/docs/ data/langchain_docs/

# Option B: Use wget to mirror the website
wget --mirror --convert-links --no-parent -P data/langchain_web/ https://python.langchain.com/docs/
```

### Step 2: Load and Chunk

- Use `DirectoryLoader` with `UnstructuredMarkdownLoader` (or `TextLoader`) to load all `.md` files
- Use `RecursiveCharacterTextSplitter` with:
  - `chunk_size=800` (tokens, not characters — use `tiktoken` for accurate counting)
  - `chunk_overlap=100` (overlap between adjacent chunks to avoid cutting off context mid-sentence)
- Each chunk retains metadata: `source` (filename), `page` (if applicable)

**Why chunk at ~800 tokens?**
- Too small (e.g., 200): chunks lose context, retrieval becomes inaccurate
- Too large (e.g., 2000): too much irrelevant content gets passed to the LLM
- 800 is a good starting point; experiment with this

### Step 3: Embed and Store

- Use `OpenAIEmbeddings(model="text-embedding-3-small")` — cheap and accurate
- `Chroma.from_documents(chunks, embedding, persist_directory="db/chroma_langchain")`
- This generates one embedding per chunk and stores everything to disk
- Run time: approximately 5-15 minutes for the full LangChain docs, depending on API speed

### Step 4: Build the Retriever

- `vectorstore.as_retriever(search_kwargs={"k": 4})`
- This retriever, when called with a query string, returns the 4 most semantically similar chunks

### Step 5: Build the Prompt Template

```
System: You are a helpful documentation assistant for LangChain.
        Answer the user's question using ONLY the context below.
        Always cite the source document at the end of your answer.
        If the context does not contain the answer, say:
        "I could not find this in the LangChain documentation."
        Do not make up information.

Context:
{context}

Question: {question}

Answer:
```

### Step 6: Build the RAG Chain (LCEL Style)

Using LangChain's modern LCEL (LangChain Expression Language) approach:

```
question → retriever → relevant chunks
         ↘ (pass through)
           prompt template (question + chunks) → LLM → answer
```

### Step 7: Streamlit UI

- `st.chat_input()` for the user's question
- `st.chat_message()` for displaying conversation
- Show retrieved source documents in an expander below each answer
- Show loading spinner while retrieving and generating

### Step 8: Evaluate

- Run your 20-30 test questions through the pipeline
- Record: retrieved sources, generated answers
- Score manually or with RAGAS
- Document results in `results/eval_results.csv` and summarize in README

---

## 13. Code Skeleton

Below is a minimal skeleton — not complete code, but the key structure of each file to get started.

### `src/ingest.py`
```python
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

DATA_PATH = "data/langchain_docs/"
DB_PATH = "db/chroma_langchain"

def load_documents(path: str):
    loader = DirectoryLoader(path, glob="**/*.md", loader_cls=TextLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    return documents

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks

def embed_and_store(chunks):
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=DB_PATH,
        collection_name="langchain_docs"
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB at {DB_PATH}")
    return vectorstore

if __name__ == "__main__":
    docs = load_documents(DATA_PATH)
    chunks = chunk_documents(docs)
    embed_and_store(chunks)
```

### `src/retriever.py`
```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

DB_PATH = "db/chroma_langchain"

def get_retriever(k: int = 4):
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding,
        collection_name="langchain_docs"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever

def get_relevant_chunks(query: str, k: int = 4):
    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
        collection_name="langchain_docs"
    )
    results = vectorstore.similarity_search_with_score(query, k=k)
    return results   # list of (Document, score) tuples
```

### `src/prompt_templates.py`
```python
from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant for LangChain documentation.
Answer the user's question using ONLY the context provided below.
Always mention which documentation section your answer is from.
If the context does not contain enough information, say exactly:
"I could not find a clear answer in the LangChain documentation."
Never make up information that is not in the context.

Context:
{context}"""),
    ("human", "{question}")
])
```

### `src/chain.py`
```python
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.retriever import get_retriever
from src.prompt_templates import RAG_PROMPT

def format_docs(docs):
    return "\n\n".join([
        f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
        for doc in docs
    ])

def build_rag_chain(model_name: str = "gpt-4o-mini"):
    llm = ChatOpenAI(model=model_name, temperature=0)
    retriever = get_retriever(k=4)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain
```

### `app.py`
```python
import streamlit as st
from src.chain import build_rag_chain
from src.retriever import get_relevant_chunks

st.set_page_config(page_title="DocBrain", page_icon="🧠", layout="centered")
st.title("🧠 DocBrain — LangChain Docs Assistant")
st.caption("Ask anything about LangChain. Answers are grounded in the official documentation.")

# Initialize chain (cache it so it doesn't reload on every interaction)
@st.cache_resource
def load_chain():
    return build_rag_chain(model_name="gpt-4o-mini")

chain = load_chain()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask about LangChain..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching documentation..."):
            answer = chain.invoke(query)
            sources = get_relevant_chunks(query, k=4)

        st.markdown(answer)

        with st.expander("📄 Source Documents Used"):
            for doc, score in sources:
                st.markdown(f"**Source:** `{doc.metadata.get('source', 'Unknown')}`")
                st.markdown(f"**Similarity Score:** `{score:.3f}`")
                st.text(doc.page_content[:300] + "...")
                st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer})
```

### `src/evaluation.py`
```python
import json
import pandas as pd
from src.chain import build_rag_chain
from src.retriever import get_relevant_chunks

def run_evaluation(eval_path: str = "eval_data/test_questions.json"):
    with open(eval_path) as f:
        test_cases = json.load(f)

    chain = build_rag_chain()
    results = []

    for case in test_cases:
        question = case["question"]
        expected_keywords = case["expected_keywords"]
        expected_source = case.get("expected_source", "")

        answer = chain.invoke(question)
        sources = get_relevant_chunks(question, k=4)
        retrieved_sources = [doc.metadata.get("source", "") for doc, _ in sources]

        keyword_hit = any(kw.lower() in answer.lower() for kw in expected_keywords)
        source_hit = any(expected_source in src for src in retrieved_sources) if expected_source else None

        results.append({
            "question": question,
            "answer": answer,
            "keyword_hit": keyword_hit,
            "source_hit": source_hit,
            "retrieved_sources": ", ".join(retrieved_sources),
        })
        print(f"Q: {question[:60]}... | Keyword Hit: {keyword_hit} | Source Hit: {source_hit}")

    df = pd.DataFrame(results)
    df.to_csv("results/eval_results.csv", index=False)
    print(f"\nKeyword Hit Rate: {df['keyword_hit'].mean():.1%}")
    if df['source_hit'].notna().any():
        print(f"Source Hit Rate: {df['source_hit'].mean():.1%}")
    return df

if __name__ == "__main__":
    run_evaluation()
```

### `eval_data/test_questions.json`
```json
[
  {
    "question": "What are the types of document loaders available in LangChain?",
    "expected_keywords": ["PDF", "CSV", "WebBase", "loader"],
    "expected_source": "document_loaders"
  },
  {
    "question": "How do I split text into chunks in LangChain?",
    "expected_keywords": ["RecursiveCharacterTextSplitter", "chunk_size", "split_documents"],
    "expected_source": "text_splitters"
  },
  {
    "question": "How do I create a vector store in LangChain?",
    "expected_keywords": ["Chroma", "FAISS", "from_documents", "embeddings"],
    "expected_source": "vectorstores"
  },
  {
    "question": "What is LCEL in LangChain?",
    "expected_keywords": ["expression language", "pipe", "runnable", "|"],
    "expected_source": "expression_language"
  }
]
```

### `.env.example`
```
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_gemini_api_key_here
MODEL_NAME=gpt-4o-mini
```

### `requirements.txt`
```
langchain
langchain-openai
langchain-google-genai
langchain-community
chromadb
openai
streamlit
python-dotenv
unstructured
tiktoken
tqdm
loguru
ragas
datasets
pandas
beautifulsoup4
requests
```

---

## 14. Evaluation Strategy

Good RAG projects are not just "does it work?" — they show measured quality. This is what separates portfolio projects from tutorials.

### Evaluation Set
Build 20-30 question-answer pairs manually by:
1. Reading a section of LangChain docs
2. Writing a question a developer would naturally ask about that section
3. Writing the expected keywords or phrases that a good answer must contain
4. Noting the expected source document

Store in `eval_data/test_questions.json`.

### Metrics to Measure

| Metric | What It Means | How to Measure |
|--------|--------------|----------------|
| **Retrieval Hit Rate** | Did the right doc section appear in top-4 retrieved chunks? | Check if `expected_source` is in retrieved sources |
| **Keyword Hit Rate** | Did the answer contain expected keywords? | String matching |
| **Faithfulness** (with RAGAS) | Is the answer actually supported by the retrieved context? | RAGAS library |
| **Answer Relevancy** (with RAGAS) | Is the answer relevant to the question asked? | RAGAS library |

### Iteration Loop (Important)
Run eval → find weak spots → change one thing → run eval again → compare scores.

Things to experiment with and measure:
- Chunk size: 500 vs 800 vs 1200
- k (number of retrieved chunks): 3 vs 4 vs 6
- Prompt wording (strict citation vs relaxed)
- Embedding model: `text-embedding-3-small` vs `text-embedding-3-large`

Document every experiment and its result in your README. Showing that you improved from 60% to 85% retrieval hit rate through iteration is far more impressive than just building the tool.

---

## 15. Deployment Plan

### Step 1: Push to GitHub
- Make sure `.env`, `data/`, and `db/` are in `.gitignore`
- Write a clear `README.md` with: what the project does, architecture diagram, eval results, how to run locally

### Step 2: Deploy to Streamlit Community Cloud
1. Go to `share.streamlit.io`
2. Connect your GitHub repository
3. Set your `OPENAI_API_KEY` in the Streamlit secrets manager (instead of `.env`)
4. Click deploy — that's it

### Step 3: Host the Vector DB
- For demo purposes, you have two options:
  - **Commit a pre-built small DB**: Build the ChromaDB with a subset of docs (100-200 chunks) and commit it — this lets the demo work without needing the full ingestion pipeline on the server
  - **Use a cloud vector store**: Replace Chroma with **Pinecone Free Tier** or **Weaviate Cloud Free Tier** — these are hosted, so no local DB needed in deployment

For a portfolio demo, committing a small pre-built DB is perfectly fine and much simpler.

---

## 16. Handling Documentation Updates

LangChain (and any framework) updates its documentation regularly. Here's how to handle this:

### For the Portfolio Project (Good Enough)
- Re-run `ingest.py` manually whenever you want to update
- Mention in your README: *"This system uses documentation indexed as of [date]. The ingestion pipeline can be re-run at any time to pick up new docs."*

### Production-Ready Approach (Mention in README as Future Work)

**Incremental Update Strategy:**
1. For each chunk, compute a hash of its content and store it as metadata alongside the chunk in ChromaDB
2. When re-running ingestion, re-scrape the docs and compute hashes for new content
3. Compare new hashes to existing stored hashes
4. Only re-embed and upsert chunks that have changed (use Chroma's `upsert` method with a stable chunk ID)
5. Delete chunks whose IDs no longer appear in the new docs (for removed pages)

**Automation:**
- Schedule the ingestion script to run weekly using:
  - A simple cron job on a server
  - GitHub Actions with a `schedule:` trigger (runs for free)
  - Any cloud function on a timer

**Mention this in your README** — it shows you understand real-world production considerations without spending time building it.

---

## 17. Milestones and Timeline

### Week 1: Build the Core Pipeline

| Day | Tasks |
|-----|-------|
| Day 1 | Set up project structure, install dependencies, configure `.env`, test API keys |
| Day 2 | Download LangChain docs (clone GitHub repo), run `DirectoryLoader`, verify documents load |
| Day 3 | Implement chunking, experiment with chunk sizes, print sample chunks to verify quality |
| Day 4 | Implement embedding + ChromaDB storage, run `ingest.py` end to end |
| Day 5 | Build `retriever.py`, test `similarity_search_with_score` manually with 5-10 questions |
| Day 6 | Build `chain.py` with LCEL, test full RAG chain in a Python script (no UI yet) |
| Day 7 | Refine prompt template, test edge cases (unknown question, ambiguous question) |

### Week 2: UI, Evaluation, and Deployment

| Day | Tasks |
|-----|-------|
| Day 8 | Build Streamlit UI with chat interface and source expander |
| Day 9 | Polish UI: sidebar with model info, loading state, error handling |
| Day 10 | Write 20-30 evaluation questions, run `evaluation.py`, document results |
| Day 11 | Iterate on one variable (chunk size or k) based on eval results, re-measure |
| Day 12 | Write README with architecture diagram, eval results table, how-to-run instructions |
| Day 13 | Deploy to Streamlit Community Cloud, test live |
| Day 14 | Buffer / polish / add one optional enhancement if time allows |

---

## 18. Success Metrics

You can confidently say this project is successful and portfolio-ready when:

| Metric | Target |
|--------|--------|
| Retrieval Hit Rate (eval set) | > 80% |
| Keyword Hit Rate (eval set) | > 75% |
| Latency per query | < 5 seconds end-to-end |
| App deployed and publicly accessible | ✅ |
| README documents eval results and architecture | ✅ |
| At least one iteration documented (changed X, score improved from A to B) | ✅ |

---

## 19. Resume / Portfolio Framing

### One-Line Description
> "Built DocBrain, a RAG-based developer assistant that answers natural-language questions about LangChain documentation with cited sources, achieving 84% retrieval accuracy on a hand-built 25-question eval set."

### Bullet Points for Resume
- Designed and implemented an end-to-end RAG pipeline (ingestion → chunking → embedding → vector retrieval → generation) using LangChain and ChromaDB
- Integrated OpenAI `text-embedding-3-small` for semantic search over 3,000+ documentation chunks
- Built a Streamlit chat interface with source attribution, showing end-users which documentation sections grounded each answer
- Evaluated system using a 25-question hand-built test set; measured and improved retrieval hit rate from 67% to 84% by tuning chunk size and retrieval k
- Deployed publicly on Streamlit Community Cloud

### Interview Talking Points This Project Enables
- "Walk me through how RAG works" — you built it from scratch, you can explain every step
- "How did you evaluate your RAG system?" — you have real numbers and an eval methodology
- "What is LCEL?" — you used it in `chain.py`
- "What is a vector database?" — you chose Chroma, understand what it stores and why
- "How would you improve retrieval quality?" — you experimented with chunk size, k, and can discuss MMR and re-ranking
- "How would you handle knowledge freshness?" — you can describe the hash-based incremental update strategy

---

## 20. Future Scope

These ideas extend DocBrain into a more powerful system, suitable for a follow-up v2:

### Multi-Framework Support
Index documentation for multiple frameworks — LangChain, FastAPI, HuggingFace, React — and let the user choose which one to query from a dropdown. Each framework gets its own ChromaDB collection.

### Hybrid Retrieval
Combine semantic (vector) search with keyword (BM25) search for better results on exact API names and function names. LangChain provides `EnsembleRetriever` for this out of the box.

### Re-Ranking
After retrieving top-10 chunks, use a lightweight HuggingFace cross-encoder model to re-score and select the best 4 before passing to the LLM. Measurably improves answer quality.

### Automated RAGAS Evaluation Dashboard
Add a tab in the Streamlit app that runs the full RAGAS evaluation and displays a live metric dashboard — Faithfulness, Answer Relevancy, Context Precision, Context Recall.

### Conversational Memory
Allow follow-up questions that reference previous answers in the same session. Use LangChain's `RunnableWithMessageHistory`.

### Agentic Routing (v3)
Add an agent layer that decides: "Is this question answerable from the pre-indexed docs, or is it about something new enough that I should do a live web search?" This transitions the project from classic RAG into agentic RAG — a natural progression after mastering the fundamentals.

---

*Built with LangChain, ChromaDB, OpenAI, and Streamlit.*
*License: MIT*
