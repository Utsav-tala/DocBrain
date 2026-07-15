"""
app.py — DocBrain chat UI (native Streamlit chat primitives)

Rewritten from hand-rolled HTML bubbles to st.chat_message + st.write_stream:
  - st.chat_message() is a real DOM container → it wraps ALL content correctly
    (markdown, code blocks, source-link pills, feedback), so nothing spills
    edge-to-edge or detaches from its bubble.
  - st.write_stream() streams tokens through the SAME native markdown renderer used
    for the stored message → no reflow/jump when streaming ends.
  - Source links + a meta caption + thumbs feedback live INSIDE the assistant bubble.
  - Starter prompts are real buttons that submit; the empty-state input is normal size.
"""

import time
import uuid
import streamlit as st
from loguru import logger
from dotenv import load_dotenv

from ui_components import (
    DOCBRAIN_CSS,
    HEADER_HTML,
    render_hero,
    render_source_links,
    STARTER_QUERIES,
    SIDEBAR_LOGO,
    sidebar_section,
)

load_dotenv()

st.set_page_config(
    page_title="DocBrain",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)
st.markdown(DOCBRAIN_CSS, unsafe_allow_html=True)

# ── Ensure the vector DB exists (cloud cold-start) ────────────────────────────
# The prebuilt Chroma index isn't in git (chroma.sqlite3 alone exceeds GitHub's
# 100 MB per-file limit), so on Streamlit Cloud we lazy-download it from a GitHub
# Release on first run. Locally the folder already exists → this is a no-op.
from src.bootstrap_db import ensure_db, DB_SENTINEL
if not DB_SENTINEL.exists():
    with st.spinner("Downloading the knowledge base (first run only, ~1 min)…"):
        ensure_db()

USER_AVATAR = "🧑‍💻"
BOT_AVATAR  = "🧠"


# ── Cached chain ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_chain(model_name: str):
    from src.chain import build_streaming_chain
    return build_streaming_chain(model_name=model_name)


# ── Session state ─────────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "messages":    [],
        "query_count": 0,
        "model":       "gpt-4o-mini",
        "session_id":  str(uuid.uuid4()),
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

init_session()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(SIDEBAR_LOGO, unsafe_allow_html=True)

    st.markdown(sidebar_section("Model"), unsafe_allow_html=True)
    models = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]
    st.session_state.model = st.selectbox(
        "LLM", models,
        index=models.index(st.session_state.model),
        label_visibility="collapsed",
    )

    st.markdown(sidebar_section("Pipeline"), unsafe_allow_html=True)
    st.markdown(
        "<div class='db-cfg'>"
        "<div><span>Candidate pool</span><b>60</b></div>"
        "<div><span>Chunks → LLM</span><b>5</b></div>"
        "<div><span>Gap threshold</span><b>0.80</b></div>"
        "<div><span>Corpus</span><b>13,093 chunks</b></div>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(sidebar_section("Session"), unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Clear chat", use_container_width=True):
            from src.chain import clear_session_history
            clear_session_history(st.session_state.session_id)
            st.session_state.messages    = []
            st.session_state.query_count = 0
            st.session_state.session_id  = str(uuid.uuid4())
            st.rerun()
    with c2:
        st.metric("Queries", st.session_state.query_count)

    st.markdown(sidebar_section("About"), unsafe_allow_html=True)
    st.markdown(
        "<div class='db-about'>RAG over LangChain docs, source code, error references, "
        "migration guides & GitHub issues. Agentic web-search fallback fires when local "
        "coverage is weak.</div>",
        unsafe_allow_html=True,
    )


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(HEADER_HTML, unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _meta_caption(intent: str, elapsed, model: str) -> str:
    bits = []
    if intent:
        bits.append(intent.replace("_", " "))
    if elapsed is not None:
        bits.append(f"{elapsed}s")
    if model:
        bits.append(model)
    return " · ".join(bits)


def render_assistant_extras(msg: dict, idx: int):
    """Source-link pills + meta caption + thumbs feedback, inside the bubble."""
    links = msg.get("links", [])
    if links:
        st.markdown(render_source_links(links), unsafe_allow_html=True)
    caption = _meta_caption(msg.get("intent", ""), msg.get("elapsed"), msg.get("model", ""))
    if caption:
        st.caption(caption)
    st.feedback("thumbs", key=f"fb_{idx}")


# ── Resolve the incoming prompt (typed input OR a clicked starter) ────────────
typed = st.chat_input("Ask anything about LangChain, LangGraph, or LCEL…")
prompt = typed or st.session_state.pop("pending_query", None)
prompt = prompt.strip() if prompt else None

# Append the user turn BEFORE rendering, so the hero hides and history includes it.
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})


# ── Conversation OR empty-state hero ──────────────────────────────────────────
if not st.session_state.messages:
    st.markdown(render_hero(), unsafe_allow_html=True)
    st.markdown("<div class='db-starter-label'>Try asking</div>", unsafe_allow_html=True)
    cols = st.columns(2)
    for i, q in enumerate(STARTER_QUERIES):
        with cols[i % 2]:
            if st.button(q, key=f"starter_{i}", use_container_width=True):
                st.session_state.pending_query = q
                st.rerun()
else:
    for idx, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            with st.chat_message("user", avatar=USER_AVATAR):
                st.markdown(msg["content"])
                # hidden hook so CSS can tint the user's bubble (see .db-user-mark)
                st.markdown("<span class='db-user-mark'></span>", unsafe_allow_html=True)
        else:
            with st.chat_message("assistant", avatar=BOT_AVATAR):
                st.markdown(msg["content"])
                render_assistant_extras(msg, idx)


# ── Generate the assistant turn for a freshly-added user prompt ───────────────
if prompt:
    logger.info(f"[QUERY #{st.session_state.query_count + 1}] {prompt}")

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        try:
            chain  = get_chain(st.session_state.model)
            t0     = time.time()
            result = chain(prompt, session_id=st.session_state.session_id)

            # One native renderer for streaming AND storage → no reflow on completion.
            answer  = st.write_stream(result["stream"])
            elapsed = round(time.time() - t0, 1)

            # links_holder is mutated as a side effect of fully consuming the stream,
            # so it must be read AFTER st.write_stream returns. Source-link pills, the
            # meta caption and the feedback widget are rendered by the history path
            # (render_assistant_extras) after the rerun below — not inline — so the
            # completed turn has one canonical render with stable widget keys.
            links  = result["links"]["links"]
            intent = result.get("intent", "")
            logger.info(
                f"[RESPONSE] q#{st.session_state.query_count + 1} | elapsed={elapsed}s | "
                f"links={len(links)} | intent={intent} | model={st.session_state.model}"
            )

        except Exception as e:
            logger.error(f"[ERROR] {e}")
            answer  = (
                "Something went wrong.\n\n"
                f"**Error:** `{e}`\n\n"
                "Check your `.env` and that ChromaDB is populated (`python -m src.ingest`)."
            )
            st.markdown(answer)
            links, intent, elapsed = [], "", None

    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "links":   links,
        "intent":  intent,
        "elapsed": elapsed,
        "model":   st.session_state.model,
    })
    st.session_state.query_count += 1
    # Re-run so the completed turn re-renders through the history path (feedback widget
    # with a stable key) and the sidebar "Queries" count reflects this turn.
    st.rerun()
