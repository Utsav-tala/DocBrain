"""
app.py — DocBrain v3 UI
- User messages RIGHT side, DB messages LEFT side
- Code blocks rendered via st.code() (proper syntax highlighting, no HTML mess)
- Text parts rendered as HTML markdown
- Larger font sizes, cleaner layout
"""

import streamlit as st
import time
import os
from loguru import logger
from dotenv import load_dotenv
import uuid

from ui_components import (
    DOCBRAIN_CSS,
    split_answer_into_parts,
    render_source_links,
    render_thought_stream,
    render_empty_state,
    render_stats_bar,
    render_live_message,
    SIDEBAR_LOGO,
    sidebar_section,
)

load_dotenv()

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocBrain",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(DOCBRAIN_CSS, unsafe_allow_html=True)

# ── Cached Chain ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_chain(model_name: str):
    from src.chain import build_streaming_chain
    return build_streaming_chain(model_name=model_name)


# ── Session State ──────────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "messages":    [],
        "query_count": 0,
        "model":       "gpt-4o-mini",
        "fetch_k":     60,
        "top_k":       5,
        "loading":     False,
        "session_id":  str(uuid.uuid4()),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(SIDEBAR_LOGO, unsafe_allow_html=True)

    st.markdown(sidebar_section("Model"), unsafe_allow_html=True)
    model_choice = st.selectbox(
        "LLM",
        options=["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
        index=["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"].index(st.session_state.model),
        label_visibility="collapsed",
    )
    if model_choice != st.session_state.model:
        st.session_state.model = model_choice

    st.markdown(sidebar_section("Retrieval"), unsafe_allow_html=True)
    fetch_k = st.slider(
        "Candidate Pool (fetch_k)",
        min_value=20, max_value=100, step=10,
        value=st.session_state.fetch_k,
        help="MMR casts this many candidates before re-ranking.",
    )
    top_k = st.slider(
        "Chunks Returned (top_k)",
        min_value=3, max_value=10, step=1,
        value=st.session_state.top_k,
        help="Final chunks passed to the LLM.",
    )
    st.session_state.fetch_k = fetch_k
    st.session_state.top_k   = top_k

    st.markdown(sidebar_section("Session"), unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear chat", use_container_width=True):
            from src.chain import clear_session_history
            clear_session_history(st.session_state.session_id)
            st.session_state.messages    = []
            st.session_state.query_count = 0
            st.session_state.session_id  = str(uuid.uuid4())
            st.rerun()
    with col2:
        st.metric("Queries", st.session_state.query_count)

    st.markdown(sidebar_section("About"), unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:12px; color:#3D4560; line-height:1.6;'>"
        "RAG over LangChain docs, source code, error references, migration guides, "
        "and GitHub issues.<br>v3 pipeline · 13,093 chunks · 9 doc types."
        "</div>",
        unsafe_allow_html=True,
    )


# ── Main Chat Area ─────────────────────────────────────────────────────────────
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# Page header
st.markdown("""
<div class="page-header">
  <h1>🧠 DocBrain</h1>
  <p>LangChain developer documentation assistant · v3 pipeline</p>
</div>
""", unsafe_allow_html=True)


# ── Message Rendering ──────────────────────────────────────────────────────────

def render_assistant_message(msg: dict, idx: int):
    """
    Render one assistant turn:
      - Left-aligned bubble with DB avatar
      - Text rendered as HTML markdown
      - Code blocks rendered via st.code() (NO HTML — proper highlighting)
      - Verified source link buttons below
      - Feedback buttons
    """
    ts = msg.get("ts", "")

    # Open assistant row + avatar + body
    st.markdown(f"""
<div class="msg-assistant-row">
  <div class="msg-avatar">DB</div>
  <div class="msg-assistant-body">
    <div class="msg-meta">DocBrain · {ts}</div>
    <div class="msg-text">
""", unsafe_allow_html=True)

    # Split answer into html parts and code parts
    parts = split_answer_into_parts(msg["content"])
    for part in parts:
        if part["type"] == "html":
            st.markdown(part["content"], unsafe_allow_html=True)
        else:
            # st.code() gives proper syntax highlighting, copy button, clean display
            st.code(part["content"], language=part.get("lang", "python"))

    # Close msg-text div, add source link buttons, close body + row
    links_html = render_source_links(msg.get("links", []))
    st.markdown(f"""
    </div>
    {links_html}
  </div>
</div>
""", unsafe_allow_html=True)

    # Feedback buttons (must be real Streamlit widgets — outside the HTML)
    fb_state = msg.get("feedback", None)
    fb_col1, fb_col2, fb_col3 = st.columns([0.05, 0.05, 0.9])
    with fb_col1:
        label = "✅" if fb_state == "up" else "👍"
        if st.button(label, key=f"fb_up_{idx}", help="Good answer"):
            st.session_state.messages[idx]["feedback"] = "up"
            logger.info(f"[FEEDBACK] idx={idx} vote=up")
            st.rerun()
    with fb_col2:
        label = "❌" if fb_state == "down" else "👎"
        if st.button(label, key=f"fb_down_{idx}", help="Bad answer"):
            st.session_state.messages[idx]["feedback"] = "down"
            logger.info(f"[FEEDBACK] idx={idx} vote=down")
            st.rerun()


# ── Show conversation ──────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown(render_empty_state(), unsafe_allow_html=True)
else:
    for idx, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            # RIGHT side — just HTML, no widgets needed
            st.markdown(f"""
<div class="msg-user-row">
  <div class="msg-user-bubble">{msg["content"]}</div>
</div>
""", unsafe_allow_html=True)

        else:
            render_assistant_message(msg, idx)

            if idx < len(st.session_state.messages) - 1:
                st.markdown('<div class="turn-divider"></div>', unsafe_allow_html=True)


# ── Loading State ──────────────────────────────────────────────────────────────
if st.session_state.loading:
    st.markdown(f"""
<div class="msg-assistant-row">
  <div class="msg-avatar">DB</div>
  <div class="msg-assistant-body">
    {render_thought_stream()}
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # close .main-content


# ── Chat Input ─────────────────────────────────────────────────────────────────
user_input = st.chat_input(
    "Ask anything about LangChain, LangGraph, or LCEL…",
    key="chat_input",
)

if user_input and user_input.strip():
    query = user_input.strip()
    logger.info(f"[QUERY #{st.session_state.query_count + 1}] {query}")
    st.session_state.messages.append({
        "role":    "user",
        "content": query,
        "ts":      time.strftime("%H:%M"),
    })
    st.session_state.loading = True
    st.rerun()


# ── Generate Response ──────────────────────────────────────────────────────────
if (
    st.session_state.loading
    and st.session_state.messages
    and st.session_state.messages[-1]["role"] == "user"
):
    query      = st.session_state.messages[-1]["content"]
    session_id = st.session_state.session_id

    try:
        chain   = get_chain(st.session_state.model)
        t_start = time.time()

        # Get stream generator
        result = chain(query, session_id=session_id)
        
        # Create a placeholder and stream text into it
        placeholder = st.empty()
        full_response = ""
        
        for chunk in result["stream"]:
            full_response += chunk
            placeholder.markdown(render_live_message(full_response + " ▌", "Generating..."), unsafe_allow_html=True)
            
        elapsed = round(time.time() - t_start, 1)

        # links_holder is mutated by the generator above as a side effect of
        # fully consuming result["stream"] — must be read AFTER the loop, not before.
        resolved_links = result["links"]["links"]

        logger.info(
            f"[RESPONSE] idx={st.session_state.query_count} | "
            f"elapsed={elapsed}s | chunks={len(result['docs'])} | "
            f"links={len(resolved_links)} | "
            f"intent={result['intent']} | model={st.session_state.model}"
        )

        st.session_state.messages.append({
            "role":     "assistant",
            "content":  full_response,
            "docs":     result["docs"],
            "links":    resolved_links,
            "intent":   result["intent"],
            "feedback": None,
            "ts":       time.strftime("%H:%M"),
            "elapsed":  elapsed,
        })
        st.session_state.query_count += 1

    except Exception as e:
        logger.error(f"[ERROR] {e}")
        st.session_state.messages.append({
            "role":     "assistant",
            "content":  f"Something went wrong.\n\n**Error:** `{e}`\n\nCheck your `.env` and that ChromaDB is populated.",
            "docs":     [],
            "links":    [],
            "feedback": None,
            "ts":       time.strftime("%H:%M"),
        })

    finally:
        st.session_state.loading = False
        st.rerun()