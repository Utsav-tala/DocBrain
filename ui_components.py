"""
ui_components.py — DocBrain UI Layer v2
Fixes:
  - User messages RIGHT aligned, DB messages LEFT aligned in proper chat layout
  - Code blocks: use st.code() native Streamlit renderer (no custom HTML highlighter)
    The custom regex highlighter was double-encoding HTML → raw span tags visible
  - Increased font sizes: body 16px, headings 18px/15px
  - Cleaner overall layout, removed noisy stats bar
"""

import streamlit as st
import re
import html as html_lib
import markdown

# ── CSS Theme ─────────────────────────────────────────────────────────────────

DOCBRAIN_CSS = """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root Tokens ── */
:root {
    --void:        #0A0B0F;
    --surface:     #111318;
    --card:        #1C1E26;
    --card-hover:  #21242E;
    --border:      #2A2D3A;
    --border-soft: #1F2130;
    --indigo:      #6366F1;
    --indigo-dim:  #4F52C8;
    --indigo-glow: rgba(99,102,241,0.18);
    --indigo-chip: rgba(99,102,241,0.12);
    --text-prime:  #E2E8F0;
    --text-body:   #94A3B8;
    --text-muted:  #64748B;
    --text-dim:    #3D4560;
    --green:       #34D399;
    --red:         #F87171;
    --amber:       #FBBF24;
    --font-ui:     'Inter', system-ui, -apple-system, sans-serif;
    --font-mono:   'JetBrains Mono', 'Fira Code', monospace;
    --radius-sm:   6px;
    --radius-md:   10px;
    --radius-lg:   16px;
    --radius-xl:   20px;
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: var(--font-ui) !important;
    background-color: var(--void) !important;
    color: var(--text-prime) !important;
}

/* ── Strip Streamlit Chrome ── */
#MainMenu, footer, header { visibility: hidden; height: 0; }
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}
section[data-testid="stSidebar"] > div:first-child {
    background: var(--surface) !important;
    border-right: 1px solid var(--border-soft) !important;
    padding: 1.5rem 1.25rem !important;
}
.stApp { background: var(--void) !important; }

/* ── Main Content ── */
.main-content {
    max-width: 900px;
    margin: 0 auto;
    padding: 2rem 1.5rem 7rem 4rem !important;
    display: flex;
    flex-direction: column;
    gap: 0;
}

/* ── Sidebar ── */
.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border-soft);
}
.sidebar-logo-dot {
    width: 32px; height: 32px;
    background: var(--indigo);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; font-weight: 700; color: white;
    flex-shrink: 0;
}
.sidebar-logo-text {
    font-size: 17px;
    font-weight: 600;
    color: var(--text-prime);
    letter-spacing: -0.02em;
}
.sidebar-section-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 1.5rem 0 0.6rem 0;
}

/* ── Streamlit Widget Overrides ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stSlider"] > div,
[data-testid="stNumberInput"] > div > div {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-prime) !important;
}
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: var(--indigo) !important;
    box-shadow: 0 0 0 3px var(--indigo-glow) !important;
}
label, .stSelectbox label, .stSlider label {
    color: var(--text-body) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}

/* ── Page Header ── */
.page-header {
    padding: 2rem 0 1.25rem 0;
    border-bottom: 1px solid var(--border-soft);
    margin-bottom: 2rem;
}
.page-header h1 {
    font-size: 22px;
    font-weight: 600;
    color: var(--text-prime);
    letter-spacing: -0.03em;
    margin: 0 0 4px 0;
}
.page-header p {
    font-size: 14px;
    color: var(--text-muted);
    margin: 0;
}

/* ═══════════════════════════════════════════════
   CHAT LAYOUT — user RIGHT, assistant LEFT
   Each turn is a full-width row with flex layout.
═══════════════════════════════════════════════ */

/* Turn container — full width row */
.chat-turn {
    display: flex;
    flex-direction: column;
    gap: 0;
    margin-bottom: 1.5rem;
}

/* ── USER MESSAGE (right side) ── */
.msg-user-row {
    display: flex;
    justify-content: flex-end;   /* push to RIGHT */
    margin-bottom: 0.5rem;
}
.msg-user-bubble {
    max-width: 68%;
    background: var(--indigo-chip);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: var(--radius-lg) var(--radius-sm) var(--radius-lg) var(--radius-lg);
    padding: 0.9rem 1.2rem;
    font-size: 16px;
    line-height: 1.6;
    color: var(--text-prime);
    word-wrap: break-word;
}

/* ── ASSISTANT MESSAGE (left side) ── */
.msg-assistant-row {
    display: flex;
    justify-content: flex-start;  /* push to LEFT */
    align-items: flex-start;
    gap: 12px;
}
.msg-avatar {
    width: 32px; height: 32px;
    background: var(--indigo);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: 700; color: white;
    flex-shrink: 0;
    margin-top: 4px;
}
.msg-assistant-body {
    flex: 1;
    min-width: 0;
    background: var(--card);
    border: 1px solid var(--border-soft);
    border-radius: var(--radius-sm) var(--radius-lg) var(--radius-lg) var(--radius-lg);
    padding: 1.1rem 1.4rem;
}
.msg-meta {
    font-size: 11px;
    color: var(--text-dim);
    margin-bottom: 8px;
    font-weight: 500;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* ── Assistant answer text ── */
.msg-text {
    font-size: 16px;
    line-height: 1.8;
    color: var(--text-prime);
}
.msg-text h2 {
    font-size: 18px;
    font-weight: 600;
    color: var(--text-prime);
    margin: 1.3rem 0 0.5rem 0;
    letter-spacing: -0.02em;
    padding-bottom: 4px;
    border-bottom: 1px solid var(--border-soft);
}
.msg-text h3 {
    font-size: 15px;
    font-weight: 600;
    color: #A5B4FC;
    margin: 1.1rem 0 0.4rem 0;
}
.msg-text p { margin: 0 0 0.9rem 0; }
.msg-text ul, .msg-text ol {
    margin: 0.5rem 0 0.9rem 1.5rem;
    padding: 0;
}
.msg-text li { margin-bottom: 0.35rem; line-height: 1.7; font-size: 16px; }
.msg-text strong { color: var(--text-prime); font-weight: 600; }
.msg-text em { color: var(--text-body); font-style: italic; }
.msg-text code {
    font-family: var(--font-mono);
    font-size: 14px;
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 4px;
    padding: 2px 7px;
    color: #A5B4FC;
}
/* Table styles */
.msg-text table {
    width: 100%;
    border-collapse: collapse;
    margin: 0.75rem 0 1rem 0;
    font-size: 14.5px;
}
.msg-text th {
    background: var(--card-hover);
    color: var(--text-prime);
    font-weight: 600;
    padding: 8px 12px;
    text-align: left;
    border: 1px solid var(--border);
}
.msg-text td {
    padding: 8px 12px;
    border: 1px solid var(--border-soft);
    color: var(--text-body);
    vertical-align: top;
}
.msg-text tr:nth-child(even) td { background: rgba(255,255,255,0.02); }

/* ── Divider between turns ── */
.turn-divider {
    height: 1px;
    background: var(--border-soft);
    margin: 1.5rem 0;
    opacity: 0.4;
}

/* ── Source Link Buttons (replaces old pill-chip Sources row) ── */
.source-links-section {
    margin-top: 1.25rem;
    padding-top: 0.85rem;
    border-top: 1px solid var(--border-soft);
}
.source-links-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 8px;
}
.source-links-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}
.source-link-btn {
    display: inline-flex;
    align-items: center;
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
    color: white !important;
    padding: 7px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    text-decoration: none;
    transition: all 0.2s ease;
    box-shadow: 0 4px 12px rgba(168, 85, 247, 0.25);
    white-space: nowrap;
}
.source-link-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(168, 85, 247, 0.45);
}

/* ── Feedback Row ── */
.feedback-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 0.85rem;
    padding-top: 0.75rem;
    border-top: 1px solid var(--border-soft);
}
.feedback-label { font-size: 12px; color: var(--text-dim); }

/* ── Streamlit chat_input ── */
[data-testid="stChatInput"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-xl) !important;
    transition: box-shadow 0.25s ease, border-color 0.25s ease !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(99,102,241,0.5) !important;
    box-shadow: 0 0 0 4px var(--indigo-glow) !important;
}
[data-testid="stChatInput"] textarea {
    font-family: var(--font-ui) !important;
    font-size: 15px !important;
    color: var(--text-prime) !important;
    background: transparent !important;
    caret-color: var(--indigo) !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: var(--text-dim) !important; }
[data-testid="stChatInput"] button {
    background: var(--indigo) !important;
    border-radius: 10px !important;
    border: none !important;
}
[data-testid="stChatInput"] button:hover { background: var(--indigo-dim) !important; }

/* ── Thought Stream ── */
.thought-stream {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0.7rem 0 0.3rem 44px;
}
.ts-dots { display: flex; gap: 5px; align-items: center; }
.ts-dot {
    width: 7px; height: 7px;
    background: var(--indigo);
    border-radius: 50%;
    opacity: 0.3;
    animation: tsPulse 1.2s ease-in-out infinite;
}
.ts-dot:nth-child(2) { animation-delay: 0.2s; }
.ts-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes tsPulse {
    0%, 100% { opacity: 0.2; transform: scale(0.85); }
    50%       { opacity: 1;   transform: scale(1.1); }
}
.ts-label { font-size: 13px; color: var(--text-dim); font-style: italic; }

/* ── Empty State ── */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 5rem 2rem;
    text-align: center;
    gap: 1rem;
}
.empty-icon {
    width: 56px; height: 56px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    display: flex; align-items: center; justify-content: center;
    font-size: 24px;
    margin-bottom: 0.25rem;
}
.empty-state h2 { font-size: 20px; font-weight: 600; color: var(--text-prime); margin: 0; }
.empty-state p { font-size: 15px; color: var(--text-muted); margin: 0; max-width: 340px; line-height: 1.65; }
.starter-chips { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 0.75rem; }
.starter-chip {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 7px 16px;
    font-size: 13.5px;
    color: var(--text-body);
    cursor: pointer;
    transition: all 0.15s ease;
    white-space: nowrap;
}
.starter-chip:hover {
    border-color: rgba(99,102,241,0.4);
    color: #A5B4FC;
    background: var(--indigo-chip);
}

/* ── Source Button ── */
.source-btn {
    display: inline-flex;
    align-items: center;
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
    color: white !important;
    padding: 8px 16px;
    border-radius: 20px;
    font-size: 14px;
    font-weight: 600;
    text-decoration: none;
    transition: all 0.2s ease;
    box-shadow: 0 4px 15px rgba(168, 85, 247, 0.3);
    margin-top: 15px;
}
.source-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(168, 85, 247, 0.5);
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 10px; }

/* ── Mobile ── */
@media (max-width: 640px) {
    .main-content { padding: 1.5rem 1rem 6rem 1rem; }
    .msg-user-bubble { max-width: 90%; font-size: 15px; }
    .msg-text { font-size: 15px; }
}
</style>
"""

# ── Priority / Icon Maps ────────────────────────────────────────────────────────
PRIORITY_CLASS = {1: "p1", 2: "p2", 3: "p3"}
PRIORITY_LABEL = {1: "P1", 2: "P2", 3: "P3"}

DOC_TYPE_ICON = {
    "core_guide":      "📘",
    "core_concepts":   "🧠",
    "error_reference": "🔴",
    "langgraph_guide": "🕸",
    "migration_guide": "🔄",
    "source_code":     "⚙",
    "integration":     "🔌",
    "troubleshooting": "🐛",
    "deepagents":      "🤖",
}


# ── Markdown → Clean HTML ──────────────────────────────────────────────────────
# IMPORTANT: Code blocks are NOT rendered here.
# They are extracted, their positions stored as placeholders,
# and then rendered via st.code() in app.py AFTER this HTML is displayed.
# This avoids the double-encoding bug where span tags appeared as raw text.

def split_answer_into_parts(md: str) -> list:
    """
    Split the LLM markdown answer into a list of parts:
      {"type": "html",   "content": "<rendered markdown html>"}
      {"type": "code",   "content": "raw code", "lang": "python"}

    This allows app.py to render text as HTML and code blocks via
    native st.code() — which is syntax-highlighted correctly by Streamlit.
    """
    parts = []
    # Split on fenced code blocks
    segments = re.split(r'```(\w*)\n([\s\S]*?)```', md)
    # segments alternates: text, lang, code, text, lang, code, ...

    i = 0
    while i < len(segments):
        if i % 3 == 0:
            # Text segment — convert markdown to HTML
            text = segments[i].strip()
            if text:
                parts.append({"type": "html", "content": _md_to_html(text)})
        elif i % 3 == 1:
            # Language tag (e.g. "python", "bash")
            lang = segments[i].strip().lower() or "python"
            code = segments[i + 1].strip() if (i + 1) < len(segments) else ""
            if code:
                parts.append({"type": "code", "content": code, "lang": lang})
            i += 1  # skip the code body (consumed here)
        i += 1

    return parts

def render_live_message(text: str, ts: str) -> str:
    """Render a single HTML string for live streaming (avoids st.code flickering)."""
    html_content = markdown.markdown(text, extensions=['fenced_code', 'tables'])

    return f'''
    <div class="msg-assistant-row">
      <div class="msg-avatar">DB</div>
      <div class="msg-assistant-body">
        <div class="msg-meta">DocBrain · {ts}</div>
        <div class="msg-text">
          {html_content}
        </div>
      </div>
    </div>
    '''

def _md_to_html(text: str) -> str:
    """Convert basic markdown to HTML. No code blocks — those are handled separately."""
    # Headings
    text = re.sub(r'^## (.+)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^### (.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    # Bold
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    # Italic
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    # Inline code — escape first to avoid HTML injection
    def inline_code(m):
        escaped = html_lib.escape(m.group(1))
        return f'<code>{escaped}</code>'
    text = re.sub(r'`([^`]+)`', inline_code, text)
    # Tables (simple pipe tables)
    text = _render_table(text)
    # Bullet lists
    text = re.sub(r'^[-*] (.+)$', r'<li>\1</li>', text, flags=re.MULTILINE)
    text = re.sub(r'((?:<li>.*?</li>\s*)+)', r'<ul>\1</ul>', text, flags=re.DOTALL)
    # Numbered lists
    text = re.sub(r'^\d+\. (.+)$', r'<li>\1</li>', text, flags=re.MULTILINE)
    # Paragraphs
    parts = text.split('\n\n')
    out = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.startswith('<h') or part.startswith('<ul') or part.startswith('<ol') or part.startswith('<table'):
            out.append(part)
        else:
            # Replace single newlines with space inside paragraphs
            cleaned = part.replace('\n', ' ')
            out.append(f'<p>{cleaned}</p>')
    return '\n'.join(out)


def _render_table(text: str) -> str:
    """Convert pipe-table markdown to HTML table."""
    lines = text.split('\n')
    out = []
    i = 0
    while i < len(lines):
        if '|' in lines[i] and i + 1 < len(lines) and re.match(r'^\|[-| :]+\|', lines[i + 1]):
            # Table start
            header_cells = [c.strip() for c in lines[i].split('|') if c.strip()]
            i += 2  # skip separator line
            rows = []
            while i < len(lines) and '|' in lines[i]:
                cells = [c.strip() for c in lines[i].split('|') if c.strip()]
                rows.append(cells)
                i += 1
            th = ''.join(f'<th>{html_lib.escape(c)}</th>' for c in header_cells)
            trs = ''.join(
                '<tr>' + ''.join(f'<td>{html_lib.escape(c)}</td>' for c in row) + '</tr>'
                for row in rows
            )
            out.append(f'<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>')
        else:
            out.append(lines[i])
            i += 1
    return '\n'.join(out)


# ── Source Link Buttons ─────────────────────────────────────────────────────
# Replaces the old pill-chip "Sources" row. Each link is a SourceLink
# (label, url, relevance) from link_resolver.resolve_links() — already
# filtered for topical relevance and verified to actually resolve, so
# everything shown here is both correct AND useful to the user.

def render_source_links(links: list) -> str:
    if not links:
        return ""

    buttons = []
    for link in links:
        label = getattr(link, "label", None) or link.get("label", "Documentation")
        url   = getattr(link, "url", None) or link.get("url", "")
        if not url:
            continue
        buttons.append(
            f'<a class="source-link-btn" href="{url}" target="_blank">{label}</a>'
        )

    if not buttons:
        return ""

    return f"""<div class="source-links-section">
  <div class="source-links-label">For more understanding</div>
  <div class="source-links-row">
    {''.join(buttons)}
  </div>
</div>"""


# ── Thought Stream ─────────────────────────────────────────────────────────────
def render_thought_stream() -> str:
    return """<div class="thought-stream">
  <div class="ts-dots">
    <div class="ts-dot"></div>
    <div class="ts-dot"></div>
    <div class="ts-dot"></div>
  </div>
  <span class="ts-label">DocBrain is thinking…</span>
</div>"""


# ── Empty State ────────────────────────────────────────────────────────────────
STARTER_QUERIES = [
    "What is LCEL in LangChain?",
    "How to use Chroma as a vector store?",
    "Why am I getting OUTPUT_PARSING_FAILURE?",
    "What is LangGraph and when should I use it?",
    "Difference between LLMChain and LCEL?",
]

def render_empty_state() -> str:
    chips = "".join(
        f'<div class="starter-chip">{q}</div>'
        for q in STARTER_QUERIES
    )
    return f"""<div class="empty-state">
  <div class="empty-icon">🧠</div>
  <h2>Ask anything about LangChain</h2>
  <p>Searches across concepts, source code, error references, and migration guides.</p>
  <div class="starter-chips">{chips}</div>
</div>"""


# ── Sidebar Snippets ──────────────────────────────────────────────────────────
SIDEBAR_LOGO = """<div class="sidebar-logo">
  <div class="sidebar-logo-dot">DB</div>
  <div class="sidebar-logo-text">DocBrain</div>
</div>"""

def sidebar_section(label: str) -> str:
    return f'<div class="sidebar-section-label">{label}</div>'


# ── Stats Bar (simplified) ────────────────────────────────────────────────────
def render_stats_bar(chunk_count: int = 13093, model: str = "gpt-4o-mini",
                     query_count: int = 0) -> str:
    return ""   # removed — too noisy; model shown in sidebar