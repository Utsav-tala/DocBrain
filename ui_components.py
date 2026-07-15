"""
ui_components.py — DocBrain UI layer (native-component edition)

Rebuilt around Streamlit's native chat primitives. The previous version hand-rolled
chat bubbles as raw HTML and interleaved them with real Streamlit widgets (st.code,
st.button) — but a raw <div> opened in one st.markdown() call cannot wrap a widget
rendered by another call, so every "container" (the centred column, the assistant
card) was visually broken and content spilled edge-to-edge.

Now app.py uses st.chat_message() (a real DOM container) + st.markdown()/st.write_stream()
(native markdown + fenced-code rendering, one renderer for streaming AND final). This
module only provides:
  - DOCBRAIN_CSS   : theme refinements on top of the dark base in .streamlit/config.toml
  - HEADER_HTML    : the top brand bar
  - render_hero()  : empty-state hero
  - render_source_links() : the "Learn more" verified-link pills (HTML is fine here —
                     it's leaf content rendered inside a chat_message container)
  - STARTER_QUERIES / SIDEBAR_LOGO / sidebar_section()
"""

# ── Theme ─────────────────────────────────────────────────────────────────────
DOCBRAIN_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --void:#0A0B0F; --surface:#0F1117; --card:#171923; --card-2:#1C1F2A;
  --border:#262A38; --border-soft:#1C1F2A;
  --indigo:#6366F1; --indigo-chip:rgba(99,102,241,.10); --indigo-glow:rgba(99,102,241,.25);
  --text-prime:#E6EAF3; --text-body:#AEB6C8; --text-muted:#6B7385;
  --font-ui:'Inter',system-ui,-apple-system,sans-serif; --font-mono:'JetBrains Mono',monospace;
}

/* ── Base ── */
.stApp, [data-testid="stAppViewContainer"] { background: var(--void) !important; }
html, body, [class*="css"] { font-family: var(--font-ui) !important; color: var(--text-prime); }
#MainMenu, header[data-testid="stHeader"], footer { display: none !important; }

/* Centre column — a comfortable reading width (the whole point) */
[data-testid="stMainBlockContainer"] { max-width: 820px !important; padding: 1.8rem 1rem 7rem !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border-soft) !important; }
[data-testid="stSidebar"] [data-testid="stMainBlockContainer"] { padding-top: 1.5rem !important; }
.sidebar-logo { display:flex; align-items:center; gap:10px; margin-bottom:1rem; padding-bottom:1rem; border-bottom:1px solid var(--border-soft); }
.sidebar-logo-dot { width:30px; height:30px; border-radius:8px; background:linear-gradient(135deg,#6366F1,#A855F7); display:flex; align-items:center; justify-content:center; font-size:15px; flex-shrink:0; }
.sidebar-logo-text { font-size:16px; font-weight:700; letter-spacing:-.02em; color:var(--text-prime); }
.sidebar-section-label { font-size:10px; font-weight:600; letter-spacing:.11em; text-transform:uppercase; color:var(--text-muted); margin:1.3rem 0 .5rem; }
.db-cfg { display:flex; flex-direction:column; gap:7px; }
.db-cfg > div { display:flex; justify-content:space-between; align-items:center; font-size:12.5px; color:var(--text-muted); }
.db-cfg b { color:var(--text-body); font-weight:600; }
.db-about { font-size:12px; color:var(--text-muted); line-height:1.6; }

/* ── Header bar ── */
.db-header { display:flex; align-items:center; gap:12px; padding:.1rem 0 1.1rem; border-bottom:1px solid var(--border-soft); margin-bottom:1.3rem; }
.db-logo { width:40px; height:40px; border-radius:11px; background:linear-gradient(135deg,#6366F1,#A855F7); display:flex; align-items:center; justify-content:center; font-size:21px; flex-shrink:0; }
.db-title { font-size:19px; font-weight:700; letter-spacing:-.02em; color:var(--text-prime); line-height:1.15; }
.db-sub { font-size:12.5px; color:var(--text-muted); margin-top:1px; }

/* ── Chat bubbles ── */
[data-testid="stChatMessage"] {
  background: var(--card) !important; border: 1px solid var(--border-soft) !important;
  border-radius: 16px !important; padding: .95rem 1.15rem !important; margin-bottom: .85rem !important;
  box-shadow: 0 1px 2px rgba(0,0,0,.18);
}
/* tint the user's turn — hooked via a hidden marker span rendered inside the user
   bubble, because custom emoji avatars don't carry a stChatMessageAvatarUser testid
   to :has() on, and Streamlit's per-role emotion classes are unstable to target. */
[data-testid="stChatMessage"]:has(.db-user-mark) {
  background: var(--indigo-chip) !important; border-color: rgba(99,102,241,.26) !important;
}
.db-user-mark { display: none; }
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"] { background: transparent !important; border: none !important; }

/* answer text */
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] { font-size:15.5px; line-height:1.75; color:var(--text-prime); }
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p { margin: 0 0 .7rem; }
[data-testid="stChatMessage"] h1, [data-testid="stChatMessage"] h2 { font-size:18px; font-weight:600; margin:1rem 0 .5rem; letter-spacing:-.01em; }
[data-testid="stChatMessage"] h3 { font-size:15px; font-weight:600; color:#A5B4FC; margin:.9rem 0 .35rem; }
[data-testid="stChatMessage"] li { margin-bottom:.3rem; line-height:1.7; }
[data-testid="stChatMessage"] code { font-family:var(--font-mono); font-size:13.5px; background:rgba(99,102,241,.12); border:1px solid rgba(99,102,241,.20); border-radius:5px; padding:1px 6px; color:#C7D2FE; }
[data-testid="stChatMessage"] a { color:#A5B4FC; }

/* code blocks (native st.code / fenced markdown) */
[data-testid="stChatMessage"] pre, [data-testid="stCode"] { background:#0C0E15 !important; border:1px solid var(--border) !important; border-radius:12px !important; }
[data-testid="stCode"] { margin:.35rem 0 .7rem; }

/* tables */
[data-testid="stChatMessage"] table { border-collapse:collapse; width:100%; font-size:14px; margin:.5rem 0 .8rem; }
[data-testid="stChatMessage"] th { background:var(--card-2); border:1px solid var(--border); padding:7px 11px; text-align:left; font-weight:600; }
[data-testid="stChatMessage"] td { border:1px solid var(--border-soft); padding:7px 11px; color:var(--text-body); }

/* meta caption + feedback */
[data-testid="stChatMessage"] [data-testid="stCaptionContainer"] { color:var(--text-muted) !important; font-size:11.5px !important; margin-top:.5rem; }

/* ── Source-link pills ── */
.src-wrap { margin-top:.85rem; padding-top:.7rem; border-top:1px solid var(--border-soft); }
.src-label { font-size:10px; font-weight:600; letter-spacing:.1em; text-transform:uppercase; color:var(--text-muted); margin-bottom:.5rem; }
.src-row { display:flex; flex-wrap:wrap; gap:8px; }
.src-btn { display:inline-flex; align-items:center; gap:6px; background:var(--card-2); border:1px solid var(--border); color:var(--text-body) !important; padding:6px 13px; border-radius:9px; font-size:12.5px; font-weight:500; text-decoration:none !important; transition:.15s ease; }
.src-btn:hover { border-color:rgba(99,102,241,.5); color:#C7D2FE !important; background:rgba(99,102,241,.08); transform:translateY(-1px); }

/* ── Buttons (starters, clear) ── */
.stButton > button { background:var(--card) !important; border:1px solid var(--border) !important; color:var(--text-body) !important; border-radius:10px !important; font-size:13.5px !important; font-weight:500 !important; transition:.15s ease !important; }
.stButton > button:hover { border-color:rgba(99,102,241,.5) !important; color:#C7D2FE !important; background:var(--card-2) !important; }

/* ── Chat input ── */
[data-testid="stChatInput"] { background:var(--card) !important; border:1px solid var(--border) !important; border-radius:14px !important; }
[data-testid="stChatInput"]:focus-within { border-color:rgba(99,102,241,.6) !important; box-shadow:0 0 0 3px var(--indigo-glow) !important; }
/* Streamlit's chat textarea ships pinned to its max height (~189px) even when empty
   and 1-row; force it to size to content so the empty state isn't a giant box. */
[data-testid="stChatInput"] textarea { color:var(--text-prime) !important; font-size:15px !important; height:auto !important; min-height:0 !important; max-height:180px !important; }
[data-testid="stChatInput"] textarea::placeholder { color:var(--text-muted) !important; }
[data-testid="stChatInput"] button { background:var(--indigo) !important; border:none !important; }
[data-testid="stBottomBlockContainer"], [data-testid="stBottom"] > div { background:var(--void) !important; }

/* ── Empty-state hero ── */
.db-hero { text-align:center; padding:2.2rem 1rem 1rem; }
.db-hero-icon { font-size:42px; margin-bottom:.4rem; }
.db-hero h2 { font-size:22px; font-weight:700; margin:0 0 .5rem; color:var(--text-prime); }
.db-hero p { font-size:14.5px; color:var(--text-muted); margin:0 auto; max-width:440px; line-height:1.6; }
.db-starter-label { font-size:10px; font-weight:600; letter-spacing:.11em; text-transform:uppercase; color:var(--text-muted); margin:1.3rem 0 .6rem; text-align:center; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width:8px; height:8px; }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:8px; }
::-webkit-scrollbar-track { background:transparent; }
</style>
"""

# ── Header ────────────────────────────────────────────────────────────────────
HEADER_HTML = """
<div class="db-header">
  <div class="db-logo">🧠</div>
  <div>
    <div class="db-title">DocBrain</div>
    <div class="db-sub">LangChain documentation assistant · agentic RAG</div>
  </div>
</div>
"""

# ── Empty-state hero ──────────────────────────────────────────────────────────
def render_hero() -> str:
    return """
<div class="db-hero">
  <div class="db-hero-icon">🧠</div>
  <h2>Ask anything about LangChain</h2>
  <p>Grounded answers with cited sources — across concepts, source code, error
  references, migration guides, and GitHub issues. Falls back to live web search
  when the local docs don't cover your question.</p>
</div>
"""

STARTER_QUERIES = [
    "What is LCEL in LangChain?",
    "How do I use Chroma as a vector store?",
    "Why am I getting OUTPUT_PARSING_FAILURE?",
    "LLMChain vs LCEL — what's the difference?",
]

# ── Source-link pills ─────────────────────────────────────────────────────────
# Each link is a SourceLink (label, url, relevance) from link_resolver.resolve_links()
# — already filtered for topical relevance and HEAD-verified, so everything shown
# here is both correct AND useful. Rendered as HTML because it's leaf content living
# inside a chat_message container (no widgets nested inside).
def render_source_links(links: list) -> str:
    if not links:
        return ""
    buttons = []
    for link in links:
        label = getattr(link, "label", None) or (link.get("label", "Documentation") if isinstance(link, dict) else "Documentation")
        url   = getattr(link, "url", None) or (link.get("url", "") if isinstance(link, dict) else "")
        if not url:
            continue
        buttons.append(f'<a class="src-btn" href="{url}" target="_blank" rel="noopener">↗ {label}</a>')
    if not buttons:
        return ""
    return (
        '<div class="src-wrap"><div class="src-label">Learn more</div>'
        f'<div class="src-row">{"".join(buttons)}</div></div>'
    )

# ── Sidebar snippets ──────────────────────────────────────────────────────────
SIDEBAR_LOGO = """<div class="sidebar-logo">
  <div class="sidebar-logo-dot">🧠</div>
  <div class="sidebar-logo-text">DocBrain</div>
</div>"""

def sidebar_section(label: str) -> str:
    return f'<div class="sidebar-section-label">{label}</div>'
