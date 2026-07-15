"""
bootstrap_db.py — make the Chroma vector DB present before the app queries it.

Why this exists
---------------
The prebuilt index (db/chroma_langchain/, ~200 MB) is intentionally NOT committed
to git: chroma.sqlite3 alone is ~123 MB, over GitHub's 100 MB per-file hard limit,
and rebuilding it via `python -m src.ingest` needs the ~1.1 GB raw corpus plus a
full re-embed (OpenAI cost + minutes of work). Neither is viable on a free cloud
build.

So for cloud deploys we ship the index as a single zipped GitHub Release asset and
lazy-download it on first run. Locally the folder already exists, so ensure_db()
is a cheap no-op (one Path.exists() check).

The archive is expected to contain `chroma_langchain/...` at its root, so it
extracts cleanly into db/  →  db/chroma_langchain/...
"""

import os
import tempfile
import zipfile
from pathlib import Path

import requests
from loguru import logger

# Local layout — must match DB_PATH / COLLECTION_NAME in src/retriever.py
DB_DIR      = Path("db/chroma_langchain")
DB_SENTINEL = DB_DIR / "chroma.sqlite3"     # presence => the DB is materialized

# GitHub Release asset holding the zipped index. Override via the DOCBRAIN_DB_URL
# env var / Streamlit secret whenever you cut a new release tag, so the code
# doesn't need editing to point at a fresh index.
DEFAULT_DB_URL = (
    "https://github.com/Utsav-tala/DocBrain/releases/download/db-v5/chroma_langchain.zip"
)
DB_URL = os.getenv("DOCBRAIN_DB_URL", DEFAULT_DB_URL)

_DOWNLOAD_TIMEOUT = 180  # seconds — generous for a ~130 MB asset on a cold container


def ensure_db() -> None:
    """Download + extract the vector DB if it isn't already on disk. Idempotent."""
    if DB_SENTINEL.exists():
        logger.info(f"[bootstrap] DB present at {DB_DIR} — nothing to do")
        return

    logger.info(f"[bootstrap] DB missing — fetching from {DB_URL}")
    DB_DIR.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = Path(tempfile.mkdtemp()) / "chroma_langchain.zip"
    bytes_read = 0
    with requests.get(DB_URL, stream=True, timeout=_DOWNLOAD_TIMEOUT) as resp:
        resp.raise_for_status()
        with open(tmp_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                fh.write(chunk)
                bytes_read += len(chunk)
    logger.info(f"[bootstrap] downloaded {bytes_read / 1e6:.1f} MB → {tmp_path}")

    with zipfile.ZipFile(tmp_path) as zf:
        zf.extractall(DB_DIR.parent)   # archive root is chroma_langchain/ → db/chroma_langchain/
    tmp_path.unlink(missing_ok=True)

    if not DB_SENTINEL.exists():
        raise RuntimeError(
            f"[bootstrap] extracted the archive but {DB_SENTINEL} is still missing. "
            "Check the release asset's layout — its root folder must be "
            "'chroma_langchain/' so it lands at db/chroma_langchain/."
        )
    logger.info(f"[bootstrap] DB ready at {DB_DIR}")
