# ui/ui_app.py
# Minimal Streamlit chatbot: build index → ask questions

from __future__ import annotations

import os
import sys
import pathlib as _pl
from typing import List, Dict
import traceback
import streamlit as st

# Keep HF quiet on Windows; harmless if you see a warning
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# Make project root importable (so we can import engine.py)
_PROJECT_ROOT = _pl.Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from engine import RAGFAISS  # local module


def _list_pdfs(folder: str) -> List[str]:
    p = _pl.Path(folder)
    if not p.exists() or not p.is_dir():
        return []
    return [str(fp) for fp in p.glob("*.pdf")]


def _init_engine() -> RAGFAISS:
    # Match engine signature exactly
    return RAGFAISS(
        model_name="BAAI/bge-small-en-v1.5",
        chunk_size=450,
        chunk_overlap=80,
        alpha=0.6,
        threshold=0.35,
        embed_batch=8,   # reduce to 4/2 if RAM is tight
    )


st.set_page_config(page_title="PCOS RAG Chatbot", page_icon="💬", layout="centered")
st.title(" PCOS RAG Chatbot")

# Sidebar
with st.sidebar:
    st.header("Documents")
    default_dir = str((_PROJECT_ROOT / "data/pcos_docs").resolve())
    folder = st.text_input("PDF folder", value=default_dir)
    build = st.button("Build index", type="primary")
    clear = st.button("Clear index")

if "engine" not in st.session_state:
    st.session_state.engine = None
if "index_loaded" not in st.session_state:
    st.session_state.index_loaded = False
if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, str]] = []

if clear:
    st.session_state.engine = None
    st.session_state.index_loaded = False
    st.session_state.messages = []
    st.toast("Index cleared.", icon="🗑️")

if build:
    pdfs = _list_pdfs(folder)
    if not pdfs:
        st.error(f"No PDFs found in: {folder}")
    else:
        with st.spinner("Building index (first run may download the model)…"):
            try:
                st.session_state.engine = _init_engine()
                st.session_state.engine.ingest_pdfs(pdfs)
                st.session_state.index_loaded = True
                st.success(f"Indexed {len(pdfs)} PDF(s). You can start chatting below.")
            except Exception:
                st.error("Indexing failed.")
                st.code("".join(traceback.format_exc()))

st.divider()
st.subheader("Chat")

if not st.session_state.index_loaded:
    st.info("Click **Build index** in the sidebar to load your PDFs, then ask a question.")
    st.stop()

# Render chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask a question… (e.g., 'Give two PCOS-friendly breakfast ideas')")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    eng: RAGFAISS = st.session_state.engine
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                answer, _ = eng.answer(prompt, k=4)
                ctx_texts, scores, metas = eng.search(prompt, k=3)  # show top sources
            except Exception:
                st.error("Sorry, something went wrong.")
                st.code("".join(traceback.format_exc()))
                answer = "unanswered"
                ctx_texts, scores, metas = [], [], []

        if answer == "unanswered":
            st.warning("I’m not confident enough to answer from the current documents.")
            final_text = "unanswered"
        else:
            st.markdown(answer)
            final_text = answer

        if ctx_texts:
            with st.expander("Sources & supporting passages"):
                for txt, sc, md in zip(ctx_texts, scores, metas):
                    name = _pl.Path(md.get("source", "")).name or "(unknown)"
                    page = md.get("page")
                    title = f"• {name}" + (f" (p.{page})" if page is not None else "")
                    st.caption(f"{title} — score={sc:.3f}")
                    st.write(txt)

    st.session_state.messages.append({"role": "assistant", "content": final_text})
