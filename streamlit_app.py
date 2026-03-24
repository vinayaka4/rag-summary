"""
Web UI: upload PDF, Word (.docx), or Excel (.xlsx), then ask questions.

Local:
  streamlit run streamlit_app.py

Render: set OPENAI_API_KEY in Environment; use start command from render.yaml.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag_core import (
    answer_question,
    build_vectorstore,
    default_embeddings,
    default_llm,
    load_documents,
    require_openai_key,
)

load_dotenv()

st.set_page_config(page_title="Document Q&A", page_icon="📄", layout="centered")
st.title("Document Q&A")
st.caption("Upload a PDF, Word (.docx), or Excel file, then ask questions about it.")

try:
    require_openai_key()
except RuntimeError as e:
    st.error(str(e))
    st.stop()

if "db" not in st.session_state:
    st.session_state.db = None
if "filename" not in st.session_state:
    st.session_state.filename = None
if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded = st.file_uploader(
    "Upload document",
    type=["pdf", "docx", "xlsx"],
    help="Use .docx (not legacy .doc). Excel: .xlsx only.",
)

if uploaded is not None:
    suffix = Path(uploaded.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = tmp.name
    try:
        with st.spinner("Loading and indexing your document…"):
            docs = load_documents(tmp_path)
            embeddings = default_embeddings()
            st.session_state.db = build_vectorstore(docs, embeddings)
            st.session_state.filename = uploaded.name
            st.session_state.messages = []
        st.success(f"Ready: **{uploaded.name}** ({len(docs)} loaded segment(s)). Ask below.")
    except Exception as e:
        st.session_state.db = None
        st.error(f"Could not process file: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

if st.session_state.db is None:
    st.info("Upload a document to begin.")
else:
    st.markdown(f"**Current file:** `{st.session_state.filename}`")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about the document"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    llm = default_llm()
                    answer = answer_question(st.session_state.db, llm, prompt)
                except Exception as e:
                    answer = f"Error: {e}"
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
