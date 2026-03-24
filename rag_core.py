"""Shared RAG helpers: load PDF / Word / Excel, chunk, embed, answer."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import pandas as pd
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 30
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_MAX_TOKENS = 200
CONTEXT_CHAR_CAP = 4000
DEFAULT_K = 3


def load_documents(file_path: str) -> List[Document]:
    """Load one file into LangChain Documents. Supports .pdf, .docx, .xlsx."""
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        return PyPDFLoader(str(path)).load()

    if ext == ".docx":
        return Docx2txtLoader(str(path)).load()

    if ext == ".xlsx":
        return _excel_to_documents(str(path))

    raise ValueError(
        f"Unsupported file type '{ext}'. Use PDF (.pdf), Word (.docx), or Excel (.xlsx)."
    )


def _excel_to_documents(path: str) -> List[Document]:
    """Flatten each sheet to text rows for embedding."""
    docs: List[Document] = []
    xl = pd.ExcelFile(path)
    for sheet in xl.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet, header=None)
        df = df.fillna("")
        lines = []
        for _, row in df.iterrows():
            cells = [str(c).strip() for c in row if str(c).strip()]
            if cells:
                lines.append(" | ".join(cells))
        text = f"Sheet: {sheet}\n" + "\n".join(lines)
        if text.strip():
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": path, "sheet": sheet},
                )
            )
    if not docs:
        docs.append(Document(page_content="(empty spreadsheet)", metadata={"source": path}))
    return docs


def make_text_splitter(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def build_vectorstore(documents: List[Document], embeddings: OpenAIEmbeddings) -> Chroma:
    splitter = make_text_splitter()
    chunks = splitter.split_documents(documents)
    return Chroma.from_documents(chunks, embeddings)


def answer_question(
    db: Chroma,
    llm: ChatOpenAI,
    query: str,
    k: int = DEFAULT_K,
) -> str:
    docs = db.similarity_search(query, k=k)
    context = "\n".join(d.page_content for d in docs)
    context = context[:CONTEXT_CHAR_CAP]
    prompt = f"""Answer based only on the context below. If the answer is not in the context, say you don't know from the document.

{context}

Question:
{query}
"""
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


def default_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=DEFAULT_EMBEDDING_MODEL)


def default_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=DEFAULT_CHAT_MODEL,
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=0,
    )


def require_openai_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Add it to your .env file in the project folder."
        )
