"""
Vector store utilities for the ingestion pipeline.

Currently supports:
- Building and persisting a FAISS vector store from child documents.
"""

from pathlib import Path

from langchain.docstore.document import Document

try:
    from langchain_community.vectorstores import FAISS
except Exception:  # pragma: no cover - fallback for older LangChain layouts
    from langchain.vectorstores import FAISS  # type: ignore


def build_vector_store(child_docs: list[Document], out_dir: Path, embeddings) -> None:
    """Build and persist FAISS vector store from child docs."""
    vs = FAISS.from_documents(child_docs, embeddings)
    out_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(out_dir))
    print(f"Saved FAISS vector store to: {out_dir}")

