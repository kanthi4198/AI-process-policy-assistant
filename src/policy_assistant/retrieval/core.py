"""
Retrieval from the FAISS vector store and parent docstore.

Loads the same embedding model as used at ingest time, retrieves relevant
chunks, and optionally expands to full parent content for LLM context.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from policy_assistant.data.chunking import load_parent_docstore
from policy_assistant.embeddings.local import select_embeddings
from policy_assistant.store.vectorstore import load_vector_store


DEFAULT_VECTOR_STORE_DIR = Path("vector_store")
DEFAULT_HF_MODEL = "google/embeddinggemma-300m"


def load_retrieval_artifacts(vector_store_dir: Path | str, hf_model: str):
    """Load vector store, parent docstore, and embeddings (paths resolved from cwd)."""
    base = Path.cwd()
    out_dir = (base / str(vector_store_dir)).resolve()
    embeddings = select_embeddings(hf_model)
    vector_store = load_vector_store(out_dir, embeddings)
    parent_docstore = load_parent_docstore(out_dir)
    return vector_store, parent_docstore, embeddings


def get_relevant_chunks(
    query: str,
    vector_store,
    k: int = 5,
    score_threshold: float | None = None,
) -> list[Document]:
    """Return top-k documents from the vector store for the query."""
    if score_threshold is not None:
        pairs = vector_store.similarity_search_with_score(query, k=k)
        return [doc for doc, score in pairs if score <= score_threshold]
    return vector_store.similarity_search(query, k=k)


def get_context_for_llm(
    query: str,
    vector_store,
    parent_docstore: dict[str, Document],
    k: int = 5,
    use_parent_content: bool = True,
    max_parents: int = 5,
) -> str:
    """
    Get formatted context for an LLM: either parent text (deduplicated, capped)
    or raw child text, with [Source: ...] headers.
    """
    chunks = get_relevant_chunks(query, vector_store, k=k)
    if not chunks:
        return ""

    if use_parent_content:
        seen_parent_ids: set[str] = set()
        segments: list[str] = []
        for doc in chunks:
            parent_id = doc.metadata.get("parent_id")
            if not parent_id or parent_id in seen_parent_ids:
                continue
            if len(seen_parent_ids) >= max_parents:
                break
            parent = parent_docstore.get(parent_id)
            if parent is None:
                continue
            seen_parent_ids.add(parent_id)
            source = parent.metadata.get("source", "unknown")
            segments.append(f"[Source: {source}]\n{parent.page_content}")
        return "\n\n---\n\n".join(segments)

    segments = []
    for doc in chunks:
        source = doc.metadata.get("source", "unknown")
        segments.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(segments)
