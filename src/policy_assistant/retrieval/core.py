"""
Retrieval from the FAISS vector store and parent docstore.

Loads the same embedding model as used at ingest time, retrieves relevant
chunks, and optionally expands to full parent content for LLM context.
"""

from __future__ import annotations

import os
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
    callbacks=None,
    project_name=None,
) -> list[Document]:
    """Return top-k documents from the vector store for the query."""
    # Only pass callbacks if they're provided and the method supports them
    # Custom _FaissIndexRetriever doesn't support callbacks
    kwargs = {}
    if callbacks is not None:
        # Check if similarity_search accepts callbacks by inspecting signature
        import inspect
        try:
            sig = inspect.signature(vector_store.similarity_search)
            if 'callbacks' in sig.parameters:
                kwargs['callbacks'] = callbacks
        except (AttributeError, ValueError):
            pass  # Method doesn't support signature inspection, skip callbacks
    
    # Manual tracing for custom retrievers that don't support callbacks
    # Use langsmith.trace() context manager for proper tracing
    if project_name and not kwargs.get('callbacks'):
        try:
            import langsmith as ls
            # Use trace() context manager - it handles creation and completion automatically
            with ls.trace(
                name="similarity_search",
                run_type="retriever",
                inputs={"query": query, "k": k},
                project_name=project_name,
            ) as trace_run:
                try:
                    if score_threshold is not None:
                        pairs = vector_store.similarity_search_with_score(query, k=k, **kwargs)
                        results = [doc for doc, score in pairs if score <= score_threshold]
                    else:
                        results = vector_store.similarity_search(query, k=k, **kwargs)
                    
                    # Format outputs for LangSmith
                    outputs = {
                        "documents": [
                            {
                                "page_content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                                "metadata": doc.metadata
                            }
                            for doc in results[:5]  # Show up to 5 docs
                        ],
                        "count": len(results)
                    }
                    trace_run.end(outputs=outputs)
                    
                    if os.environ.get("DEBUG_LANGSMITH") == "1":
                        print(f"[DEBUG] Traced similarity_search with {len(results)} documents")
                    
                    return results
                except Exception as e:
                    # End trace with error
                    trace_run.end(error=str(e))
                    if os.environ.get("DEBUG_LANGSMITH") == "1":
                        print(f"[DEBUG] Traced similarity_search with error: {e}")
                    raise
        except Exception as trace_error:
            # If tracing fails, still execute the retrieval
            if os.environ.get("DEBUG_LANGSMITH") == "1":
                print(f"[DEBUG] Tracing failed, continuing without trace: {trace_error}")
            # Fall through to non-traced execution
    
    # Non-traced execution (fallback or when callbacks are supported)
    if score_threshold is not None:
        pairs = vector_store.similarity_search_with_score(query, k=k, **kwargs)
        return [doc for doc, score in pairs if score <= score_threshold]
    return vector_store.similarity_search(query, k=k, **kwargs)


def get_context_for_llm(
    query: str,
    vector_store,
    parent_docstore: dict[str, Document],
    k: int = 5,
    use_parent_content: bool = True,
    max_parents: int = 5,
    callbacks=None,
    project_name=None,
) -> str:
    """
    Get formatted context for an LLM: either parent text (deduplicated, capped)
    or raw child text, with [Source: ...] headers.
    """
    chunks = get_relevant_chunks(query, vector_store, k=k, callbacks=callbacks, project_name=project_name)
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
