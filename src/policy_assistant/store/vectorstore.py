"""
Vector store utilities for the ingestion pipeline.

Supports building and persisting FAISS vector stores with selectable algorithms:
- flat: exact k-NN (LangChain FAISS default)
- hnsw: approximate k-NN via HNSW graph (same as retrieval_eval)
"""

import json
import pickle
from pathlib import Path

from langchain_core.documents import Document

try:
    from langchain_community.vectorstores import FAISS
except Exception:
    from langchain.vectorstores import FAISS  # type: ignore[import-not-found]

CONFIG_FILENAME = "vector_store_config.json"
FAISS_INDEX_FILENAME = "faiss_index.bin"
CHILD_DOCSTORE_FILENAME = "child_docstore.pkl"


def build_vector_store(
    child_docs: list[Document],
    out_dir: Path,
    embeddings,
    algorithm: str = "flat",
) -> None:
    """Build and persist vector store from child docs.

    algorithm: "flat" (default, exact k-NN) or "hnsw" (approximate k-NN).
    For "flat", uses LangChain FAISS and saves in LangChain format (backward compatible).
    For "hnsw", uses the same HNSW builder as retrieval_eval and saves a custom format.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if algorithm == "flat":
        vs = FAISS.from_documents(child_docs, embeddings)
        vs.save_local(str(out_dir))
        print(f"Saved FAISS vector store (flat) to: {out_dir}")
        return

    if algorithm == "hnsw":
        from policy_assistant.retrieval.algorithms import build_faiss_hnsw, get_algorithm_builders

        builders = get_algorithm_builders()
        _builder, kwargs = builders["hnsw"]
        retriever = build_faiss_hnsw(child_docs, embeddings, **kwargs)
        index = getattr(retriever, "_index")
        doc_list = getattr(retriever, "_doc_list", child_docs)

        import faiss
        index_path = out_dir / FAISS_INDEX_FILENAME
        faiss.write_index(index, str(index_path))
        with open(out_dir / CHILD_DOCSTORE_FILENAME, "wb") as f:
            pickle.dump(doc_list, f)
        config = {"algorithm": "hnsw"}
        with open(out_dir / CONFIG_FILENAME, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Saved FAISS vector store (hnsw) to: {out_dir}")
        return

    raise ValueError(
        f"Unsupported algorithm: {algorithm!r}. Use 'flat' or 'hnsw'."
    )


def load_vector_store(out_dir: Path, embeddings):
    """Load vector store from disk (same embeddings as at ingest time).

    If vector_store_config.json exists with algorithm \"hnsw\", loads the custom
    HNSW index and docstore. Otherwise uses LangChain FAISS.load_local (flat, backward compatible).
    """
    out_dir = Path(out_dir)
    config_path = out_dir / CONFIG_FILENAME

    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        algo = config.get("algorithm", "flat")
        if algo == "hnsw":
            import faiss
            from policy_assistant.retrieval.algorithms import _FaissIndexRetriever

            index = faiss.read_index(str(out_dir / FAISS_INDEX_FILENAME))
            with open(out_dir / CHILD_DOCSTORE_FILENAME, "rb") as f:
                doc_list = pickle.load(f)
            return _FaissIndexRetriever(index, doc_list, embeddings, metric_inner_product=True)

    return FAISS.load_local(
        str(out_dir), embeddings, allow_dangerous_deserialization=True
    )
