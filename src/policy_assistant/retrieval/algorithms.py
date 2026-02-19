"""
Retrieval algorithm backends for evaluation and production.

Provides a common interface (similarity_search(query, k) -> list[Document])
for multiple algorithms: FAISS Flat, HNSW, IVF, LSH, and Hybrid (dense + BM25).
Uses faiss for HNSW/IVF/LSH; LangChain FAISS for flat; rank_bm25 for sparse.
"""

from __future__ import annotations

import re
from typing import Protocol

from langchain_core.documents import Document

try:
    from langchain_community.vectorstores import FAISS
except Exception:
    from langchain.vectorstores import FAISS  # type: ignore


class RetrieverLike(Protocol):
    """Protocol for retrieval: same interface as LangChain VectorStore.similarity_search."""

    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        ...


# ---------------------------------------------------------------------------
# FAISS-based retrievers (raw faiss index + doc list)
# ---------------------------------------------------------------------------


def _embed_docs(embeddings, documents: list[Document]) -> list[list[float]]:
    """Embed documents; returns list of vectors (normalized if embeddings use normalize_embeddings)."""
    texts = [d.page_content for d in documents]
    return embeddings.embed_documents(texts)


def _embed_query(embeddings, query: str) -> list[float]:
    return embeddings.embed_query(query)


class _FaissIndexRetriever:
    """Thin wrapper around a faiss index + doc list for similarity_search."""

    def __init__(self, index, doc_list: list[Document], embeddings, metric_inner_product: bool = True):
        import numpy as np
        self._index = index
        self._doc_list = doc_list
        self._embeddings = embeddings
        self._metric_ip = metric_inner_product
        self._np = np

    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        np = self._np
        q = _embed_query(self._embeddings, query)
        q_vec = np.array([q], dtype="float32")
        k = min(k, len(self._doc_list))
        if k <= 0:
            return []
        distances, indices = self._index.search(q_vec, k)
        out: list[Document] = []
        for idx in indices[0]:
            if 0 <= idx < len(self._doc_list):
                out.append(self._doc_list[idx])
        return out

    def similarity_search_with_score(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        """Return (document, score) pairs. For inner product, higher score = more similar."""
        np = self._np
        q = _embed_query(self._embeddings, query)
        q_vec = np.array([q], dtype="float32")
        k = min(k, len(self._doc_list))
        if k <= 0:
            return []
        distances, indices = self._index.search(q_vec, k)
        out: list[tuple[Document, float]] = []
        for j, idx in enumerate(indices[0]):
            if 0 <= idx < len(self._doc_list):
                # FAISS returns distances; for METRIC_INNER_PRODUCT it is inner product (higher = better)
                score = float(distances[0][j])
                out.append((self._doc_list[idx], score))
        return out


def _ensure_numpy_float32(vectors: list[list[float]]):
    import numpy as np
    return np.array(vectors, dtype="float32")


def build_faiss_flat(child_docs: list[Document], embeddings) -> RetrieverLike:
    """Baseline: LangChain FAISS with default flat index (exact k-NN)."""
    return FAISS.from_documents(child_docs, embeddings)


def build_faiss_hnsw(
    child_docs: list[Document],
    embeddings,
    M: int = 32,
    ef_construction: int = 200,
    ef_search: int = 100,
) -> RetrieverLike:
    """HNSW graph index (approximate k-NN). Uses inner product for normalized embeddings."""
    import faiss
    np = __import__("numpy")
    vectors = _ensure_numpy_float32(_embed_docs(embeddings, child_docs))
    d = vectors.shape[1]
    index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    index.add(vectors)
    return _FaissIndexRetriever(index, child_docs, embeddings, metric_inner_product=True)


def build_faiss_ivf(
    child_docs: list[Document],
    embeddings,
    nlist: int = 100,
    nprobe: int = 10,
) -> RetrieverLike:
    """IVF (inverted file) index. Requires training on the same vectors."""
    import faiss
    np = __import__("numpy")
    vectors = _ensure_numpy_float32(_embed_docs(embeddings, child_docs))
    d = vectors.shape[1]
    nlist = min(nlist, len(child_docs) // 10, 1000)
    nlist = max(1, nlist)
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(vectors)
    index.add(vectors)
    index.nprobe = min(nprobe, nlist)
    return _FaissIndexRetriever(index, child_docs, embeddings, metric_inner_product=True)


def build_faiss_lsh(
    child_docs: list[Document],
    embeddings,
    nbits: int | None = None,
) -> RetrieverLike:
    """LSH (locality-sensitive hashing) index. nbits default 2*d."""
    import faiss
    np = __import__("numpy")
    vectors = _ensure_numpy_float32(_embed_docs(embeddings, child_docs))
    d = vectors.shape[1]
    if nbits is None:
        nbits = min(2 * d, 1024)
    index = faiss.IndexLSH(d, nbits)
    index.train(vectors)
    index.add(vectors)
    return _FaissIndexRetriever(index, child_docs, embeddings, metric_inner_product=False)


def _simple_tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, drop empty."""
    text = (text or "").lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return tokens


def build_hybrid(
    child_docs: list[Document],
    embeddings,
    dense_algorithm: str = "flat",
    rrf_k: int = 60,
    dense_top_n: int = 50,
    bm25_top_n: int = 50,
    **dense_kwargs,
) -> RetrieverLike:
    """Hybrid: dense (FAISS) + BM25, merged with Reciprocal Rank Fusion (RRF)."""
    from rank_bm25 import BM25Okapi

    # Dense retriever
    if dense_algorithm == "flat":
        dense_retriever = build_faiss_flat(child_docs, embeddings)
    elif dense_algorithm == "hnsw":
        dense_retriever = build_faiss_hnsw(child_docs, embeddings, **dense_kwargs)
    elif dense_algorithm == "ivf":
        dense_retriever = build_faiss_ivf(child_docs, embeddings, **dense_kwargs)
    elif dense_algorithm == "lsh":
        dense_retriever = build_faiss_lsh(child_docs, embeddings, **dense_kwargs)
    else:
        raise ValueError(f"Unknown dense_algorithm: {dense_algorithm}")

    # BM25 on document texts
    tokenized_corpus = [_simple_tokenize(d.page_content) for d in child_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    doc_list = child_docs

    class _HybridRetriever:
        def similarity_search(self, query: str, k: int = 5) -> list[Document]:
            # Dense top-N
            dense_docs = dense_retriever.similarity_search(query, k=dense_top_n)
            dense_rank = {id(d): r for r, d in enumerate(dense_docs, start=1)}

            # BM25 top-N
            tokenized_query = _simple_tokenize(query)
            scores = bm25.get_scores(tokenized_query)
            bm25_indices = scores.argsort()[::-1][:bm25_top_n]
            bm25_rank = {id(doc_list[i]): r for r, i in enumerate(bm25_indices, start=1)}

            # Collect candidate doc ids (Documents are not hashable; use id())
            candidate_ids: set[int] = set()
            id_to_doc: dict[int, Document] = {}
            for d in dense_docs:
                candidate_ids.add(id(d))
                id_to_doc[id(d)] = d
            for i in bm25_indices:
                d = doc_list[i]
                candidate_ids.add(id(d))
                id_to_doc[id(d)] = d

            # RRF: score = sum 1/(rrf_k + rank)
            rrf_scores: dict[int, float] = {}
            for doc_id in candidate_ids:
                rrf_scores[doc_id] = 0.0
                if doc_id in dense_rank:
                    rrf_scores[doc_id] += 1.0 / (rrf_k + dense_rank[doc_id])
                if doc_id in bm25_rank:
                    rrf_scores[doc_id] += 1.0 / (rrf_k + bm25_rank[doc_id])

            # Sort by RRF score descending, take top-k
            sorted_docs = sorted(
                (id_to_doc[doc_id] for doc_id in candidate_ids),
                key=lambda d: rrf_scores[id(d)],
                reverse=True,
            )
            return sorted_docs[:k]

    return _HybridRetriever()


def get_algorithm_builders():
    """Return a dict of name -> (builder_fn, default_kwargs) for evaluation."""
    return {
        "flat": (build_faiss_flat, {}),
        "hnsw": (build_faiss_hnsw, {"M": 32, "ef_construction": 200, "ef_search": 100}),
        "ivf": (build_faiss_ivf, {"nlist": 100, "nprobe": 10}),
        "lsh": (build_faiss_lsh, {"nbits": None}),
        "hybrid": (build_hybrid, {"dense_algorithm": "flat", "rrf_k": 60}),
    }
