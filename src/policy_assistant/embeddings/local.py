"""
Local embeddings utilities for the ingestion pipeline.

Uses Hugging Face / Sentence-Transformers embeddings (local cache or local path).
Includes automatic prefix detection for instruction-tuned embedding models.
Automatically uses CUDA GPU when available for accelerated inference.
"""

import os
from typing import Any

import torch
from langchain_core.embeddings import Embeddings

try:
    from langchain_huggingface.embeddings import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore[import-untyped]



# Prefix configurations for instruction-tuned embedding models

# Keys are substrings matched against the model name (case-insensitive).
# Values are the query prefix prepended at search time.
#
# Models not listed here (e.g. gte-multilingual-base, granite-embedding-*,
# embeddinggemma) are non-instruct and require no prefixes.
# Valid IBM Granite embedding IDs: granite-embedding-278m-multilingual,
# granite-embedding-125m-english, granite-embedding-english-r2, etc.

MODEL_PREFIX_REGISTRY: dict[str, str] = {
    "Qwen3-Embedding": (
        "Instruct: Given a clarification query regarding company and external policy, retrieve relevant passages"
        "that answer the query\nQuery: "
    ),
    "gte-Qwen2-1.5B-instruct": (
        "Instruct: Given a clarification query regarding company and external policy, retrieve relevant passages"
        "that answer the query\nQuery: "
    ),
    "multilingual-e5-large-instruct": (
        "Instruct: Given a clarification query regarding company and external policy, retrieve relevant passages"
        "that answer the query\nQuery: "
    ),
}


def _lookup_query_prefix(model_name_or_path: str) -> str:
    """Look up query prefix by matching model name substrings (case-insensitive)."""
    name_lower = model_name_or_path.lower()
    for key, prefix in MODEL_PREFIX_REGISTRY.items():
        if key.lower() in name_lower:
            return prefix
    return ""


# ---------------------------------------------------------------------------
# Prefix wrapper
# ---------------------------------------------------------------------------


class PrefixedEmbeddings(Embeddings):
    """Wrapper around HuggingFaceEmbeddings that prepends a query prefix
    for instruction-tuned models.

    Implements the LangChain ``Embeddings`` interface so vector stores
    (e.g. FAISS) treat it as a proper embeddings object instead of a
    generic callable.
    """

    def __init__(self, base_embeddings: Embeddings, query_prefix: str = "") -> None:
        self._base = base_embeddings
        self._query_prefix = query_prefix

    # ---- Embeddings interface ---------------------------------------------

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Documents are embedded as-is; only queries are prefixed. This keeps
        # stored vectors consistent while still using instruct prompts at
        # query time.
        return self._base.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        prefixed = self._query_prefix + text if self._query_prefix else text
        return self._base.embed_query(prefixed)

    # ---- Delegation / repr -------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def __repr__(self) -> str:
        model = getattr(self._base, "model_name", "unknown")
        return (
            f"PrefixedEmbeddings(model={model!r}, "
            f"query_prefix={self._query_prefix!r})"
        )


# ---------------------------------------------------------------------------
# Builder functions
# ---------------------------------------------------------------------------


def _resolve_device() -> str:
    """Return device string for embedding model: prefer GPU when available or when forced via env."""
    # Allow forcing device via env (e.g. EMBEDDING_DEVICE=cuda for GPU)
    env_device = os.environ.get("EMBEDDING_DEVICE", "").strip().lower()
    if env_device in ("cuda", "cuda:0", "gpu"):
        device = "cuda"
        if not torch.cuda.is_available():
            print(
                "[embeddings] EMBEDDING_DEVICE=cuda but PyTorch reports no CUDA. "
                "Install CUDA-enabled PyTorch, e.g.: pip install torch --index-url https://download.pytorch.org/whl/cu121"
            )
        else:
            gpu_name = torch.cuda.get_device_name(0)
            cuda_ver = getattr(torch.version, "cuda", None) or "N/A"
            print(f"[embeddings] Using GPU: {gpu_name} (CUDA {cuda_ver})")
        return device
    if env_device in ("cpu",):
        print("[embeddings] Using CPU (EMBEDDING_DEVICE=cpu)")
        return "cpu"

    # Auto-detect
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        cuda_ver = getattr(torch.version, "cuda", None) or "N/A"
        print(f"[embeddings] Using GPU: {gpu_name} (CUDA {cuda_ver})")
    else:
        print(
            "[embeddings] No CUDA GPU detected; using CPU. "
            "To force GPU set EMBEDDING_DEVICE=cuda in .env (requires CUDA-enabled PyTorch)."
        )
    return device


def build_hf_embeddings(
    model_name_or_path: str,
    query_prefix: str = "",
):
    """Build local HuggingFace/Sentence-Transformers embeddings.

    Automatically uses CUDA GPU when available for accelerated inference.

    If *query_prefix* is provided, the returned object is a
    ``PrefixedEmbeddings`` wrapper that transparently prepends the prefix
    on ``embed_query`` calls.
    """
    if HuggingFaceEmbeddings is None:
        raise RuntimeError(
            "HuggingFaceEmbeddings not available. Install dependencies:\n"
            "  pip install -U langchain-huggingface sentence-transformers transformers"
        )

    device = _resolve_device()

    # normalize embeddings if supported (helps cosine/IP similarity)
    try:
        base = HuggingFaceEmbeddings(
            model_name=model_name_or_path,
            # Some models (e.g. Alibaba-NLP/gte-multilingual-base) require
            # `trust_remote_code=True` to load their custom implementations.
            # We enable this here explicitly, since we are only loading
            # vetted embedding models from Hugging Face.
            model_kwargs={"device": device, "trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": True},
        )
    except TypeError:
        # For older versions that don't support encode_kwargs / model_kwargs
        base = HuggingFaceEmbeddings(model_name=model_name_or_path)

    # Wrap with prefix support if a query prefix is specified
    if query_prefix:
        return PrefixedEmbeddings(base, query_prefix)

    return base


def select_embeddings(hf_model: str):
    """Build Hugging Face embeddings with auto-detected prefix configuration.

    Looks up the model name in ``MODEL_PREFIX_REGISTRY`` and applies the
    correct query prefix for instruction-tuned models.  Non-instruct models
    are returned without any wrapper.
    """
    return build_hf_embeddings(
        hf_model,
        query_prefix=_lookup_query_prefix(hf_model),
    )


def default_hf_model() -> str:
    """Resolve the default HF model from env or fallback value."""
    return os.getenv("HF_MODEL_NAME_OR_PATH", "google/embeddinggemma-300m")
