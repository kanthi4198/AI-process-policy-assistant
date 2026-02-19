"""Local (Hugging Face / Sentence-Transformers) embedding models."""

from policy_assistant.embeddings.local import (
    default_hf_model,
    select_embeddings,
)

__all__ = [
    "default_hf_model",
    "select_embeddings",
]
