"""
Reusable chat completion wrapper for Ollama.

Supports both blocking and streaming modes so callers (eval scripts,
Streamlit app, CLI) can share one implementation.
"""

from __future__ import annotations

from typing import Generator

import ollama


def chat_completion(
    model_id: str,
    messages: list[dict],
    *,
    stream: bool = False,
    temperature: float = 0.0,
) -> str | Generator[str, None, None]:
    """Call Ollama and return the full response or a token-by-token generator.

    Parameters
    ----------
    model_id : str
        Ollama model tag, e.g. ``"olmo2:7b"``.
    messages : list[dict]
        OpenAI-style message list (role / content dicts).
    stream : bool
        If *True*, return a generator that yields content deltas.
    temperature : float
        Sampling temperature (0.0 = deterministic).
    """
    if stream:
        return _stream(model_id, messages, temperature)

    r = ollama.chat(
        model=model_id,
        messages=messages,
        options={"temperature": temperature},
    )
    return r["message"]["content"]


def _stream(
    model_id: str,
    messages: list[dict],
    temperature: float,
) -> Generator[str, None, None]:
    """Yield content token strings as they arrive from Ollama."""
    for chunk in ollama.chat(
        model=model_id,
        messages=messages,
        options={"temperature": temperature},
        stream=True,
    ):
        token = chunk.get("message", {}).get("content", "")
        if token:
            yield token
