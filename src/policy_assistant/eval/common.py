"""
Shared evaluation data structures and loaders.

Used by embed_eval, chunk_eval, and retrieval_eval.
"""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvalItem:
    id: str
    question: str
    expected_sources: list[str]


def load_eval_items(path: Path) -> list[EvalItem]:
    """Load eval questions from JSON file."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("eval JSON must be a list of {id, question, expected_sources}")

    items: list[EvalItem] = []
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ValueError(f"eval entry at index {i} must be an object")
        raw_sources = entry.get("expected_sources", [])
        expected_sources = [str(s).lower() for s in raw_sources] if isinstance(raw_sources, list) else []
        items.append(
            EvalItem(
                id=str(entry.get("id", "")),
                question=str(entry.get("question", "")),
                expected_sources=expected_sources,
            )
        )
    return items
