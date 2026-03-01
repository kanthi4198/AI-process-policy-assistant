"""
Load the RAG scoring rubric and compute weighted scores.

Use with eval/RAG_SCORING_RUBRIC.md and eval/rag_rubric.json.
"""

import json
from pathlib import Path
from typing import Any


def load_rubric(path: Path | None = None) -> dict[str, Any]:
    """Load the rubric from JSON. If path is None, looks for eval/rag_rubric.json from cwd."""
    if path is None:
        path = Path.cwd() / "eval" / "rag_rubric.json"
    if not path.exists():
        raise FileNotFoundError(f"Rubric not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_criterion_weights(rubric: dict[str, Any]) -> dict[str, float]:
    """Return a flat dict of criterion_id -> weight from the rubric."""
    weights: dict[str, float] = {}
    for cat in rubric.get("categories", []):
        for c in cat.get("criteria", []):
            cid = c.get("id")
            w = c.get("weight", 0.0)
            if cid is not None:
                weights[cid] = w
    return weights


def compute_weighted_score(
    rubric: dict[str, Any],
    scores: dict[str, int],
) -> tuple[float, float, float]:
    """
    Compute raw weighted score and normalized scores from criterion scores.

    Args:
        rubric: Loaded rubric (from load_rubric).
        scores: Dict mapping criterion_id -> score (0, 1, or 2). Missing criteria are treated as 0.

    Returns:
        (raw_score, normalized_0_1, normalized_0_5).
        raw_score range depends on weights and max_score_per_criterion.
        normalized_0_1 in [0, 1], normalized_0_5 in [0, 5].
    """
    weights = get_criterion_weights(rubric)
    raw = 0.0
    for cid, w in weights.items():
        raw += w * scores.get(cid, 0)
    max_per = rubric.get("max_score_per_criterion", 2)
    max_raw = max_per * sum(weights.values())
    normalized_0_1 = raw / max_raw if max_raw else 0.0
    normalized_0_5 = normalized_0_1 * 5.0
    return (raw, normalized_0_1, normalized_0_5)


def list_criterion_ids(rubric: dict[str, Any]) -> list[str]:
    """Return ordered list of criterion IDs for consistent scoring sheets."""
    ids: list[str] = []
    for cat in rubric.get("categories", []):
        for c in cat.get("criteria", []):
            cid = c.get("id")
            if cid is not None:
                ids.append(cid)
    return ids
