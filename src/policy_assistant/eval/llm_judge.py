"""
LLM-as-Judge evaluation using the Gemini free tier.

Scores RAG system responses against the v2 rubric criteria using a structured
prompt that feeds the rubric definitions, key_points, and expected sources to
the judge model. Returns per-criterion scores (0/1/2) that plug directly into
compute_weighted_score().

Usage:
    python -m policy_assistant.eval.llm_judge \
        --rubric eval/questions/v2_rubric.json \
        --scoring_rubric eval/rag_rubric.json \
        --responses eval/responses.json \
        --out eval/judge_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from policy_assistant.eval.common import load_eval_items
from policy_assistant.eval.rag_rubric import compute_weighted_score, load_rubric

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert evaluation judge for a corporate policy RAG (Retrieval-Augmented \
Generation) system. Your task is to score the system's response to a policy question \
using a detailed rubric.

You MUST be strict and objective. Base scores ONLY on the evidence provided. \
Do not give credit for information not present in the response."""

SINGLE_TURN_PROMPT = """\
## Evaluation Task

Score the following RAG system response on the criteria listed below.

### Question Asked
{question}

### System Response
{response}

### Retrieved Source IDs
{retrieved_sources}

### Expected Source IDs (ground truth)
{expected_sources}

### Key Points (ground truth checklist)
The response MUST cover these points to score well on correctness and completeness:
{key_points}

---

## Scoring Criteria

Score each criterion on a 0-1-2 scale as defined below. Only score criteria \
listed in "Criteria to Score" — skip the rest.

### Criteria to Score
{criteria_to_score}

### Full Criteria Definitions

{criteria_definitions}

---

## Output Format

Return ONLY a JSON object (no markdown fences, no commentary) with this structure:

{{
  "scores": {{
    "<criterion_id>": {{
      "score": <0 | 1 | 2>,
      "justification": "<1-2 sentence explanation>"
    }}
  }}
}}

Score each criterion in "Criteria to Score" exactly once. Do not add extra keys."""

MULTI_TURN_PROMPT = """\
## Evaluation Task (Multi-Turn)

Score turn {turn_number} of a multi-turn conversation sequence.

### Conversation History
{conversation_history}

### Current Turn Question
{question}

### System Response (this turn)
{response}

### Retrieved Source IDs (this turn)
{retrieved_sources}

### Expected Source IDs (ground truth)
{expected_sources}

### Key Points (ground truth checklist for this turn)
{key_points}

---

## Scoring Criteria

Score each criterion on a 0-1-2 scale as defined below. Only score criteria \
listed in "Criteria to Score" — skip the rest.

### Criteria to Score
{criteria_to_score}

### Full Criteria Definitions

{criteria_definitions}

### Additional Multi-Turn Criteria

In addition to the standard criteria above, also score these:

- **context_retention** (0-2): Does the system remember and use information from \
prior turns without requiring the user to repeat themselves?
  - 0 = Ignores prior context entirely
  - 1 = Partially uses prior context
  - 2 = Fully retains and leverages prior context

- **cross_turn_consistency** (0-2): Are the answers across turns consistent \
(no contradictions with earlier responses)?
  - 0 = Contradicts earlier turns
  - 1 = Minor inconsistencies
  - 2 = Fully consistent across turns

- **progressive_refinement** (0-2): Does the system correctly narrow or expand \
scope based on follow-up questions?
  - 0 = Ignores the refinement direction
  - 1 = Partially adjusts scope
  - 2 = Correctly refines scope based on the follow-up

---

## Output Format

Return ONLY a JSON object (no markdown fences, no commentary) with this structure:

{{
  "scores": {{
    "<criterion_id>": {{
      "score": <0 | 1 | 2>,
      "justification": "<1-2 sentence explanation>"
    }},
    "context_retention": {{
      "score": <0 | 1 | 2>,
      "justification": "<1-2 sentence explanation>"
    }},
    "cross_turn_consistency": {{
      "score": <0 | 1 | 2>,
      "justification": "<1-2 sentence explanation>"
    }},
    "progressive_refinement": {{
      "score": <0 | 1 | 2>,
      "justification": "<1-2 sentence explanation>"
    }}
  }}
}}

Score each criterion in "Criteria to Score" plus the three multi-turn criteria \
exactly once. Do not add extra keys."""


# ---------------------------------------------------------------------------
# Build criterion definitions block from rag_rubric.json
# ---------------------------------------------------------------------------


def _format_criteria_definitions(rubric: dict[str, Any]) -> str:
    """Render every criterion from the scoring rubric as a readable text block."""
    lines: list[str] = []
    for cat in rubric.get("categories", []):
        for c in cat.get("criteria", []):
            cid = c["id"]
            name = c.get("name", cid)
            weight = c.get("weight", 0)
            metric = c.get("metric", "")
            scores = c.get("scores", {})
            measures = c.get("measure", [])
            lines.append(f"**{name}** (`{cid}`, weight={weight})")
            if metric:
                lines.append(f"  Metric: {metric}")
            if measures:
                lines.append("  Guiding questions: " + " | ".join(measures))
            for level in ("0", "1", "2"):
                if level in scores:
                    lines.append(f"  {level} = {scores[level]}")
            lines.append("")
    return "\n".join(lines)


def _format_key_points(key_points: list[str]) -> str:
    return "\n".join(f"  - {kp}" for kp in key_points) if key_points else "  (none provided)"


# ---------------------------------------------------------------------------
# Local judge client (Llama 3.1 8B via Ollama)
# ---------------------------------------------------------------------------

import ollama

_JUDGE_MODEL_ID = "llama3.1:8b"
_MAX_RETRIES = 3
_RETRY_BACKOFF = 2.0


def _init_judge() -> str:
    """Return the Ollama model ID to use as the judge."""
    return _JUDGE_MODEL_ID


def _call_judge(model_id: str, prompt: str) -> dict[str, Any]:
    """Call local Llama judge via Ollama and parse the JSON response, with retries."""
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = ollama.chat(
                model=model_id,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0.0},
            )
            text = response["message"]["content"].strip()
            # Strip optional markdown fences
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                # Fallback: extract the first JSON-looking substring
                start = text.find("{")
                end = text.rfind("}") + 1
                if start != -1 and end > start:
                    candidate = text[start:end]
                    return json.loads(candidate)
                raise
        except json.JSONDecodeError as e:
            print(f"  [judge] JSON parse error on attempt {attempt}: {e}")
            if attempt == _MAX_RETRIES:
                raise
        except Exception as e:
            print(f"  [judge] API error on attempt {attempt}: {e}")
            if attempt == _MAX_RETRIES:
                raise
            time.sleep(_RETRY_BACKOFF * attempt)
    return {}


# ---------------------------------------------------------------------------
# Score a single eval item
# ---------------------------------------------------------------------------


def score_item(
    model: str,
    item: dict[str, Any],
    response_text: str,
    retrieved_sources: list[str],
    criteria_defs: str,
    conversation_history: str | None = None,
) -> dict[str, Any]:
    """
    Build the judge prompt for one eval item and return parsed scores.

    Args:
        model: Initialized Gemini model.
        item: One entry from v2_rubric.json "items" array.
        response_text: The RAG system's answer to score.
        retrieved_sources: Source IDs the retriever returned (e.g. ["CORP-01", "CORP-03"]).
        criteria_defs: Pre-formatted criteria definitions string.
        conversation_history: For multi-turn items, formatted prior turns.

    Returns:
        {"scores": {criterion_id: {"score": int, "justification": str}, ...}}
    """
    question = item["question"]
    expected_sources = item.get("expected_sources", [])
    key_points = item.get("key_points", [])
    criteria_focus = item.get("criteria_focus", [])
    is_multi_turn = item.get("question_type") == "multi_turn"

    criteria_to_score = ", ".join(f"`{c}`" for c in criteria_focus) if criteria_focus else "(all applicable criteria)"

    if is_multi_turn:
        mt = item.get("multi_turn", {})
        prompt = MULTI_TURN_PROMPT.format(
            turn_number=mt.get("turn", "?"),
            conversation_history=conversation_history or "(first turn — no prior history)",
            question=question,
            response=response_text,
            retrieved_sources=", ".join(retrieved_sources) if retrieved_sources else "(none)",
            expected_sources=", ".join(expected_sources),
            key_points=_format_key_points(key_points),
            criteria_to_score=criteria_to_score,
            criteria_definitions=criteria_defs,
        )
    else:
        prompt = SINGLE_TURN_PROMPT.format(
            question=question,
            response=response_text,
            retrieved_sources=", ".join(retrieved_sources) if retrieved_sources else "(none)",
            expected_sources=", ".join(expected_sources),
            key_points=_format_key_points(key_points),
            criteria_to_score=criteria_to_score,
            criteria_definitions=criteria_defs,
        )

    return _call_judge(model, prompt)


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------


def run_judge_eval(
    rubric_items_path: Path,
    scoring_rubric_path: Path,
    responses_path: Path,
) -> dict[str, Any]:
    """
    Run the LLM judge over all items.

    Args:
        rubric_items_path: Path to v2_rubric.json (questions + key_points).
        scoring_rubric_path: Path to rag_rubric.json (criteria weights + definitions).
        responses_path: Path to a JSON file with structure:
            [
                {
                    "id": "v2_q1",
                    "response": "The system's answer text...",
                    "retrieved_sources": ["CORP-01"],
                    "latency_seconds": 1.3
                },
                ...
            ]

    Returns:
        Full results dict with per-item scores and aggregate weighted score.
    """
    with rubric_items_path.open("r", encoding="utf-8") as f:
        rubric_data = json.load(f)
    items_by_id: dict[str, dict] = {item["id"]: item for item in rubric_data["items"]}

    scoring_rubric = load_rubric(scoring_rubric_path)
    criteria_defs = _format_criteria_definitions(scoring_rubric)

    with responses_path.open("r", encoding="utf-8") as f:
        responses = json.load(f)
    responses_by_id: dict[str, dict] = {r["id"]: r for r in responses}

    model = _init_judge()
    all_results: list[dict[str, Any]] = []
    aggregated_scores: dict[str, list[int]] = {}

    # Track multi-turn conversation histories
    mt_histories: dict[str, list[dict[str, str]]] = {}

    for item_id, item in items_by_id.items():
        resp_entry = responses_by_id.get(item_id)
        if not resp_entry:
            print(f"  [judge] No response found for {item_id}, skipping.")
            continue

        response_text = resp_entry.get("response", "")
        retrieved_sources = resp_entry.get("retrieved_sources", [])

        # Build conversation history for multi-turn
        conversation_history = None
        if item.get("question_type") == "multi_turn":
            seq_id = item["multi_turn"]["sequence_id"]
            if seq_id not in mt_histories:
                mt_histories[seq_id] = []
            if mt_histories[seq_id]:
                history_lines = []
                for i, turn in enumerate(mt_histories[seq_id], 1):
                    history_lines.append(f"Turn {i} Q: {turn['question']}")
                    history_lines.append(f"Turn {i} A: {turn['response']}")
                conversation_history = "\n".join(history_lines)

        print(f"  [judge] Scoring {item_id} ...")
        result = score_item(
            model=model,
            item=item,
            response_text=response_text,
            retrieved_sources=retrieved_sources,
            criteria_defs=criteria_defs,
            conversation_history=conversation_history,
        )

        # Inject deterministic scores the LLM shouldn't judge
        scores_dict = result.get("scores", {})

        # Latency — scored from measured time, not by the LLM
        if item.get("measure_latency") and "latency_seconds" in resp_entry:
            t = resp_entry["latency_seconds"]
            latency_score = 2 if t < 2 else (1 if t <= 5 else 0)
            scores_dict["latency"] = {
                "score": latency_score,
                "justification": f"Measured latency: {t:.2f}s",
            }

        result["item_id"] = item_id
        result["scores"] = scores_dict
        all_results.append(result)

        # Collect for aggregation
        for cid, entry in scores_dict.items():
            score_val = entry.get("score", 0) if isinstance(entry, dict) else 0
            aggregated_scores.setdefault(cid, []).append(score_val)

        # Update multi-turn history
        if item.get("question_type") == "multi_turn":
            seq_id = item["multi_turn"]["sequence_id"]
            mt_histories[seq_id].append({
                "question": item["question"],
                "response": response_text,
            })

        # Rate-limit courtesy for free tier (15 RPM for Gemini 2.0 Flash free)
        time.sleep(4.5)

    # Compute aggregate: average each criterion, then weighted score
    avg_scores: dict[str, int] = {}
    for cid, values in aggregated_scores.items():
        mean = sum(values) / len(values)
        avg_scores[cid] = round(mean)

    raw, norm_01, norm_05 = compute_weighted_score(scoring_rubric, avg_scores)

    return {
        "item_results": all_results,
        "aggregate_scores": {
            cid: {
                "mean": sum(v) / len(v),
                "rounded": round(sum(v) / len(v)),
                "n": len(v),
            }
            for cid, v in aggregated_scores.items()
        },
        "weighted_score": {
            "raw": raw,
            "normalized_0_1": norm_01,
            "normalized_0_5": norm_05,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM-as-Judge evaluation on RAG responses.")
    parser.add_argument(
        "--rubric",
        type=str,
        default="eval/questions/v2_rubric.json",
        help="Path to v2_rubric.json (questions + key_points).",
    )
    parser.add_argument(
        "--scoring_rubric",
        type=str,
        default="eval/rag_rubric.json",
        help="Path to rag_rubric.json (criteria weights).",
    )
    parser.add_argument(
        "--responses",
        type=str,
        required=True,
        help="Path to JSON with system responses to evaluate.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="eval/judge_results.json",
        help="Where to save judge results.",
    )

    args = parser.parse_args()
    base = _project_root()

    results = run_judge_eval(
        rubric_items_path=(base / args.rubric).resolve(),
        scoring_rubric_path=(base / args.scoring_rubric).resolve(),
        responses_path=(base / args.responses).resolve(),
    )

    out_path = (base / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    ws = results["weighted_score"]
    print(f"\n{'='*60}")
    print(f"Judge evaluation complete.")
    print(f"  Items scored : {len(results['item_results'])}")
    print(f"  Weighted raw : {ws['raw']:.3f}")
    print(f"  Normalized   : {ws['normalized_0_1']:.3f} (0-1) / {ws['normalized_0_5']:.2f} (0-5)")
    print(f"  Results saved: {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
