import json
import sys
from pathlib import Path


def main() -> None:
    """
    Recompute aggregate_scores and weighted_score for judge_results_olmo2-7b.json
    based on the existing per-item scores.
    """
    root = Path(__file__).resolve().parent

    # Ensure we can import policy_assistant.*
    src_path = root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from policy_assistant.eval.rag_rubric import (  # type: ignore
        load_rubric,
        compute_weighted_score,
    )

    eval_dir = root / "eval"
    judge_path = eval_dir / "judge_results_olmo2-7b.json"
    rubric_path = eval_dir / "rag_rubric.json"

    data = json.loads(judge_path.read_text(encoding="utf-8"))
    items = data.get("item_results", [])

    # Rebuild aggregated_scores from item_results
    aggregated_scores: dict[str, list[int]] = {}
    for item in items:
        scores = item.get("scores", {})
        for cid, entry in scores.items():
            if not isinstance(entry, dict):
                continue
            score_val = entry.get("score", 0)
            aggregated_scores.setdefault(cid, []).append(score_val)

    # Compute new aggregate_scores
    agg_out: dict[str, dict] = {}
    for cid, values in aggregated_scores.items():
        mean = sum(values) / len(values) if values else 0.0
        agg_out[cid] = {
            "mean": mean,
            "rounded": round(mean),
            "n": len(values),
        }

    # Compute new weighted_score using rag_rubric
    rubric = load_rubric(rubric_path)
    rounded_means = {cid: info["rounded"] for cid, info in agg_out.items()}
    raw, norm_01, norm_05 = compute_weighted_score(rubric, rounded_means)

    data["aggregate_scores"] = agg_out
    data["weighted_score"] = {
        "raw": raw,
        "normalized_0_1": norm_01,
        "normalized_0_5": norm_05,
    }

    judge_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Recomputed aggregates for:", judge_path)
    print(f"  Weighted raw: {raw:.3f}")
    print(f"  Normalized  : {norm_01:.3f} (0-1) / {norm_05:.2f} (0-5)")


if __name__ == "__main__":
    main()

