import csv
import json
import sys
from pathlib import Path


def main() -> None:
    """
    Export per-criterion mean scores and overall weighted scores for all
    evaluated models into a CSV that you can open in Excel.

    Input:  eval/judge_results_*.json  (one per model, as written by run_generator_eval.py)
    Output: eval/model_comparison.csv
    """
    root = Path(__file__).resolve().parent
    eval_dir = root / "eval"

    # Ensure we can import policy_assistant.*
    src_path = root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from policy_assistant.eval.rag_rubric import (  # type: ignore
        load_rubric,
        list_criterion_ids,
    )

    # Load rubric to get stable criterion ordering
    rubric = load_rubric(eval_dir / "rag_rubric.json")
    criterion_ids = list_criterion_ids(rubric)

    # Collect all judge_results_*.json files, excluding any *_subset.json
    judge_files = sorted(
        p for p in eval_dir.glob("judge_results_*.json")
        if not p.name.endswith("_subset.json")
    )
    if not judge_files:
        print("No judge_results_*.json files found in eval/.")
        return

    out_path = eval_dir / "model_comparison.csv"

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Header: model + weighted + each criterion
        header = [
            "model",
            "weighted_raw",
            "weighted_norm_0_1",
            "weighted_norm_0_5",
        ] + [f"mean_{cid}" for cid in criterion_ids]
        writer.writerow(header)

        for path in judge_files:
            data = json.loads(path.read_text(encoding="utf-8"))

            # Derive a human-readable model name from filename
            # judge_results_phi3.5-mini.json -> phi3.5-mini
            stem = path.stem  # judge_results_<slug>
            slug = stem.replace("judge_results_", "", 1)
            model_name = slug

            ws = data.get("weighted_score", {}) or {}
            raw = ws.get("raw")
            n01 = ws.get("normalized_0_1")
            n05 = ws.get("normalized_0_5")

            means = data.get("mean_scores", {}) or {}

            row = [
                model_name,
                f"{raw:.6f}" if isinstance(raw, (int, float)) else "",
                f"{n01:.6f}" if isinstance(n01, (int, float)) else "",
                f"{n05:.6f}" if isinstance(n05, (int, float)) else "",
            ]
            for cid in criterion_ids:
                v = means.get(cid)
                row.append(f"{v:.6f}" if isinstance(v, (int, float)) else "")

            writer.writerow(row)

    print("Wrote:", out_path)


if __name__ == "__main__":
    main()

