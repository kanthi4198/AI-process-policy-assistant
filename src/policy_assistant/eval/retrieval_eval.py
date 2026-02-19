"""
Retrieval algorithm evaluation pipeline.

Evaluates multiple retrieval algorithms (Flat, HNSW, IVF, LSH, Hybrid) on the
policy corpus using eval/eval_questions.json. Computes Hit@1, Hit@k, MRR, Precision and Recall;
recommends the best algorithm for production.

Usage (from project root): python -m policy_assistant.eval.retrieval_eval ...
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from langchain_core.documents import Document

from policy_assistant.data.chunking import parent_child_chunk_documents
from policy_assistant.data.loaders import find_pdfs, load_pdfs
from policy_assistant.embeddings.local import select_embeddings
from policy_assistant.eval.common import EvalItem, load_eval_items
from policy_assistant.retrieval.algorithms import get_algorithm_builders


# ---------------------------------------------------------------------------
# Eval data structures
# ---------------------------------------------------------------------------


@dataclass
class AlgorithmResult:
    name: str
    hit_at_1: float
    hit_at_k: float
    mrr: float
    precision: float
    recall: float
    total_questions: int
    scored_count: int


# ---------------------------------------------------------------------------
# Load eval questions and corpus
# ---------------------------------------------------------------------------


def load_corpus(
    docs_dir: Path,
    parent_chunk_size: int,
    parent_chunk_overlap: int,
    child_chunk_size: int,
    child_chunk_overlap: int,
) -> tuple[list[Document], dict[str, Document]]:
    """Load PDFs and create child chunks + parent docstore (same as ingest)."""
    print(f"[corpus] Scanning PDFs in: {docs_dir}")
    pdf_paths = find_pdfs(docs_dir)
    print(f"[corpus] Found {len(pdf_paths)} PDF(s)")
    pdf_docs = load_pdfs(pdf_paths)
    print(f"[corpus] Loaded {len(pdf_docs)} page document(s)")
    if not pdf_docs:
        raise RuntimeError("No documents found in corpus.")
    print("[corpus] Creating parent + child chunks...")
    child_chunks, parent_docstore = parent_child_chunk_documents(
        pdf_docs,
        parent_size=parent_chunk_size,
        parent_overlap=parent_chunk_overlap,
        child_size=child_chunk_size,
        child_overlap=child_chunk_overlap,
    )
    print(f"[corpus] Created {len(child_chunks)} child chunk(s)")
    return child_chunks, parent_docstore


# ---------------------------------------------------------------------------
# Metrics: Hit@1, Hit@k, MRR, Precision@k, Recall@k
# ---------------------------------------------------------------------------


def _doc_is_relevant(doc: Document, expected_sources: list[str]) -> bool:
    """True if doc's source or parent_id matches any expected_sources substring."""
    source = str(doc.metadata.get("source", "")).lower()
    parent_id = str(doc.metadata.get("parent_id", "")).lower()
    return any((s in source) or (s in parent_id) for s in expected_sources)


def _first_relevant_rank(docs: list[Document], expected_sources: list[str]) -> int | None:
    """Return 1-based rank of first doc whose source/parent_id matches any expected; None if none."""
    for rank, d in enumerate(docs, start=1):
        if _doc_is_relevant(d, expected_sources):
            return rank
    return None


def _count_relevant_in_corpus(child_docs: list[Document], expected_sources: list[str]) -> int:
    """Number of docs in child_docs that are relevant for the given expected_sources."""
    return sum(1 for d in child_docs if _doc_is_relevant(d, expected_sources))


def evaluate_retriever(
    retriever,
    eval_items: Sequence[EvalItem],
    top_k: int,
    child_docs: list[Document],
) -> AlgorithmResult:
    """Compute Hit@1, Hit@k, MRR, Precision@k, and Recall@k for a retriever."""
    hit_at_1_count = 0
    hit_at_k_count = 0
    mrr_sum = 0.0
    precision_sum = 0.0
    recall_sum = 0.0
    scored_count = 0
    recall_denom = 0  # number of queries with at least one relevant doc in corpus

    for item in eval_items:
        if not item.expected_sources:
            continue
        scored_count += 1
        total_relevant = _count_relevant_in_corpus(child_docs, item.expected_sources)
        docs = retriever.similarity_search(item.question, k=top_k)
        rank = _first_relevant_rank(docs, item.expected_sources)
        relevant_in_topk = sum(1 for d in docs if _doc_is_relevant(d, item.expected_sources))

        if rank is not None:
            if rank == 1:
                hit_at_1_count += 1
            if rank <= top_k:
                hit_at_k_count += 1
            mrr_sum += 1.0 / rank

        # Precision@k = relevant in top-k / k
        precision_sum += relevant_in_topk / top_k
        # Recall@k = relevant in top-k / total relevant (for this query); skip if no relevant in corpus
        if total_relevant > 0:
            recall_sum += relevant_in_topk / total_relevant
            recall_denom += 1

    n = max(scored_count, 1)
    avg_precision = precision_sum / n
    avg_recall = recall_sum / max(recall_denom, 1)
    return AlgorithmResult(
        name="",  # filled by caller
        hit_at_1=hit_at_1_count / n,
        hit_at_k=hit_at_k_count / n,
        mrr=mrr_sum / n,
        precision=avg_precision,
        recall=avg_recall,
        total_questions=len(eval_items),
        scored_count=scored_count,
    )


# ---------------------------------------------------------------------------
# Run evaluation for all selected algorithms
# ---------------------------------------------------------------------------


def run_evaluation(
    child_docs: list[Document],
    embeddings,
    eval_items: list[EvalItem],
    top_k: int,
    algorithms: list[str],
) -> list[AlgorithmResult]:
    """Build each selected algorithm, evaluate, return results."""
    builders = get_algorithm_builders()
    results: list[AlgorithmResult] = []

    for name in algorithms:
        if name not in builders:
            print(f"[warn] Unknown algorithm {name!r}, skipping.")
            continue
        fn, default_kwargs = builders[name]
        print(f"[eval] Building and evaluating: {name} ...")
        try:
            retriever = fn(child_docs, embeddings, **default_kwargs)
            res = evaluate_retriever(retriever, eval_items, top_k, child_docs)
            res.name = name
            results.append(res)
        except Exception as e:
            print(f"[warn] Algorithm {name} failed: {e}")
            continue

    return results


# ---------------------------------------------------------------------------
# Output: table + best recommendation
# ---------------------------------------------------------------------------


def print_results(results: list[AlgorithmResult], top_k: int) -> None:
    """Print a leaderboard table."""
    if not results:
        print("No results to display.")
        return
    # Sort by Hit@k desc, then MRR desc, then Hit@1 desc
    sorted_results = sorted(
        results,
        key=lambda r: (r.hit_at_k, r.mrr, r.hit_at_1),
        reverse=True,
    )
    print("\n" + "=" * 80)
    print("Retrieval algorithm evaluation")
    print("=" * 80)
    print(f"Metrics: Hit@1, Hit@{top_k}, MRR, Precision@{top_k}, Recall@{top_k} (higher is better)")
    print("-" * 95)
    print(f"{'Algorithm':<12} {'Hit@1':>8} {'Hit@K':>8} {'MRR':>8} {'Prec':>8} {'Recall':>8} {'N':>6}")
    print("-" * 95)
    for r in sorted_results:
        print(
            f"{r.name:<12} {r.hit_at_1*100:7.1f}% {r.hit_at_k*100:7.1f}% {r.mrr:8.3f} {r.precision*100:7.1f}% {r.recall*100:7.1f}% {r.scored_count:6d}"
        )
    print("=" * 80)
    best = sorted_results[0]
    print(f"\nRecommended for production: {best.name!r} (best Hit@{top_k} and MRR)")
    print("Set this in retrieval config or use the same index type at ingest.")


def save_best_algorithm(results: list[AlgorithmResult], out_path: Path) -> None:
    """Write the best algorithm name to a JSON file for production config."""
    if not results:
        return
    sorted_results = sorted(
        results,
        key=lambda r: (r.hit_at_k, r.mrr, r.hit_at_1),
        reverse=True,
    )
    best = sorted_results[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_algorithm": best.name,
                "metrics": {
                    "hit_at_k": best.hit_at_k,
                    "mrr": best.mrr,
                    "precision": best.precision,
                    "recall": best.recall,
                },
            },
            f,
            indent=2,
        )
    print(f"\nBest algorithm saved to: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval algorithms (Flat, HNSW, IVF, LSH, Hybrid) on policy corpus."
    )
    parser.add_argument("--docs_dir", type=str, default="data/docs", help="Directory containing PDFs.")
    parser.add_argument(
        "--eval_questions_file",
        type=str,
        default="eval/eval_questions.json",
        help="JSON file with questions and expected_sources.",
    )
    parser.add_argument(
        "--parent_chunk_size",
        type=int,
        default=2500,
        help="Parent chunk size (match ingest).",
    )
    parser.add_argument(
        "--parent_chunk_overlap",
        type=int,
        default=250,
        help="Parent chunk overlap.",
    )
    parser.add_argument(
        "--child_chunk_size",
        type=int,
        default=1500,
        help="Child chunk size (match ingest).",
    )
    parser.add_argument(
        "--child_chunk_overlap",
        type=int,
        default=180,
        help="Child chunk overlap.",
    )
    parser.add_argument("--top_k", type=int, default=5, help="k for Hit@k and retrieval.")
    parser.add_argument(
        "--algorithms",
        type=str,
        nargs="*",
        default=["flat", "hnsw", "ivf", "lsh", "hybrid"],
        help="Algorithms to evaluate: flat, hnsw, ivf, lsh, hybrid.",
    )
    parser.add_argument(
        "--hf_model",
        type=str,
        default="google/embeddinggemma-300m",
        help="Embedding model (must match ingest).",
    )
    parser.add_argument(
        "--save_best",
        type=str,
        default="",
        help="Path to save best algorithm name (e.g. eval/best_retrieval.json).",
    )
    return parser.parse_args()


def _project_root() -> Path:
    """Project root (parent of src/), so paths work when run from any cwd."""
    return Path(__file__).resolve().parent.parent.parent.parent


def main() -> None:
    args = parse_args()
    base = _project_root()
    docs_dir = (base / args.docs_dir).resolve()
    eval_path = (base / args.eval_questions_file).resolve()

    if not eval_path.exists():
        raise FileNotFoundError(f"Eval questions file not found: {eval_path}")

    eval_items = load_eval_items(eval_path)
    print(f"[retrieval_eval] Loaded {len(eval_items)} evaluation question(s) from {eval_path}")

    child_docs, _ = load_corpus(
        docs_dir=docs_dir,
        parent_chunk_size=args.parent_chunk_size,
        parent_chunk_overlap=args.parent_chunk_overlap,
        child_chunk_size=args.child_chunk_size,
        child_chunk_overlap=args.child_chunk_overlap,
    )

    print(f"[retrieval_eval] Loading embeddings: {args.hf_model}")
    embeddings = select_embeddings(args.hf_model)

    results = run_evaluation(
        child_docs=child_docs,
        embeddings=embeddings,
        eval_items=eval_items,
        top_k=args.top_k,
        algorithms=args.algorithms,
    )

    print_results(results, args.top_k)

    if args.save_best:
        save_best_algorithm(results, base / args.save_best)


if __name__ == "__main__":
    main()
