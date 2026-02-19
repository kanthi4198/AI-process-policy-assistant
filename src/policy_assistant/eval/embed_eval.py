"""
Evaluate different embedding models on the local policy corpus by measuring
retrieval quality for a set of evaluation questions.

This script reuses the ingestion pipeline from ingest (same loaders and
chunking) but **does not** persist a vector store. Instead, for each candidate
embedding model it:

- Loads and chunks the corpus (PDFs from `data/docs`)
- Builds an in-memory FAISS index
- For each eval question:
  - Runs similarity search to get top-k chunks
  - Checks whether any of the top-k chunks come from an expected source
- Computes simple retrieval metrics (Hit@1, Hit@k) per model

Usage (from project root): python -m policy_assistant.eval.embed_eval ...
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from langchain_core.documents import Document

try:
    from langchain_community.vectorstores import FAISS
except Exception:
    from langchain.vectorstores import FAISS  # type: ignore

from policy_assistant.data.chunking import parent_child_chunk_documents
from policy_assistant.data.loaders import find_pdfs, load_pdfs
from policy_assistant.embeddings.local import select_embeddings
from policy_assistant.eval.common import EvalItem, load_eval_items


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ModelEvalResult:
    model_name: str
    provider: str
    total_questions: int
    hit_at_1: float
    hit_at_k: float


# ---------------------------------------------------------------------------
# Corpus loading & vector store build (in-memory)
# ---------------------------------------------------------------------------


def load_corpus(
    docs_dir: Path,
    parent_chunk_size: int,
    parent_chunk_overlap: int,
    child_chunk_size: int,
    child_chunk_overlap: int,
) -> tuple[list[Document], dict[str, Document]]:
    """Load PDFs and create child chunks + parent docstore."""
    print(f"[corpus] Scanning PDFs in: {docs_dir}")
    pdf_paths = find_pdfs(docs_dir)
    print(f"[corpus] Found {len(pdf_paths)} PDF(s)")

    pdf_docs = load_pdfs(pdf_paths)
    print(f"[corpus] Loaded {len(pdf_docs)} page document(s) from PDFs")

    if not pdf_docs:
        raise RuntimeError("No documents found in corpus; nothing to evaluate.")

    print("[corpus] Creating parent + child chunks...")
    child_chunks, parent_docstore = parent_child_chunk_documents(
        pdf_docs,
        parent_size=parent_chunk_size,
        parent_overlap=parent_chunk_overlap,
        child_size=child_chunk_size,
        child_overlap=child_chunk_overlap,
    )
    print(f"[corpus] Created {len(child_chunks)} child chunk(s)")
    print(f"[corpus] Created {len(parent_docstore)} parent chunk(s)")

    return child_chunks, parent_docstore


def build_vector_store(child_docs: list[Document], embeddings):
    """Build an in-memory FAISS vector store from child docs."""
    print(f"[vs] Building FAISS index over {len(child_docs)} chunks...")
    vs = FAISS.from_documents(child_docs, embeddings)
    return vs


# ---------------------------------------------------------------------------
# Retrieval evaluation
# ---------------------------------------------------------------------------


def evaluate_model_on_questions(
    model_name: str,
    provider: str,
    vs: FAISS,
    eval_items: Sequence[EvalItem],
    top_k: int,
) -> ModelEvalResult:
    """Compute simple retrieval metrics (Hit@1, Hit@k) for one model."""
    total = len(eval_items)
    hit_at_1_count = 0
    hit_at_k_count = 0

    for item in eval_items:
        query = item.question
        expected = item.expected_sources

        if not expected:
            # If no ground truth is specified, skip from metrics but still run
            # retrieval to exercise the pipeline.
            docs: list[Document] = vs.similarity_search(query, k=top_k)
            print(f"[warn] Eval item {item.id} has no expected_sources; skipping for metrics.")
            continue

        docs = vs.similarity_search(query, k=top_k)
        if not docs:
            continue

        # Determine the best (lowest) rank where any expected source matches.
        best_rank: int | None = None

        for rank, d in enumerate(docs, start=1):
            source = str(d.metadata.get("source", "")).lower()
            parent_id = str(d.metadata.get("parent_id", "")).lower()

            # Match on any substring of source or parent_id.
            matched = any(
                (substr in source) or (substr in parent_id) for substr in expected
            )
            if matched:
                best_rank = rank
                break

        if best_rank is not None:
            if best_rank == 1:
                hit_at_1_count += 1
            if best_rank <= top_k:
                hit_at_k_count += 1

    # Avoid division by zero
    denom = max(total, 1)
    return ModelEvalResult(
        model_name=model_name,
        provider=provider,
        total_questions=total,
        hit_at_1=hit_at_1_count / denom,
        hit_at_k=hit_at_k_count / denom,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate different embedding models on local policy corpus (retrieval-level metrics)."
    )

    parser.add_argument(
        "--docs_dir",
        type=str,
        default="data/docs",
        help="Directory containing PDFs (recursively).",
    )
    parser.add_argument(
        "--eval_questions_file",
        type=str,
        default="eval/eval_questions.json",
        help="JSON file with eval questions and expected_sources.",
    )

    # Chunking settings (mirroring ingest defaults)
    parser.add_argument(
        "--parent_chunk_size",
        type=int,
        default=5000,
        help="Parent chunk size (characters).",
    )
    parser.add_argument(
        "--parent_chunk_overlap",
        type=int,
        default=300,
        help="Parent overlap (characters).",
    )
    parser.add_argument(
        "--child_chunk_size",
        type=int,
        default=1200,
        help="Child chunk size (characters).",
    )
    parser.add_argument(
        "--child_chunk_overlap",
        type=int,
        default=150,
        help="Child overlap (characters).",
    )

    # Evaluation settings
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top documents to consider for Hit@k.",
    )

    # Hugging Face models to evaluate
    parser.add_argument(
        "--hf_models",
        type=str,
        nargs="*",
        help="List of Hugging Face / sentence-transformers model names or local paths to evaluate.",
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
    print(f"[embed_eval] Loaded {len(eval_items)} evaluation question(s) from: {eval_path}")

    # Load and chunk corpus once; reuse for all models.
    child_docs, parent_docstore = load_corpus(
        docs_dir=docs_dir,
        parent_chunk_size=args.parent_chunk_size,
        parent_chunk_overlap=args.parent_chunk_overlap,
        child_chunk_size=args.child_chunk_size,
        child_chunk_overlap=args.child_chunk_overlap,
    )
    # parent_docstore is available if you later want to do parent-level evaluation.
    _ = parent_docstore

    if not args.hf_models:
        raise ValueError("No HF models provided. Use --hf_models model1 model2 ...")

    results: list[ModelEvalResult] = []
    model_names: Sequence[str] = args.hf_models
    for model_name in model_names:
        print(f"\n[model] Evaluating Hugging Face model: {model_name}")
        embeddings = select_embeddings(model_name)
        vs = build_vector_store(child_docs, embeddings)
        result = evaluate_model_on_questions(
            model_name=model_name,
            provider="hf",
            vs=vs,
            eval_items=eval_items,
            top_k=args.top_k,
        )
        results.append(result)

    # Print leaderboard
    print("\n======================")
    print("Retrieval Evaluation Results")
    print("======================")
    if not results:
        print("No results computed (no models?).")
        return

    # Sort by Hit@k descending, then Hit@1
    results_sorted = sorted(
        results,
        key=lambda r: (r.hit_at_k, r.hit_at_1),
        reverse=True,
    )

    print(f"{'Rank':<6}{'Provider':<10}{'Model':<45}{'Hit@1':>8}{'Hit@K':>8}{'N':>6}")
    for idx, r in enumerate(results_sorted, start=1):
        print(
            f"{idx:<6}{r.provider:<10}{r.model_name:<45}"
            f"{r.hit_at_1*100:7.1f}%{r.hit_at_k*100:7.1f}%{r.total_questions:6d}"
        )


if __name__ == "__main__":
    main()
