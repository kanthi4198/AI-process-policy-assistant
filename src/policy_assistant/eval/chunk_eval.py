"""
Evaluate different chunking configurations on the local policy corpus to find
the optimal parent/child chunk sizes for retrieval quality.

This script:
- Loads the raw corpus once (PDFs from data/docs)
- Builds the embedding model once (fixed across all configs)
- For each chunking configuration:
  - Creates parent + child chunks with the given sizes
  - Builds an in-memory FAISS index over the child chunks
  - Runs each eval question and measures retrieval quality (Hit@1, Hit@k)
- Prints a ranked leaderboard of all configurations

Usage (from project root): python -m policy_assistant.eval.chunk_eval ...
"""

import argparse
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from langchain_core.documents import Document

try:
    from langchain_community.vectorstores import FAISS
except Exception:
    from langchain.vectorstores import FAISS  # type: ignore

from policy_assistant.data.chunking import parent_child_chunk_documents
from policy_assistant.data.loaders import find_pdfs, load_pdfs
from policy_assistant.embeddings.local import default_hf_model, select_embeddings
from policy_assistant.eval.common import EvalItem, load_eval_items


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ChunkConfig:
    """A single chunking configuration to evaluate."""

    parent_size: int
    parent_overlap: int
    child_size: int
    child_overlap: int
    label: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = f"P{self.parent_size}/C{self.child_size}"


@dataclass
class ChunkEvalResult:
    """Evaluation result for one chunking configuration."""

    config: ChunkConfig
    num_parents: int
    num_children: int
    avg_child_len: float
    hit_at_1: float
    hit_at_k: float
    eval_questions: int
    elapsed_secs: float


# ---------------------------------------------------------------------------
# Corpus loading (raw documents — before chunking)
# ---------------------------------------------------------------------------


def load_raw_docs(docs_dir: Path) -> list[Document]:
    """Load PDFs into raw Document objects (not yet chunked)."""
    print(f"[corpus] Scanning PDFs in: {docs_dir}")
    pdf_paths = find_pdfs(docs_dir)
    print(f"[corpus] Found {len(pdf_paths)} PDF(s)")

    pdf_docs = load_pdfs(pdf_paths)
    print(f"[corpus] Loaded {len(pdf_docs)} page document(s) from PDFs")

    if not pdf_docs:
        raise RuntimeError("No documents found in corpus; nothing to evaluate.")
    return pdf_docs


# ---------------------------------------------------------------------------
# Retrieval evaluation (same logic as embed_eval)
# ---------------------------------------------------------------------------


def evaluate_retrieval(
    vs: FAISS,
    eval_items: Sequence[EvalItem],
    top_k: int,
) -> tuple[float, float, int]:
    """Return (hit_at_1, hit_at_k, scored_count) for a vector store."""
    scored = 0
    hit_at_1_count = 0
    hit_at_k_count = 0

    for item in eval_items:
        expected = item.expected_sources
        if not expected:
            continue

        scored += 1
        docs = vs.similarity_search(item.question, k=top_k)
        if not docs:
            continue

        best_rank: int | None = None
        for rank, d in enumerate(docs, start=1):
            source = str(d.metadata.get("source", "")).lower()
            parent_id = str(d.metadata.get("parent_id", "")).lower()
            matched = any(
                (substr in source) or (substr in parent_id)
                for substr in expected
            )
            if matched:
                best_rank = rank
                break

        if best_rank is not None:
            if best_rank == 1:
                hit_at_1_count += 1
            if best_rank <= top_k:
                hit_at_k_count += 1

    denom = max(scored, 1)
    return hit_at_1_count / denom, hit_at_k_count / denom, scored


# ---------------------------------------------------------------------------
# Run one configuration
# ---------------------------------------------------------------------------


def eval_one_config(
    raw_docs: list[Document],
    config: ChunkConfig,
    embeddings,
    eval_items: Sequence[EvalItem],
    top_k: int,
) -> ChunkEvalResult:
    """Chunk, embed, and evaluate one chunking configuration."""
    t0 = time.time()

    child_docs, parent_docstore = parent_child_chunk_documents(
        raw_docs,
        parent_size=config.parent_size,
        parent_overlap=config.parent_overlap,
        child_size=config.child_size,
        child_overlap=config.child_overlap,
    )

    if not child_docs:
        return ChunkEvalResult(
            config=config,
            num_parents=0,
            num_children=0,
            avg_child_len=0.0,
            hit_at_1=0.0,
            hit_at_k=0.0,
            eval_questions=0,
            elapsed_secs=time.time() - t0,
        )

    avg_child_len = statistics.mean(len(d.page_content) for d in child_docs)

    vs = FAISS.from_documents(child_docs, embeddings)
    hit1, hitk, scored = evaluate_retrieval(vs, eval_items, top_k)

    return ChunkEvalResult(
        config=config,
        num_parents=len(parent_docstore),
        num_children=len(child_docs),
        avg_child_len=avg_child_len,
        hit_at_1=hit1,
        hit_at_k=hitk,
        eval_questions=scored,
        elapsed_secs=time.time() - t0,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_configs(
    parent_sizes: list[int],
    child_sizes: list[int],
) -> list[ChunkConfig]:
    """Build a grid of ChunkConfig from parent x child size combinations.

    Overlap is derived automatically:
      - Parent overlap: ~10% of parent size (min 100)
      - Child overlap:  ~12% of child size  (min 50)

    Skips invalid combos where child_size >= parent_size.
    """
    configs: list[ChunkConfig] = []
    for ps in parent_sizes:
        for cs in child_sizes:
            if cs >= ps:
                continue  # child must be smaller than parent
            p_overlap = max(int(ps * 0.10), 100)
            c_overlap = max(int(cs * 0.12), 50)
            configs.append(
                ChunkConfig(
                    parent_size=ps,
                    parent_overlap=p_overlap,
                    child_size=cs,
                    child_overlap=c_overlap,
                )
            )
    return configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate chunking configurations on local policy corpus (retrieval metrics)."
    )

    parser.add_argument(
        "--docs_dir", type=str, default="data/docs",
        help="Directory containing PDFs.",
    )
    parser.add_argument(
        "--eval_questions_file", type=str, default="eval/eval_questions.json",
        help="JSON file with eval questions and expected_sources.",
    )

    # Grid of sizes to sweep
    parser.add_argument(
        "--parent_sizes", type=int, nargs="*",
        default=[2500, 3500, 5000, 7000],
        help="List of parent chunk sizes (characters) to test.",
    )
    parser.add_argument(
        "--child_sizes", type=int, nargs="*",
        default=[400, 600, 800, 1000, 1200, 1500],
        help="List of child chunk sizes (characters) to test.",
    )

    # Eval settings
    parser.add_argument(
        "--top_k", type=int, default=5,
        help="Number of top documents to consider for Hit@k.",
    )
    parser.add_argument(
        "--hf_model", type=str, default=None,
        help="Hugging Face embedding model (default: same as ingest).",
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

    # ---- Load eval questions ------------------------------------------------
    eval_items = load_eval_items(eval_path)
    print(f"[chunk_eval] Loaded {len(eval_items)} evaluation question(s)")

    # ---- Load raw documents (once) ------------------------------------------
    raw_docs = load_raw_docs(docs_dir)
    print(f"[chunk_eval] Total raw documents: {len(raw_docs)}")

    # ---- Load embedding model (once) ----------------------------------------
    hf_model = args.hf_model or default_hf_model()
    print(f"[chunk_eval] Loading embedding model: {hf_model}")
    embeddings = select_embeddings(hf_model)

    # ---- Build chunking grid ------------------------------------------------
    configs = build_configs(args.parent_sizes, args.child_sizes)
    if not configs:
        raise ValueError(
            "No valid configurations generated. "
            "Ensure at least one child_size < parent_size."
        )
    print(f"[chunk_eval] Testing {len(configs)} chunking configuration(s)\n")

    # ---- Evaluate each configuration ----------------------------------------
    results: list[ChunkEvalResult] = []
    for i, cfg in enumerate(configs, start=1):
        print(
            f"[{i}/{len(configs)}] {cfg.label}  "
            f"(parent={cfg.parent_size}, overlap={cfg.parent_overlap}, "
            f"child={cfg.child_size}, overlap={cfg.child_overlap})"
        )
        result = eval_one_config(
            raw_docs=raw_docs,
            config=cfg,
            embeddings=embeddings,
            eval_items=eval_items,
            top_k=args.top_k,
        )
        print(
            f"         -> children={result.num_children:>4}, "
            f"avg_len={result.avg_child_len:>6.0f}, "
            f"Hit@1={result.hit_at_1*100:5.1f}%, "
            f"Hit@{args.top_k}={result.hit_at_k*100:5.1f}%, "
            f"time={result.elapsed_secs:.1f}s"
        )
        results.append(result)

    # ---- Print leaderboard --------------------------------------------------
    results_sorted = sorted(
        results,
        key=lambda r: (r.hit_at_k, r.hit_at_1, -r.num_children),
        reverse=True,
    )

    print("\n" + "=" * 100)
    print("CHUNKING EVALUATION RESULTS")
    print(f"Embedding model: {hf_model}  |  top_k={args.top_k}  |  eval questions={results_sorted[0].eval_questions if results_sorted else 0}")
    print("=" * 100)
    print(
        f"{'Rank':<6}"
        f"{'Parent':>7}"
        f"{'Child':>7}"
        f"{'P-Ovlp':>8}"
        f"{'C-Ovlp':>8}"
        f"{'Chunks':>8}"
        f"{'AvgLen':>8}"
        f"{'Hit@1':>8}"
        f"{'Hit@K':>8}"
        f"{'Time':>8}"
    )
    print("-" * 100)
    for idx, r in enumerate(results_sorted, start=1):
        print(
            f"{idx:<6}"
            f"{r.config.parent_size:>7}"
            f"{r.config.child_size:>7}"
            f"{r.config.parent_overlap:>8}"
            f"{r.config.child_overlap:>8}"
            f"{r.num_children:>8}"
            f"{r.avg_child_len:>8.0f}"
            f"{r.hit_at_1*100:>7.1f}%"
            f"{r.hit_at_k*100:>7.1f}%"
            f"{r.elapsed_secs:>7.1f}s"
        )

    # ---- Recommendation -----------------------------------------------------
    best = results_sorted[0]
    print(f"\nBest config: parent={best.config.parent_size}, child={best.config.child_size} "
          f"(Hit@1={best.hit_at_1*100:.1f}%, Hit@{args.top_k}={best.hit_at_k*100:.1f}%)")
    print(
        f"Use in ingest:  python src/ingest.py "
        f"--parent_chunk_size {best.config.parent_size} "
        f"--parent_chunk_overlap {best.config.parent_overlap} "
        f"--child_chunk_size {best.config.child_size} "
        f"--child_chunk_overlap {best.config.child_overlap}"
    )


if __name__ == "__main__":
    main()
