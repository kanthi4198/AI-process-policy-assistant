"""
Ingest documents into a FAISS vector store using LangChain,
using LOCAL Hugging Face embeddings (no external API calls).

Usage:
  python src/ingest.py --docs_dir data/docs --out_dir vector_store \
    --hf_model sentence-transformers/all-MiniLM-L6-v2
  python src/ingest.py --hf_model ibm-granite/granite-embedding-278m-multilingual \
    --algorithm hnsw --parent_chunk_size 3500 --child_chunk_size 1000
"""

import os
import sys
from pathlib import Path

# Ensure project root is on path when running as python src/ingest.py
if __name__ == "__main__":
    _src_dir = Path(__file__).resolve().parent
    _root = _src_dir.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    # Scripts are in src/; package is at root as policy_assistant only when run from root
    # When cwd is root and we run python src/ingest.py, sys.path[0] is src/ - so policy_assistant lives in src/
    if str(_src_dir) not in sys.path:
        sys.path.insert(0, str(_src_dir))

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from policy_assistant.data import (
    find_pdfs,
    load_pdfs,
    parent_child_chunk_documents,
    save_parent_docstore,
)
from policy_assistant.embeddings import select_embeddings
from policy_assistant.store import build_vector_store

LANGSMITH_TRACING_ENABLED = True


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Ingest documents into a FAISS vector store (Hugging Face embeddings)"
    )
    parser.add_argument("--docs_dir", type=str, default="data/docs", help="Directory containing documents (recursively)")
    parser.add_argument("--out_dir", type=str, default="vector_store", help="Output directory for vector store")
    parser.add_argument("--parent_chunk_size", type=int, default=2500, help="Parent chunk size (characters)")
    parser.add_argument("--parent_chunk_overlap", type=int, default=250, help="Parent overlap (characters)")
    parser.add_argument("--child_chunk_size", type=int, default=1500, help="Child chunk size (characters)")
    parser.add_argument("--child_chunk_overlap", type=int, default=180, help="Child overlap (characters)")
    parser.add_argument(
        "--hf_model",
        type=str,
        default="google/embeddinggemma-300m",
        help="Hugging Face / Sentence-Transformers model name (cached) or local path",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="flat",
        choices=["flat", "hnsw"],
        help="Vector index algorithm: flat (exact k-NN) or hnsw (approximate; use same as retrieval_eval best)",
    )
    args = parser.parse_args()

    base = Path.cwd()
    docs_dir = (base / args.docs_dir).resolve()
    out_dir = (base / args.out_dir).resolve()

    print(f"Scanning PDFs in: {docs_dir}")
    pdf_paths = find_pdfs(docs_dir)
    print(f"Found {len(pdf_paths)} PDF(s)")
    all_docs = load_pdfs(pdf_paths)
    print(f"Loaded {len(all_docs)} page document(s)")

    if not all_docs:
        print("No documents found to ingest. Exiting.")
        return

    api_key = os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
    project_name = os.environ.get("LANGCHAIN_PROJECT") or "policy-assistant-ingest"
    tracing_on = LANGSMITH_TRACING_ENABLED and bool(api_key)
    if os.environ.get("DEBUG_LANGSMITH") == "1":
        print(f"[DEBUG_LANGSMITH] API key present: {bool(api_key)}, project: {project_name!r}, tracing enabled: {tracing_on}")

    def _run_ingest() -> None:
        print("Creating parent + child chunks...")
        child_chunks, parent_docstore = parent_child_chunk_documents(
            all_docs,
            parent_size=args.parent_chunk_size,
            parent_overlap=args.parent_chunk_overlap,
            child_size=args.child_chunk_size,
            child_overlap=args.child_chunk_overlap,
        )
        print(f"Created {len(child_chunks)} child chunk(s)")
        print(f"Created {len(parent_docstore)} parent chunk(s)")
        parent_path = save_parent_docstore(parent_docstore, out_dir)
        print(f"Saved parent docstore to: {parent_path}")
        print(f"Using Hugging Face embeddings model: {args.hf_model}")
        embeddings = select_embeddings(hf_model=args.hf_model)
        print(f"Building FAISS vector store (algorithm={args.algorithm}; local embeddings)...")
        build_vector_store(child_chunks, out_dir, embeddings, algorithm=args.algorithm)

    if tracing_on:
        try:
            import langsmith
            from langsmith.run_helpers import tracing_context
            client = langsmith.Client(api_key=api_key)
            with tracing_context(client=client, project_name=project_name, enabled=True):
                _run_ingest()
            client.flush()
        except Exception as e:
            if os.environ.get("DEBUG_LANGSMITH") == "1":
                print(f"[DEBUG_LANGSMITH] Tracing failed: {e}")
            _run_ingest()
    else:
        _run_ingest()


if __name__ == "__main__":
    main()
