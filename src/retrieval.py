"""
Retrieval from the FAISS vector store and parent docstore.

To choose the best retrieval algorithm, run:
  python -m policy_assistant.eval.retrieval_eval --save_best eval/best_retrieval.json
"""

import os
import sys
from pathlib import Path

if __name__ == "__main__":
    _src_dir = Path(__file__).resolve().parent
    _root = _src_dir.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    if str(_src_dir) not in sys.path:
        sys.path.insert(0, str(_src_dir))

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from policy_assistant.retrieval import get_context_for_llm, load_retrieval_artifacts

DEFAULT_VECTOR_STORE_DIR = Path("vector_store")
DEFAULT_HF_MODEL = "google/embeddinggemma-300m"
LANGSMITH_TRACING_ENABLED = True


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Retrieve context from vector store for a query")
    parser.add_argument("query", type=str, help="Query string")
    parser.add_argument("--vector_store_dir", type=str, default=str(DEFAULT_VECTOR_STORE_DIR),
                        help="Directory containing FAISS index and parent_docstore.pkl")
    parser.add_argument("--k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--no_parent_expand", action="store_true", help="Use child chunk text instead of expanding to full parent")
    parser.add_argument("--hf_model", type=str, default=DEFAULT_HF_MODEL, help="Hugging Face embedding model (must match ingest)")
    args = parser.parse_args()

    api_key = os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
    project_name = os.environ.get("LANGCHAIN_PROJECT") or "policy-assistant-retrieval"
    tracing_on = LANGSMITH_TRACING_ENABLED and bool(api_key)
    if os.environ.get("DEBUG_LANGSMITH") == "1":
        print(f"[DEBUG_LANGSMITH] API key present: {bool(api_key)}, project: {project_name!r}, tracing enabled: {tracing_on}")

    def _run_retrieval() -> str:
        vector_store_dir = Path(args.vector_store_dir)
        vector_store, parent_docstore, _ = load_retrieval_artifacts(vector_store_dir, args.hf_model)
        return get_context_for_llm(
            args.query,
            vector_store,
            parent_docstore,
            k=args.k,
            use_parent_content=not args.no_parent_expand,
        )

    if tracing_on:
        try:
            import langsmith
            from langsmith.run_helpers import tracing_context
            client = langsmith.Client(api_key=api_key)
            with tracing_context(client=client, project_name=project_name, enabled=True):
                context = _run_retrieval()
            client.flush()
        except Exception as e:
            if os.environ.get("DEBUG_LANGSMITH") == "1":
                print(f"[DEBUG_LANGSMITH] Tracing failed: {e}")
            context = _run_retrieval()
    else:
        context = _run_retrieval()

    print(context)


if __name__ == "__main__":
    main()
