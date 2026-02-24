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
    _env_dir = Path(__file__).resolve().parent.parent
    load_dotenv(_env_dir / ".env")
except Exception:
    pass

# Setup LangSmith tracing BEFORE importing LangChain components
api_key = os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
if api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    if not os.environ.get("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_API_KEY"] = api_key
    # Set explicit endpoint if not already set (default: https://api.smith.langchain.com)
    if not os.environ.get("LANGCHAIN_ENDPOINT") and not os.environ.get("LANGSMITH_ENDPOINT"):
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from policy_assistant.retrieval import get_context_for_llm, load_retrieval_artifacts

DEFAULT_VECTOR_STORE_DIR = Path("vector_store")
DEFAULT_HF_MODEL = "google/embeddinggemma-300m"


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

    # Use project name from env or default to policy-assistant-ingest
    project_name = os.environ.get("LANGCHAIN_PROJECT") or os.environ.get("LANGSMITH_PROJECT") or "policy-assistant-ingest"
    
    vector_store_dir = Path(args.vector_store_dir)
    
    if api_key:
        if os.environ.get("DEBUG_LANGSMITH") == "1":
            endpoint = os.environ.get("LANGCHAIN_ENDPOINT") or os.environ.get("LANGSMITH_ENDPOINT") or "https://api.smith.langchain.com"
            print(f"[DEBUG] Tracing enabled - API key: {bool(api_key)}, Project: {project_name}, Endpoint: {endpoint}")
        
        try:
            import langsmith as ls
            from langchain_core.tracers import LangChainTracer
            from langchain_core.callbacks import CallbackManager
            
            # Verify API key and test connection
            if os.environ.get("DEBUG_LANGSMITH") == "1":
                try:
                    test_client = ls.Client(api_key=api_key)
                    print(f"[DEBUG] LangSmith client created successfully")
                except Exception as e:
                    print(f"[DEBUG] LangSmith client creation failed: {e}")
            
            # Create explicit LangChain tracer
            tracer = LangChainTracer(project_name=project_name)
            callback_manager = CallbackManager([tracer])
            
            # Load artifacts and run retrieval inside tracing context
            with ls.tracing_context(project_name=project_name, enabled=True):
                vector_store, parent_docstore, _ = load_retrieval_artifacts(vector_store_dir, args.hf_model)
                
                # Test if vector store is LangChain FAISS (supports callbacks) or custom
                if os.environ.get("DEBUG_LANGSMITH") == "1":
                    vs_type = type(vector_store).__name__
                    print(f"[DEBUG] Vector store type: {vs_type}")
                    if hasattr(vector_store, 'similarity_search'):
                        import inspect
                        sig = inspect.signature(vector_store.similarity_search)
                        supports_callbacks = 'callbacks' in sig.parameters
                        print(f"[DEBUG] Vector store supports callbacks: {supports_callbacks}")
                
                context = get_context_for_llm(
                    args.query,
                    vector_store,
                    parent_docstore,
                    k=args.k,
                    use_parent_content=not args.no_parent_expand,
                    callbacks=callback_manager,
                    project_name=project_name,
                )
                
                if os.environ.get("DEBUG_LANGSMITH") == "1":
                    print(f"[DEBUG] Retrieval completed, checking tracer...")
                    print(f"[DEBUG] Tracer runs: {len(getattr(tracer, '_runs', []))}")
        except Exception as e:
            if os.environ.get("DEBUG_LANGSMITH") == "1":
                print(f"[DEBUG] Tracing error: {e}")
                import traceback
                traceback.print_exc()
            # Fallback to non-traced execution
            vector_store, parent_docstore, _ = load_retrieval_artifacts(vector_store_dir, args.hf_model)
            context = get_context_for_llm(
                args.query,
                vector_store,
                parent_docstore,
                k=args.k,
                use_parent_content=not args.no_parent_expand,
            )
    else:
        vector_store, parent_docstore, _ = load_retrieval_artifacts(vector_store_dir, args.hf_model)
        context = get_context_for_llm(
            args.query,
            vector_store,
            parent_docstore,
            k=args.k,
            use_parent_content=not args.no_parent_expand,
        )

    print(context)


if __name__ == "__main__":
    main()
