"""Quick utility to count chunks in your vector store."""
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

from policy_assistant.data.chunking import load_parent_docstore
from policy_assistant.store.vectorstore import CONFIG_FILENAME, CHILD_DOCSTORE_FILENAME

def count_chunks(vector_store_dir: Path | str = "vector_store"):
    """Count and display chunk statistics."""
    base = Path.cwd()
    out_dir = (base / str(vector_store_dir)).resolve()
    
    if not out_dir.exists():
        print(f"Error: Vector store directory not found: {out_dir}")
        return
    
    # Count parent chunks
    try:
        parent_docstore = load_parent_docstore(out_dir)
        num_parents = len(parent_docstore)
        print(f"Parent chunks: {num_parents:,}")
    except Exception as e:
        print(f"Could not load parent chunks: {e}")
        num_parents = 0
    
    # Count child chunks
    config_path = out_dir / CONFIG_FILENAME
    child_docstore_path = out_dir / CHILD_DOCSTORE_FILENAME
    
    if config_path.exists():
        import json
        with open(config_path) as f:
            config = json.load(f)
        algorithm = config.get("algorithm", "flat")
        
        if algorithm == "hnsw" and child_docstore_path.exists():
            import pickle
            try:
                with open(child_docstore_path, "rb") as f:
                    child_docs = pickle.load(f)
                num_children = len(child_docs)
                print(f"Child chunks (HNSW): {num_children:,}")
            except Exception as e:
                print(f"Could not load child chunks: {e}")
                num_children = 0
        else:
            try:
                from policy_assistant.embeddings.local import select_embeddings
                from policy_assistant.store.vectorstore import load_vector_store
                hf_model = os.environ.get("HF_MODEL", "google/embeddinggemma-300m")
                embeddings = select_embeddings(hf_model)
                vector_store = load_vector_store(out_dir, embeddings)
                if hasattr(vector_store, "index"):
                    num_children = vector_store.index.ntotal
                elif hasattr(vector_store, "_index"):
                    num_children = vector_store._index.ntotal
                else:
                    num_children = "unknown"
                print(f"Child chunks (Flat): {num_children:,}" if isinstance(num_children, int) else f"Child chunks (Flat): {num_children}")
            except Exception as e:
                print(f"Could not count child chunks: {e}")
                num_children = 0
    else:
        try:
            from policy_assistant.embeddings.local import select_embeddings
            from policy_assistant.store.vectorstore import load_vector_store
            hf_model = os.environ.get("HF_MODEL", "google/embeddinggemma-300m")
            embeddings = select_embeddings(hf_model)
            vector_store = load_vector_store(out_dir, embeddings)
            if hasattr(vector_store, "index"):
                num_children = vector_store.index.ntotal
            elif hasattr(vector_store, "_index"):
                num_children = vector_store._index.ntotal
            else:
                num_children = "unknown"
            print(f"Child chunks (Flat): {num_children:,}" if isinstance(num_children, int) else f"Child chunks (Flat): {num_children}")
        except Exception as e:
            print(f"Could not count child chunks: {e}")
            num_children = 0
    
    print("\n" + "=" * 50)
    if isinstance(num_children, int) and num_parents > 0:
        ratio = num_children / num_parents if num_parents > 0 else 0
        print(f"Summary:")
        print(f"  Parent chunks: {num_parents:,}")
        print(f"  Child chunks: {num_children:,}")
        print(f"  Average children per parent: {ratio:.2f}")
    print("=" * 50)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Count chunks in vector store")
    parser.add_argument("--vector_store_dir", type=str, default="vector_store", help="Directory containing vector store")
    args = parser.parse_args()
    count_chunks(args.vector_store_dir)
