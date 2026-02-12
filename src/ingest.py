"""
Ingest documents into a FAISS vector store using LangChain,
using LOCAL Hugging Face embeddings (no external API calls).

This script will:
- Recursively load PDF and text files from a docs directory
- Create parent + child chunks (children go to vector store; parents go to a docstore)
- Create embeddings using a Hugging Face / Sentence-Transformers model (local cache or path)
- Persist a FAISS vector store + a parent-docstore mapping to disk

Usage examples:
  # Hugging Face embeddings (downloads/caches locally or uses local path)
  python src/ingest.py --docs_dir data/docs --out_dir vector_store \
    --hf_model sentence-transformers/all-MiniLM-L6-v2

  # Hugging Face model from local path
  python src/ingest.py --hf_model /path/to/local/model
"""

import argparse
from pathlib import Path

from chunking import parent_child_chunk_documents, save_parent_docstore
from embeddings_local import default_hf_model, select_embeddings
from loaders import find_pdfs, load_pdfs
from vectorstore_utils import build_vector_store


def main() -> None:
    """CLI entrypoint for the ingestion pipeline."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into a FAISS vector store (Hugging Face embeddings)"
    )
    parser.add_argument("--docs_dir", type=str, default="data/docs", help="Directory containing documents (recursively)")
    parser.add_argument("--out_dir", type=str, default="vector_store", help="Output directory for vector store")

    # Text chunking
    parser.add_argument(
        "--parent_chunk_size",
        type=int,
        default=2500,
        help="Parent chunk size (characters)",
    )
    parser.add_argument(
        "--parent_chunk_overlap",
        type=int,
        default=250,
        help="Parent overlap (characters)",
    )
    parser.add_argument(
        "--child_chunk_size",
        type=int,
        default=1500,
        help="Child chunk size (characters)",
    )
    parser.add_argument(
        "--child_chunk_overlap",
        type=int,
        default=180,
        help="Child overlap (characters)",
    )

    # Hugging Face embeddings
    parser.add_argument(
        "--hf_model",
        type=str,
        default="google/embeddinggemma-300m",
        help="Hugging Face / Sentence-Transformers model name (cached) or local path",
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

    print("Building FAISS vector store (local embeddings; no external API calls)...")
    build_vector_store(child_chunks, out_dir, embeddings)


if __name__ == "__main__":
    main()