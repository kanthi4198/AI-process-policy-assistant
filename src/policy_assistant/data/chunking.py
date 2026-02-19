"""
Parent-child chunking utilities for the ingestion pipeline.

Responsible for:
- Converting loaded Document objects into parent and child chunks
- Creating stable parent IDs
- Persisting the parent docstore mapping
"""

import pickle
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def make_parent_id(meta_base: dict, p_idx: int) -> str:
    """Create a stable parent_id that avoids collisions across pages/docs."""
    source = meta_base.get("source", "unknown")
    page = meta_base.get("page")

    if page is not None:
        return f"{source}::page{page}::p{p_idx}"
    return f"{source}::p{p_idx}"


def parent_child_chunk_documents(
    docs: list[Document],
    parent_size: int,
    parent_overlap: int,
    child_size: int,
    child_overlap: int,
) -> tuple[list[Document], dict[str, Document]]:
    """
    Create child chunks (for vector search) and a parent docstore mapping.

    Returns:
      - child_docs: list[Document] to embed + store in FAISS
      - parent_docstore: dict[parent_id, Document] stored separately (not embedded)
    """
    child_docs: list[Document] = []
    parent_docstore: dict[str, Document] = {}

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_size,
        chunk_overlap=parent_overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " "],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_size,
        chunk_overlap=child_overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " "],
    )

    for doc in docs:
        meta_base = dict(doc.metadata or {})
        parent_chunks = parent_splitter.split_text(doc.page_content)

        for p_idx, parent_text in enumerate(parent_chunks):
            parent_id = make_parent_id(meta_base, p_idx)

            # Store parent in docstore (NOT embedded)
            parent_meta = {
                **meta_base,
                "parent_id": parent_id,
                "parent_index": p_idx,
                "level": "parent",
                "parent_size": len(parent_text),
            }
            parent_docstore[parent_id] = Document(page_content=parent_text, metadata=parent_meta)

            # Split into children (embedded + stored in FAISS)
            child_texts = child_splitter.split_text(parent_text)
            for c_idx, child_text in enumerate(child_texts):
                child_meta = {
                    **meta_base,
                    "parent_id": parent_id,
                    "parent_index": p_idx,
                    "chunk": c_idx,
                    "level": "child",
                    "parent_size": len(parent_text),
                    "child_size": len(child_text),
                }
                child_docs.append(Document(page_content=child_text, metadata=child_meta))

    return child_docs, parent_docstore


def save_parent_docstore(parent_docstore: dict[str, Document], out_dir: Path) -> Path:
    """Persist parent docstore mapping to disk via pickle."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "parent_docstore.pkl"
    with path.open("wb") as f:
        pickle.dump(parent_docstore, f)
    return path


def load_parent_docstore(out_dir: Path) -> dict[str, Document]:
    """Load parent docstore mapping from disk (as saved by save_parent_docstore)."""
    path = out_dir / "parent_docstore.pkl"
    with path.open("rb") as f:
        return pickle.load(f)
