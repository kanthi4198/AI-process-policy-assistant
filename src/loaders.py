"""
Document loading utilities for the ingestion pipeline.

Responsible for:
- Discovering PDF files under a directory
- Loading PDFs into LangChain `Document` objects (typically one per page)
"""

from pathlib import Path

from langchain.docstore.document import Document

# Prefer community imports (newer LangChain layout); fall back if needed.
try:
    from langchain_community.document_loaders import PyPDFLoader
except Exception:
    from langchain.document_loaders import PyPDFLoader


def find_pdfs(docs_dir: Path) -> list[Path]:
    """Recursively find all PDF files under the given directory."""
    return sorted(docs_dir.rglob("*.pdf"))


def load_pdfs(paths: list[Path]) -> list[Document]:
    """Load PDFs as a list of Documents (typically one per page)."""
    docs: list[Document] = []
    for p in paths:
        try:
            loader = PyPDFLoader(str(p))
            loaded = loader.load()
            for d in loaded:
                d.metadata = dict(d.metadata or {})
                d.metadata.update(
                    {
                        "source": str(p),
                        "format": "text",
                        "page": d.metadata.get("page"),
                    }
                )
            docs.extend(loaded)
        except Exception as e:
            print(f"Failed to load {p}: {e}")
    return docs
