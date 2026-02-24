"""Quick script to count PDFs and calculate parent chunks per PDF."""
import sys
from pathlib import Path

if __name__ == "__main__":
    _src_dir = Path(__file__).resolve().parent
    _root = _src_dir.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    if str(_src_dir) not in sys.path:
        sys.path.insert(0, str(_src_dir))

from policy_assistant.data.loaders import find_pdfs
from policy_assistant.data.chunking import load_parent_docstore

# Count PDFs
docs_dir = Path("data/docs")
pdfs = find_pdfs(docs_dir)
num_pdfs = len(pdfs)

# Count parent chunks
parent_docstore = load_parent_docstore(Path("vector_store"))
num_parents = len(parent_docstore)

# Calculate average
avg_parents_per_pdf = num_parents / num_pdfs if num_pdfs > 0 else 0

print(f"Total PDFs: {num_pdfs}")
print(f"Total parent chunks: {num_parents}")
print(f"\nAverage parent chunks per PDF: {avg_parents_per_pdf:.1f}")
print(f"\nPDF files:")
for pdf in pdfs:
    print(f"  - {pdf.name}")
