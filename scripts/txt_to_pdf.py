"""
Convert all .txt files in data/docs/ to PDFs, then delete the .txt originals.

Usage:
  pip install fpdf2
  python scripts/txt_to_pdf.py

The script scans data/docs/ for .txt files, creates a PDF with the same name,
and removes the .txt file after successful conversion.
"""

from pathlib import Path

try:
    from fpdf import FPDF
except ImportError:
    print("fpdf2 is required. Install it with:  pip install fpdf2")
    raise SystemExit(1)


def txt_to_pdf(txt_path: Path, pdf_path: Path) -> None:
    """Convert a single text file to a PDF."""
    text = txt_path.read_text(encoding="utf-8")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)

    for line in text.splitlines():
        # encode to latin-1 safe characters (FPDF built-in fonts are latin-1)
        safe_line = line.encode("latin-1", errors="replace").decode("latin-1")
        pdf.cell(0, 6, safe_line, new_x="LMARGIN", new_y="NEXT")

    pdf.output(str(pdf_path))


def main() -> None:
    docs_dir = Path(__file__).resolve().parent.parent / "data" / "docs"
    txt_files = sorted(docs_dir.glob("*.txt"))

    if not txt_files:
        print("No .txt files found in data/docs/")
        return

    for txt_path in txt_files:
        pdf_path = txt_path.with_suffix(".pdf")
        if pdf_path.exists():
            print(f"  SKIP (PDF already exists): {pdf_path.name}")
            continue
        print(f"  Converting: {txt_path.name} -> {pdf_path.name}")
        txt_to_pdf(txt_path, pdf_path)

    # Delete .txt files after all conversions succeed
    for txt_path in txt_files:
        pdf_path = txt_path.with_suffix(".pdf")
        if pdf_path.exists():
            txt_path.unlink()
            print(f"  Deleted: {txt_path.name}")
        else:
            print(f"  KEPT (no PDF created): {txt_path.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
