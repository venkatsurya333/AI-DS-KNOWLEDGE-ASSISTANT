import pymupdf as fitz
from pathlib import Path

# Use relative path with forward slashes or raw string to avoid backslash escape sequence errors
base_dir = Path(__file__).resolve().parent.parent
pdf_path = base_dir / "data" / "documents" / "David Foster - Generative Deep Learning.pdf"

doc = fitz.open(pdf_path)

print("no.of pages:", len(doc))

for page_no, page in enumerate(doc):
    text = page.get_text()
    print(f"\n...Page {page_no+1} ---")
    print(text[:500])

doc.close()