"""
Step 1: Load raw text out of your source documents.

Supports .pdf, .md, and .txt. Nothing clever here — this is the part
frameworks barely abstract at all, so we keep it dead simple.
"""

from pathlib import Path
from pypdf import PdfReader


def load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n".join(pages)


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_documents(doc_dir: str) -> list[dict]:
    """
    Walk doc_dir, load every .pdf/.md/.txt file, return a list of
    {"source": filename, "text": full_text} dicts — one per file.

    We keep the whole document's text in one string per file and let
    the chunker (chunking.py) decide how to split it. Splitting here
    AND in the chunker would just make bugs harder to find.
    """
    doc_dir = Path(doc_dir)
    if not doc_dir.exists():
        raise FileNotFoundError(
            f"'{doc_dir}' doesn't exist. Put your PDFs/markdown files there."
        )

    docs = []
    for path in sorted(doc_dir.iterdir()):
        if path.suffix.lower() == ".pdf":
            text = load_pdf(path)
        elif path.suffix.lower() in (".md", ".txt"):
            text = load_text(path)
        else:
            continue

        text = text.strip()
        if not text:
            print(f"  [warn] {path.name} produced no extractable text — skipping")
            continue

        docs.append({"source": path.name, "text": text})

    if not docs:
        raise ValueError(
            f"No .pdf/.md/.txt files found in '{doc_dir}'. "
            "Drop 5-10 documents in there and rerun."
        )

    return docs
