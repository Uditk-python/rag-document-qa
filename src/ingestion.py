"""
Ingestion: turn a PDF or .txt file on disk into a list of "pages" with
enough metadata (source filename, page number) to support citations later.

Everything downstream (chunking, retrieval, citation display) depends on
this metadata surviving the whole pipeline -- so we attach it here, once,
at the source.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

from pypdf import PdfReader


@dataclass
class Page:
    """One page (or, for .txt files, the whole document) of raw text."""
    text: str
    source: str        # original filename
    page_number: int    # 1-indexed; 1 for non-paginated formats


def load_pdf(path: Path) -> List[Page]:
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append(Page(text=text, source=path.name, page_number=i))
    return pages


def load_text(path: Path) -> List[Page]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []
    return [Page(text=text, source=path.name, page_number=1)]


def load_document(path: Path) -> List[Page]:
    """Dispatch on file extension. Raises ValueError for unsupported types."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    elif suffix in (".txt", ".md"):
        return load_text(path)
    else:
        raise ValueError(
            f"Unsupported file type '{suffix}' for {path.name}. "
            "Supported: .pdf, .txt, .md"
        )


def load_documents(paths: List[Path]) -> List[Page]:
    """Load and flatten multiple documents into one list of Page objects."""
    all_pages: List[Page] = []
    for p in paths:
        try:
            all_pages.extend(load_document(p))
        except Exception as e:
            # Don't let one bad file kill the whole ingestion batch --
            # surface the problem but keep processing the rest.
            print(f"[ingestion] Failed to load {p.name}: {e}")
    return all_pages
