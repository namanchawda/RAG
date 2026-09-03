"""Document loading utilities for SEC filings and other input sources.

Phase 1 keeps this intentionally simple: there is no structural parsing of SEC
sections such as Item 1A or Item 7. We extract raw text only and defer section
understanding and semantics to a later phase.
"""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader


def _read_text_with_fallback(path: Path) -> str:
    """Read a file as text using a practical fallback sequence for SEC HTML files."""
    raw = path.read_bytes()
    encodings = ("utf-8-sig", "utf-8", "cp1252", "latin-1")

    for encoding in encodings:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue

    return raw.decode("utf-8", errors="replace")


def load_filing(filepath: str) -> str:
    """Load a SEC filing from a local plain-text, PDF, or HTML file and return extracted text.

    Phase 1 intentionally does no structural parsing or section detection. We are
    only extracting the raw contents of the filing so downstream chunking and
    retrieval can operate on plain text in the naive baseline.
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"Filing not found: {filepath}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        if not path.read_bytes().startswith(b"%PDF-"):
            raise ValueError("File does not appear to be a valid PDF (invalid header)")

        reader = PdfReader(str(path))
        pages: list[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)
        return "\n".join(pages)

    if suffix in {".txt", ".md", ".rtf"}:
        return _read_text_with_fallback(path)

    if suffix in {".html", ".htm"}:
        html = _read_text_with_fallback(path)
        soup = BeautifulSoup(html, "html.parser")

        for tag in list(soup.find_all(True)):
            tag_name = (tag.name or "").lower()
            if tag_name.startswith("ix:") or tag_name.startswith("xbrli:"):
                tag.decompose()

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        cleaned = soup.get_text(separator="\n")
        return "\n".join(line.strip() for line in cleaned.splitlines() if line.strip())

    raise ValueError(
        f"Unsupported filing format for {filepath}. Supported formats: .txt, .md, .rtf, .pdf, .html, .htm"
    )


if __name__ == "__main__":
    sample_text = """This is a sample SEC filing excerpt.
    Management's discussion and analysis highlights revenue growth and operating margin.
    The company continues to invest in research and development and expand its product portfolio.
    Risk factors include competition, macroeconomic volatility, and supply chain disruptions.
    """

    sample_path = Path("sample_filing.txt")
    sample_path.write_text(sample_text, encoding="utf-8")

    try:
        extracted = load_filing(str(sample_path))
        print("Loaded filing text preview:")
        print(extracted[:500])
    finally:
        if sample_path.exists():
            sample_path.unlink()
