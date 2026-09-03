from pathlib import Path

import pytest

from app.ingestion.loader import load_filing


def test_load_filing_rejects_pdf_without_pdf_header(tmp_path: Path):
    invalid_pdf = tmp_path / "invalid.pdf"
    invalid_pdf.write_bytes(b"not a pdf")

    with pytest.raises(ValueError, match=r"File does not appear to be a valid PDF \(invalid header\)"):
        load_filing(str(invalid_pdf))