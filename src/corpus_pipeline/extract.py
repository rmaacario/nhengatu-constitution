"""
extract.py
----------
PDF → raw text.

Strategy:
  1. Try pdftotext (poppler) with -layout flag — best quality for this document.
  2. Fall back to pdfplumber if pdftotext is unavailable.
  3. Fall back to pypdf as last resort.

All three paths return a plain Python str.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract(pdf_path: str | Path) -> str:
    """
    Extract raw text from *pdf_path*.

    Returns the full text as a single string.
    Raises FileNotFoundError if the PDF doesn't exist.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    text = _try_pdftotext(path)
    if text:
        return text

    text = _try_pdfplumber(path)
    if text:
        return text

    return _try_pypdf(path)


# ---------------------------------------------------------------------------
# Extraction backends
# ---------------------------------------------------------------------------

def _try_pdftotext(path: Path) -> str | None:
    """
    Use poppler's pdftotext with -layout flag.
    Returns None if pdftotext is not installed.
    """
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", str(path), "-"],
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
        )
        text = result.stdout
        if len(text.strip()) > 100:          # sanity check — non-empty output
            return text
        return None
    except FileNotFoundError:
        print(
            "  [extract] pdftotext not found — falling back to pdfplumber.",
            file=sys.stderr,
        )
        return None
    except subprocess.CalledProcessError as exc:
        print(f"  [extract] pdftotext error: {exc.stderr[:200]}", file=sys.stderr)
        return None


def _try_pdfplumber(path: Path) -> str | None:
    """
    Use pdfplumber for layout-aware extraction.
    Returns None if pdfplumber is not installed.
    """
    try:
        import pdfplumber  # noqa: PLC0415
    except ImportError:
        print(
            "  [extract] pdfplumber not installed — falling back to pypdf.",
            file=sys.stderr,
        )
        return None

    pages = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            t = page.extract_text(layout=True)
            if t:
                pages.append(t)

    text = "\n\f\n".join(pages)
    return text if len(text.strip()) > 100 else None


def _try_pypdf(path: Path) -> str:
    """
    Use pypdf as last-resort extractor.
    Raises ImportError if pypdf is not installed.
    """
    try:
        from pypdf import PdfReader  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "No PDF extraction backend available. "
            "Install at least one of: poppler-utils (pdftotext), pdfplumber, pypdf."
        ) from exc

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            pages.append(t)
    return "\n\f\n".join(pages)
