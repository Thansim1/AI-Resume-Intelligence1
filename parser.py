"""Resume and job-description text extraction utilities."""
from __future__ import annotations

import re
from pathlib import Path


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract selectable text from PDF bytes using PyMuPDF."""
    try:
        import fitz
    except ImportError as exc:
        raise ImportError("PyMuPDF is missing. Install dependencies with: pip install -r requirements.txt") from exc

    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as document:
            pages = [page.get_text("text") for page in document]
        return "\n".join(pages).strip()
    except Exception as exc:
        raise ValueError(f"Could not read this PDF: {exc}") from exc


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Decode plain text, trying UTF-8 before a permissive fallback."""
    try:
        return file_bytes.decode("utf-8-sig").strip()
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1").strip()


def parse_uploaded_file(uploaded_file) -> str:
    """Read a Streamlit UploadedFile with PDF or TXT content."""
    name = Path(uploaded_file.name).name.lower()
    file_bytes = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    if name.endswith(".txt"):
        return extract_text_from_txt(file_bytes)
    raise ValueError(f"Unsupported file type: {name}. Upload a PDF or TXT file.")


def parse_raw_text(text: str) -> str:
    """Trim user-pasted text."""
    return text.strip() if text else ""


def basic_clean(text: str) -> str:
    """Collapse whitespace and remove control characters while retaining punctuation."""
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", " ", text or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()
