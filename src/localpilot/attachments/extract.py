from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Tuple

log = logging.getLogger(__name__)


def _get_tesseract_cmd() -> str | None:
    """
    Resolve tesseract binary path in this order:
    1. Bundled vendor/tesseract/ — walk upward from this file
    2. PyInstaller bundle (sys._MEIPASS)
    3. System PATH fallback
    """
    # PyInstaller packaged
    if getattr(sys, "frozen", False):
        candidate = Path(sys._MEIPASS) / "vendor" / "tesseract" / "tesseract.exe"  # type: ignore[attr-defined]
        if candidate.exists():
            return str(candidate)

    # Dev mode — walk UP directory tree looking for vendor/tesseract/
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "vendor" / "tesseract" / "tesseract.exe"
        if candidate.exists():
            log.debug("Found bundled Tesseract at: %s", candidate)
            return str(candidate)

    # System PATH fallback
    return None



def _configure_tesseract() -> bool:
    """
    Configure pytesseract to use bundled or system Tesseract.
    Returns True if tesseract is available, False if not found at all.
    """
    try:
        import pytesseract  # type: ignore

        cmd = _get_tesseract_cmd()
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd
            log.debug("Tesseract binary: %s", cmd)
        else:
            log.debug("Tesseract not bundled — relying on system PATH")

        # Quick sanity check — will raise if not found
        pytesseract.get_tesseract_version()
        return True

    except ImportError:
        log.warning("pytesseract not installed")
        return False
    except Exception as exc:
        log.warning("Tesseract not available: %s", exc)
        return False


# Run once at module import
_TESSERACT_AVAILABLE: bool = _configure_tesseract()


def extract_pdf(path: Path) -> str:
    """Extract text from a PDF using PyMuPDF."""
    import fitz  # PyMuPDF

    text_parts: list[str] = []
    with fitz.open(str(path)) as doc:
        for page in doc:
            page_text = page.get_text()
            if page_text.strip():
                text_parts.append(page_text)
            elif _TESSERACT_AVAILABLE:
                # Scanned page — render to image and OCR
                pix = page.get_pixmap(dpi=200)
                img_path = Path(path).parent / f"_ocr_page_{page.number}.png"
                try:
                    pix.save(str(img_path))
                    ocr_text, _ = extract_image_ocr(img_path)
                    if ocr_text.strip():
                        text_parts.append(ocr_text)
                finally:
                    if img_path.exists():
                        img_path.unlink()

    return "\n".join(text_parts)


def extract_image_ocr(path: Path) -> Tuple[str, bool]:
    """
    Attempt OCR on an image using pytesseract + Pillow.
    Returns (text, ocr_available).
    """
    if not _TESSERACT_AVAILABLE:
        return "", False

    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore

        img = Image.open(str(path))
        text = pytesseract.image_to_string(img)
        return text.strip(), True

    except Exception as exc:
        log.error("OCR failed: %s", exc)
        return "", True  # installed but failed on this specific image


def extract_file(path: Path) -> dict:
    """
    Dispatch extraction based on file suffix.
    Returns dict with keys: text, ocr_available, message.
    """
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        text = extract_pdf(path)
        return {
            "text": text,
            "ocr_available": _TESSERACT_AVAILABLE,
            "message": "PDF extracted" + (" (with OCR fallback for scanned pages)" if _TESSERACT_AVAILABLE else " (text-only; install Tesseract for scanned pages)"),
        }

    elif suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}:
        if not _TESSERACT_AVAILABLE:
            return {
                "text": "",
                "ocr_available": False,
                "message": (
                    "OCR unavailable. Tesseract binaries not found.\n"
                    "Either place Tesseract in vendor/tesseract/ folder "
                    "or install Tesseract-OCR system-wide."
                ),
            }
        text, available = extract_image_ocr(path)
        return {
            "text": text,
            "ocr_available": True,
            "message": "Image OCR complete" if text else "OCR ran but extracted no text",
        }

    elif suffix in {".txt", ".md", ".csv", ".log"}:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            return {"text": text, "ocr_available": None, "message": "Plain text read"}
        except Exception as exc:
            return {"text": "", "ocr_available": None, "message": f"Read error: {exc}"}

    else:
        return {
            "text": "",
            "ocr_available": None,
            "message": f"Unsupported file type: {suffix}",
        }
