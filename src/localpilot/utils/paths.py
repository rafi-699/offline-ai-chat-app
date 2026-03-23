from __future__ import annotations

import os
import sys
from pathlib import Path

MODEL_URL = "https://smrafi.com/utilities/llms/llm.gguf"


def get_app_data_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "RafiGPT"


def get_models_dir() -> Path:
    return get_app_data_dir() / "models"


def get_rag_dir() -> Path:
    return get_app_data_dir() / "rag"


def get_log_path() -> Path:
    return get_app_data_dir() / "logs" / "rafigpt.log"


def get_bundled_models_dir() -> Path:
    if getattr(sys, "frozen", False):
        base = Path(sys._MEIPASS)  # type: ignore[attr-defined]
    else:
        base = Path(__file__).resolve().parents[3]
    return base / "assets" / "models"


def get_model_filename() -> str:
    return "llm.gguf"


def ensure_model_in_data_dir() -> Path:
    """
    1. Check user data dir — return if found.
    2. Check bundled assets/models/ — copy if found.
    3. Download from smrafi.com with progress callback support.
    Raises FileNotFoundError only if download also fails.
    """
    import shutil

    dest_dir = get_models_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / get_model_filename()

    if dest.exists():
        return dest

    # Try bundled copy first (dev mode or installer with model included)
    src = get_bundled_models_dir() / get_model_filename()
    if src.exists():
        shutil.copy2(src, dest)
        return dest

    # Model not found locally — caller should trigger download UI
    raise FileNotFoundError("MODEL_NOT_FOUND")


def download_model(progress_callback=None) -> Path:
    """
    Download llm.gguf from MODEL_URL into the user data models dir.
    progress_callback(downloaded_bytes, total_bytes) is called during download.
    Returns the final path on success.
    Raises RuntimeError on network/HTTP failure.
    """
    import urllib.request

    dest_dir = get_models_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / get_model_filename()
    tmp = dest.with_suffix(".tmp")

    try:
        req = urllib.request.Request(
            MODEL_URL,
            headers={"User-Agent": "RafiGPT/1.0"},
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 64  # 64KB chunks

            with open(tmp, "wb") as fh:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback:
                        progress_callback(downloaded, total)

        tmp.rename(dest)
        return dest

    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"Download failed: {exc}") from exc
