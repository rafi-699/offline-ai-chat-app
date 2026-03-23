from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import Header, HTTPException, status


def verify_api_key(
    api_key: Optional[str],
    authorization: Optional[str] = Header(default=None),
) -> None:
    """Raises 401 if API key is configured and the request does not supply it."""
    if not api_key:
        return  # auth disabled
    if authorization is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or token != api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )


_SAFE_NAME = re.compile(r"[^a-zA-Z0-9._\-]")


def sanitize_filename(name: str) -> str:
    """Strip path separators and dangerous characters from a filename."""
    name = os.path.basename(name)
    name = _SAFE_NAME.sub("_", name)
    return name or "upload"


def safe_temp_file(suffix: str = "") -> str:
    """Create a secure temporary file and return its path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return path


def prevent_path_traversal(base_dir: Path, filename: str) -> Path:
    """Resolve filename relative to base_dir; raise if it escapes."""
    target = (base_dir / filename).resolve()
    if not str(target).startswith(str(base_dir.resolve())):
        raise ValueError(f"Path traversal detected: {filename}")
    return target
