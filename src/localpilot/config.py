from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

def _find_env() -> str:
    import sys
    from pathlib import Path
    if getattr(sys, "frozen", False):
        # Running as PyInstaller bundle
        candidates = [
            Path(sys._MEIPASS) / ".env",
            Path(sys.executable).parent / ".env",
        ]
    else:
        candidates = [
            Path(__file__).resolve().parents[3] / ".env",
            Path(".env"),
        ]
    for p in candidates:
        if p.exists():
            return str(p)
    return ".env"  # fallback — pydantic_settings won't error if missing

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_find_env(),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Feature flags ──────────────────────────────────────────────────────────
    openai_compat_enabled: bool = True
    rag_enabled: bool = True

    # ── Server ─────────────────────────────────────────────────────────────────
    host: str = "127.0.0.1"
    port: int = 8765

    # ── Security ───────────────────────────────────────────────────────────────
    localpilot_api_key: Optional[str] = None
    cors_origins: str = ""  # comma-separated or blank

    # ── Model ──────────────────────────────────────────────────────────────────
    n_ctx: int = 4096
    n_threads: int = 0  # 0 = auto
    temperature: float = 0.7
    max_tokens: int = 2048
    model_idle_minutes: int = 15

    # ── Attachments ────────────────────────────────────────────────────────────
    max_upload_bytes: int = 20 * 1024 * 1024  # 20 MB

    # ── RAG ────────────────────────────────────────────────────────────────────
    rag_top_k: int = 4
    rag_chunk_size: int = 512
    rag_chunk_overlap: int = 64

    # ── Derived helpers ────────────────────────────────────────────────────────
    @property
    def cors_origins_list(self) -> List[str]:
        if not self.cors_origins.strip():
            return []
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def effective_n_threads(self) -> int:
        if self.n_threads and self.n_threads > 0:
            return self.n_threads
        return max(1, (os.cpu_count() or 2) - 1)


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
