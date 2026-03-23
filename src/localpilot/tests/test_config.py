import os
import pytest
from localpilot.config import Settings


def test_defaults():
    s = Settings()
    assert s.host == "127.0.0.1"
    assert s.port == 8765
    assert s.openai_compat_enabled is False
    assert s.rag_enabled is False
    assert s.n_ctx == 4096


def test_env_override(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPAT_ENABLED", "true")
    monkeypatch.setenv("PORT", "9000")
    s = Settings()
    assert s.openai_compat_enabled is True
    assert s.port == 9000


def test_cors_list_empty():
    s = Settings()
    assert s.cors_origins_list == []


def test_cors_list_parsed(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:3000, http://localhost:5173")
    s = Settings()
    assert "http://localhost:3000" in s.cors_origins_list
    assert len(s.cors_origins_list) == 2


def test_effective_n_threads():
    import os
    s = Settings()
    expected = max(1, (os.cpu_count() or 2) - 1)
    assert s.effective_n_threads == expected
