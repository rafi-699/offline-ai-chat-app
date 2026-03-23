"""
Tests for the LLM manager using a mocked Llama class.
This keeps tests fast without requiring a real GGUF model.
"""
import asyncio
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


def _make_mock_llama(text="Hello from mock!"):
    mock_llm = MagicMock()
    mock_llm.create_chat_completion.return_value = {
        "choices": [{"message": {"role": "assistant", "content": text}}]
    }
    return mock_llm


@pytest.fixture
def manager(tmp_path):
    # Create a dummy model file so existence check passes
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"dummy")

    from localpilot.llm.manager import LLMManager

    mgr = LLMManager(
        model_path=model_file,
        n_ctx=512,
        n_threads=1,
        idle_minutes=0,
    )
    return mgr


@pytest.mark.asyncio
async def test_generate_mocked(manager):
    mock_llama = _make_mock_llama("Hello!")
    with patch("llama_cpp.Llama", return_value=mock_llama):
        # Force _load to think Llama is importable
        manager._llm = mock_llama
        result = await manager.generate(
            system_prompt="Be helpful.",
            user_prompt="Say hello.",
            temperature=0.5,
            max_tokens=32,
        )
    assert result == "Hello!"


@pytest.mark.asyncio
async def test_generate_stream_mocked(manager):
    tokens = ["Hello", " world", "!"]

    def _fake_chat_completion(**kwargs):
        for t in tokens:
            yield {"choices": [{"delta": {"content": t}}]}

    mock_llama = MagicMock()
    mock_llama.create_chat_completion.side_effect = _fake_chat_completion

    manager._llm = mock_llama
    collected = []
    async for tok in manager.generate_stream("sys", "user", 0.7, 32):
        collected.append(tok)
    assert "".join(collected) == "Hello world!"


def test_model_missing_raises(tmp_path):
    from localpilot.llm.manager import LLMManager

    mgr = LLMManager(model_path=tmp_path / "nonexistent.gguf", n_ctx=512, n_threads=1)
    with pytest.raises(FileNotFoundError):
        mgr._load()
