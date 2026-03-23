from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import AsyncGenerator, Iterator, Optional

log = logging.getLogger(__name__)

_LOCK: Optional[asyncio.Lock] = None


def _get_lock() -> asyncio.Lock:
    global _LOCK
    if _LOCK is None:
        _LOCK = asyncio.Lock()
    return _LOCK


class LLMManager:
    """Lazy-loading, auto-unloading wrapper around llama_cpp.Llama."""

    def __init__(
        self,
        model_path: Path,
        n_ctx: int = 4096,
        n_threads: int = 3,
        idle_minutes: int = 15,
    ) -> None:
        self._model_path = model_path
        self._n_ctx = n_ctx
        self._n_threads = n_threads
        self._idle_minutes = idle_minutes
        self._llm = None
        self._last_used: float = 0.0
        self._idle_task: Optional[asyncio.Task] = None  # type: ignore[type-arg]

    # ── Load / Unload ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._llm is not None:
            return
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Model file missing: {self._model_path}\n"
                "Place the GGUF file in assets/models/ and re-run the installer."
            )
        log.info("Loading model from %s …", self._model_path)
        from llama_cpp import Llama  # type: ignore

        self._llm = Llama(
            model_path=str(self._model_path),
            n_ctx=self._n_ctx,
            n_threads=self._n_threads,
            verbose=False,
        )
        log.info("Model loaded.")

    def unload(self) -> None:
        if self._llm is not None:
            log.info("Unloading model.")
            del self._llm
            self._llm = None

    def is_loaded(self) -> bool:
        return self._llm is not None

    # ── Idle watchdog ─────────────────────────────────────────────────────────

    async def _idle_watchdog(self) -> None:
        if self._idle_minutes <= 0:
            return
        while True:
            await asyncio.sleep(60)
            if self._llm is None:
                continue
            idle_secs = time.monotonic() - self._last_used
            if idle_secs >= self._idle_minutes * 60:
                log.info("Model idle for %.0f s — unloading.", idle_secs)
                self.unload()
                return

    def _touch(self) -> None:
        self._last_used = time.monotonic()

    async def ensure_watchdog(self) -> None:
        if self._idle_minutes <= 0:
            return
        if self._idle_task is None or self._idle_task.done():
            self._idle_task = asyncio.create_task(self._idle_watchdog())

    # ── Generation ────────────────────────────────────────────────────────────

    def _build_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        history: list | None = None,
    ) -> list:
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        for msg in (history or []):
            msgs.append({"role": msg["role"], "content": msg["content"]})
        msgs.append({"role": "user", "content": user_prompt})
        return msgs


    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        history: list | None = None,
    ) -> str:

        async with _get_lock():
            self._load()
            self._touch()
            await self.ensure_watchdog()
            messages = self._build_messages(system_prompt, user_prompt, history)
            result = self._llm.create_chat_completion(  # type: ignore[union-attr]
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            return result["choices"][0]["message"]["content"]

    async def generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        history: list | None = None,
    ) -> AsyncGenerator[str, None]:

        """Yields text chunks as they are produced."""
        async with _get_lock():
            self._load()
            self._touch()
            await self.ensure_watchdog()
            messages = self._build_messages(system_prompt, user_prompt, history)

            def _iter() -> Iterator[str]:
                for chunk in self._llm.create_chat_completion(  # type: ignore[union-attr]
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                ):
                    delta = chunk["choices"][0].get("delta", {})
                    text = delta.get("content", "")
                    if text:
                        yield text

            loop = asyncio.get_event_loop()
            queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

            def _producer() -> None:
                try:
                    for token in _iter():
                        loop.call_soon_threadsafe(queue.put_nowait, token)
                except Exception as exc:
                    log.error("Stream error: %s", exc)
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            import threading
            t = threading.Thread(target=_producer, daemon=True)
            t.start()

            while True:
                token = await queue.get()
                if token is None:
                    break
                yield token


_manager: Optional[LLMManager] = None


def get_manager() -> LLMManager:
    if _manager is None:
        raise RuntimeError("LLMManager not initialised. Call init_manager() first.")
    return _manager


def init_manager(
    model_path: Path,
    n_ctx: int = 4096,
    n_threads: int = 3,
    idle_minutes: int = 15,
) -> LLMManager:
    global _manager
    _manager = LLMManager(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        idle_minutes=idle_minutes,
    )
    return _manager
