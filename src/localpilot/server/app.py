from __future__ import annotations

"""FastAPI application factory."""

import json
import logging
import os
import time
from pathlib import Path
from typing import AsyncGenerator, Optional

from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from localpilot.attachments.extract import extract_file
from localpilot.config import Settings, get_settings
from localpilot.llm.manager import LLMManager, get_manager, init_manager
from localpilot.rag.index import RAGIndex
from localpilot.rag.ingest import ingest_text
from localpilot.utils.paths import ensure_model_in_data_dir, get_rag_dir
from localpilot.utils.security import sanitize_filename, safe_temp_file, verify_api_key

log = logging.getLogger(__name__)

# ── Global RAG index (created lazily) ─────────────────────────────────────────
_rag_index: Optional[RAGIndex] = None


def _get_rag_index() -> RAGIndex:
    global _rag_index
    if _rag_index is None:
        _rag_index = RAGIndex(get_rag_dir())
    return _rag_index


# ── Pydantic models ───────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    system_prompt: str = ""
    user_prompt: str
    history: list = []
    temperature: float = 0.7
    max_tokens: int = 512
    stream: bool = False


class RAGQueryRequest(BaseModel):
    system_prompt: str = ""
    user_prompt: str
    temperature: float = 0.7
    max_tokens: int = 512
    top_k: int = 4


# ── OpenAI-compat Pydantic models ─────────────────────────────────────────────

class OAIMessage(BaseModel):
    role: str
    content: str


class OAIChatRequest(BaseModel):
    model: str = "local"
    messages: list[OAIMessage]
    temperature: float = 0.7
    max_tokens: int = 512
    stream: bool = False


# ── Auth dependency ───────────────────────────────────────────────────────────

def make_auth_dep(settings: Settings):
    def _check(request: Request) -> None:
        auth_header = request.headers.get("Authorization")
        verify_api_key(settings.localpilot_api_key, auth_header)
    return _check


# ── App factory ───────────────────────────────────────────────────────────────

def create_app(settings: Optional[Settings] = None) -> FastAPI:
    if settings is None:
        settings = get_settings()

    app = FastAPI(
        title="RafiGPT API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    cors_origins = settings.cors_origins_list
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    auth_dep = make_auth_dep(settings)

    # ── Initialise LLM manager ─────────────────────────────────────────────────
    try:
        model_path = ensure_model_in_data_dir()
    except FileNotFoundError as exc:
        log.error("Model missing: %s", exc)
        model_path = None

    if model_path:
        init_manager(
            model_path=model_path,
            n_ctx=settings.n_ctx,
            n_threads=settings.effective_n_threads,
            idle_minutes=settings.model_idle_minutes,
        )

    # ── Routes ─────────────────────────────────────────────────────────────────

    @app.get("/api/health")
    async def health(_: None = Depends(auth_dep)):
        return {"status": "ok", "timestamp": time.time()}

    @app.get("/api/status")
    async def api_status(_: None = Depends(auth_dep)):
        mgr_loaded = False
        if model_path:
            try:
                mgr_loaded = get_manager().is_loaded()
            except RuntimeError:
                pass
        rag_stats = {}
        if settings.rag_enabled:
            rag_stats = _get_rag_index().stats()
        return {
            "model_loaded": mgr_loaded,
            "model_path": str(model_path) if model_path else None,
            "rag_enabled": settings.rag_enabled,
            "rag_stats": rag_stats,
            "openai_compat_enabled": settings.openai_compat_enabled,
        }

    @app.post("/api/generate")
    async def generate(body: GenerateRequest, _: None = Depends(auth_dep)):
        if model_path is None:
            raise HTTPException(status_code=503, detail="Model file missing. See server logs.")
        mgr = get_manager()
        if body.stream:
            return StreamingResponse(
                _sse_generate(mgr, body.system_prompt, body.user_prompt, body.temperature, body.max_tokens, body.history),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        text = await mgr.generate(body.system_prompt, body.user_prompt, body.temperature, body.max_tokens, body.history)
        return {"text": text}

    @app.post("/api/attachments/extract")
    async def extract_attachment(
        file: UploadFile = File(...),
        _: None = Depends(auth_dep),
    ):
        if file.size and file.size > settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail=f"File too large. Max {settings.max_upload_bytes} bytes.")
        safe_name = sanitize_filename(file.filename or "upload")
        suffix = Path(safe_name).suffix
        tmp_path = safe_temp_file(suffix=suffix)
        try:
            content = await file.read()
            if len(content) > settings.max_upload_bytes:
                raise HTTPException(status_code=413, detail="File too large.")
            with open(tmp_path, "wb") as fh:
                fh.write(content)
            result = extract_file(Path(tmp_path))
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        return result

    @app.post("/api/model/unload")
    async def unload_model(_: None = Depends(auth_dep)):
        try:
            get_manager().unload()
            return {"status": "unloaded"}
        except RuntimeError:
            return {"status": "not_loaded"}

    @app.post("/api/rag/ingest")
    async def rag_ingest(
        file: UploadFile = File(...),
        _: None = Depends(auth_dep),
    ):
        if not settings.rag_enabled:
            raise HTTPException(status_code=403, detail="RAG is disabled.")
        if file.size and file.size > settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail="File too large.")
        safe_name = sanitize_filename(file.filename or "upload")
        suffix = Path(safe_name).suffix
        tmp_path = safe_temp_file(suffix=suffix)
        try:
            content = await file.read()
            if len(content) > settings.max_upload_bytes:
                raise HTTPException(status_code=413, detail="File too large.")
            with open(tmp_path, "wb") as fh:
                fh.write(content)
            extraction = extract_file(Path(tmp_path))
            text = extraction.get("text", "")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        if not text.strip():
            raise HTTPException(status_code=422, detail="No text could be extracted from file.")
        idx = _get_rag_index()
        n_chunks = ingest_text(
            text,
            doc_name=safe_name,
            index=idx,
            chunk_size=settings.rag_chunk_size,
            overlap=settings.rag_chunk_overlap,
        )
        return {"doc_name": safe_name, "chunks_added": n_chunks}

    @app.post("/api/rag/query")
    async def rag_query(body: RAGQueryRequest, _: None = Depends(auth_dep)):
        if not settings.rag_enabled:
            if model_path is None:
                raise HTTPException(status_code=503, detail="Model file missing.")
            mgr = get_manager()
            text = await mgr.generate(body.system_prompt, body.user_prompt, body.temperature, body.max_tokens)
            return {"text": text, "rag_used": False, "context_chunks": []}

        idx = _get_rag_index()
        stats = idx.stats()
        if stats["total_chunks"] == 0:
            if model_path is None:
                raise HTTPException(status_code=503, detail="Model file missing.")
            mgr = get_manager()
            text = await mgr.generate(body.system_prompt, body.user_prompt, body.temperature, body.max_tokens)
            return {"text": text, "rag_used": False, "context_chunks": [], "note": "No documents in knowledge base."}

        hits = idx.query(body.user_prompt, top_k=body.top_k)
        if not hits:
            if model_path is None:
                raise HTTPException(status_code=503, detail="Model file missing.")
            mgr = get_manager()
            text = await mgr.generate(body.system_prompt, body.user_prompt, body.temperature, body.max_tokens)
            return {"text": text, "rag_used": False, "context_chunks": [], "note": "No relevant chunks found."}

        context_chunks = [{"source": h[0], "text": h[1], "score": h[2]} for h in hits]
        context_text = "\n\n".join(f"[Source: {h[0]}]\n{h[1]}" for h in hits)
        augmented_system = (
            (body.system_prompt + "\n\n" if body.system_prompt else "")
            + "Use the following context to answer the user:\n\n"
            + context_text
        )
        mgr = get_manager()
        text = await mgr.generate(augmented_system, body.user_prompt, body.temperature, body.max_tokens)
        return {"text": text, "rag_used": True, "context_chunks": context_chunks}

    # ── OpenAI-compat (gated) ──────────────────────────────────────────────────

    if settings.openai_compat_enabled:

        @app.get("/v1/models")
        async def oai_list_models():
            return {
                "object": "list",
                "data": [
                    {
                        "id": "local",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "localpilot",
                    }
                ],
            }

        @app.post("/v1/chat/completions")
        async def oai_chat_completions(body: OAIChatRequest, _: None = Depends(auth_dep)):
            if model_path is None:
                raise HTTPException(status_code=503, detail="Model file missing.")

            # Extract system and build history
            system_prompt = ""
            history = []
            user_prompt = ""
            for msg in body.messages:
                if msg.role == "system":
                    system_prompt = msg.content
                elif msg.role == "user":
                    history.append({"role": "user", "content": msg.content})
                    user_prompt = msg.content
                elif msg.role == "assistant":
                    history.append({"role": "assistant", "content": msg.content})

            # Remove last user message from history (it's the current prompt)
            if history and history[-1]["role"] == "user":
                history = history[:-1]

            mgr = get_manager()
            created = int(time.time())

            if body.stream:
                async def _oai_stream():
                    chunk_id = f"chatcmpl-{created}"
                    async for token in mgr.generate_stream(system_prompt, user_prompt, body.temperature, body.max_tokens, history):
                        chunk = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": body.model,
                            "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    done_chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": body.model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                    yield f"data: {json.dumps(done_chunk)}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(
                    _oai_stream(),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )

            # Non-streaming
            text = await mgr.generate(system_prompt, user_prompt, body.temperature, body.max_tokens, history)
            return {
                "id": f"chatcmpl-{created}",
                "object": "chat.completion",
                "created": created,
                "model": body.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }

    else:
        @app.api_route(
            "/v1/{path:path}",
            methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
            include_in_schema=False,
        )
        async def oai_disabled(path: str):
            raise HTTPException(
                status_code=404,
                detail="OpenAI-compatible API is disabled. Set OPENAI_COMPAT_ENABLED=true in .env",
            )

    return app


# ── SSE helper ────────────────────────────────────────────────────────────────

async def _sse_generate(
    mgr: LLMManager,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    history: list | None = None,
) -> AsyncGenerator[str, None]:
    async for token in mgr.generate_stream(system_prompt, user_prompt, temperature, max_tokens, history):
        payload = json.dumps({"token": token})
        yield f"data: {payload}\n\n"
    yield "data: [DONE]\n\n"