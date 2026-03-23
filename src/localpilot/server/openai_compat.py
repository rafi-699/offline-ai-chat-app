from __future__ import annotations

"""
OpenAI-compatible API endpoints.
Gated by settings.openai_compat_enabled — router is only mounted when True.
"""

import json
import logging
import time
from typing import AsyncGenerator, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from localpilot.config import get_settings
from localpilot.llm.manager import get_manager
from localpilot.utils.security import verify_api_key

log = logging.getLogger(__name__)
router = APIRouter(tags=["openai-compat"])

# ── Pydantic schemas ──────────────────────────────────────────────────────────


class OAIMessage(BaseModel):
    role: str
    content: str


class OAIChatRequest(BaseModel):
    model: str = "localpilot"
    messages: List[OAIMessage]
    temperature: float = 0.7
    max_tokens: int = 512
    stream: bool = False


# ── Auth dependency ───────────────────────────────────────────────────────────


def _auth(request: Request) -> None:
    settings = get_settings()
    auth_header = request.headers.get("Authorization")
    verify_api_key(settings.localpilot_api_key, auth_header)


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get("/v1/models")
async def list_models(request: Request, _: None = Depends(_auth)):
    return {
        "object": "list",
        "data": [
            {
                "id": "localpilot",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "localpilot",
            }
        ],
    }


@router.post("/v1/chat/completions")
async def chat_completions(
    body: OAIChatRequest,
    request: Request,
    _: None = Depends(_auth),
):
    mgr = get_manager()
    messages = body.messages

    # Build system + user from the message list
    system_prompt = ""
    user_parts: list[str] = []
    for msg in messages:
        if msg.role == "system":
            system_prompt = msg.content
        else:
            user_parts.append(f"{msg.role}: {msg.content}")
    user_prompt = "\n".join(user_parts)

    if body.stream:
        return StreamingResponse(
            _stream_sse(mgr, system_prompt, user_prompt, body.temperature, body.max_tokens),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Non-streaming
    text = await mgr.generate(system_prompt, user_prompt, body.temperature, body.max_tokens)
    completion_id = f"chatcmpl-{int(time.time())}"
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": -1,
            "completion_tokens": -1,
            "total_tokens": -1,
        },
    }


async def _stream_sse(
    mgr,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> AsyncGenerator[str, None]:
    completion_id = f"chatcmpl-{int(time.time())}"
    async for token in mgr.generate_stream(system_prompt, user_prompt, temperature, max_tokens):
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "localpilot",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": token},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
    # Final stop chunk
    stop_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "localpilot",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(stop_chunk)}\n\n"
    yield "data: [DONE]\n\n"
