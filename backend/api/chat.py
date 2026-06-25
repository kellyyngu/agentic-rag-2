import json
import time
from typing import List, Optional
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from loguru import logger

from agent.graph import run_agent
from agent.citation_manager import CitationManager

router = APIRouter()


class Message(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    query: str
    conversation_history: List[Message] = []
    session_id: Optional[str] = None
    debug: bool = False


async def _event_stream(query: str, history: list, retriever, citation_manager):
    """Generate SSE events from the agent graph."""
    start = time.time()

    try:
        async for event in run_agent(query, history, retriever, citation_manager):
            payload = json.dumps(event["data"])
            yield f"event: {event['event']}\ndata: {payload}\n\n"
    except Exception as e:
        logger.error(f"[chat] stream error: {e}")
        yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
    finally:
        elapsed = time.time() - start
        yield f"event: done\ndata: {json.dumps({'latency_s': round(elapsed, 2)})}\n\n"


@router.post("/chat")
async def chat(request: Request, body: ChatRequest):
    retriever = request.app.state.retriever
    if not retriever:
        raise HTTPException(503, "Retriever not initialized")

    history = [m.model_dump() for m in body.conversation_history]

    # Per-session citation manager: reuse one per session_id so the same chunk keeps
    # a stable citation ID across turns; with no session_id, isolate per request.
    # No cross-user shared state.
    managers = getattr(request.app.state, "citation_managers", None)
    if body.session_id and managers is not None:
        citation_manager = managers.get(body.session_id)
        if citation_manager is None:
            citation_manager = CitationManager()
            managers[body.session_id] = citation_manager
    else:
        citation_manager = CitationManager()

    return StreamingResponse(
        _event_stream(body.query, history, retriever, citation_manager),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/chat/health")
async def chat_health():
    return {"status": "ok"}
