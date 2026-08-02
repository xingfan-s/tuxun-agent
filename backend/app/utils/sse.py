import asyncio
import json
import time
from app.utils.logging import structlog
from enum import StrEnum
from typing import AsyncGenerator

logger = structlog.get_logger()


class SSEEvent(StrEnum):
    STEP_UPDATE = "step_update"
    PROGRESS = "progress"
    TOOL_WARNING = "tool_warning"
    RESULT = "result"
    ERROR = "error"
    KEEPALIVE = "keepalive"
    REASONING_SUMMARY = "reasoning_summary"

HEARTBEAT_INTERVAL = 15  # seconds


def format_sse(event: SSEEvent, data: dict, retry: int = 3000, event_id: int | None = None) -> str:
    """Format a Server-Sent Event message."""
    lines = ([f"id: {event_id}"] if event_id is not None else []) + [
        f"event: {event.value}",
        f"data: {json.dumps(data, ensure_ascii=False)}",
        f"retry: {retry}",
        "",
        "",
    ]
    return "\n".join(lines)


async def sse_event_generator(event_queue, on_close=None) -> AsyncGenerator[str, None]:
    """Async generator yielding SSE-formatted events from a queue.

    Sends periodic keepalive heartbeats to prevent browser/proxy timeout
    during long-running operations (LLM calls, tool execution).
    """
    try:
        while True:
            try:
                msg = await asyncio.wait_for(event_queue.get(), timeout=HEARTBEAT_INTERVAL)
            except asyncio.TimeoutError:
                # Send keepalive as named event (more proxy-friendly than comments)
                yield format_sse(SSEEvent.KEEPALIVE, {"ts": int(time.time())})
                continue

            if msg is None:  # Sentinel to stop
                break
            event_type = msg.get("type", msg.get("event", "step_update"))
            data = msg.get("data", {})
            yield format_sse(SSEEvent(event_type), data, event_id=msg.get("id"))
            event_queue.task_done()
    except Exception as e:
        logger.error("sse_generator_error", error=str(e))
        yield format_sse(SSEEvent.ERROR, {
            "message": f"SSE stream error: {str(e)[:200]}",
            "step": -1, "recoverable": False,
        })
    finally:
        if on_close:
            on_close()
