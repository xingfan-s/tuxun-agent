import json
from enum import StrEnum
from typing import AsyncGenerator


class SSEEvent(StrEnum):
    STEP_UPDATE = "step_update"
    PROGRESS = "progress"
    TOOL_WARNING = "tool_warning"
    RESULT = "result"
    ERROR = "error"


def format_sse(event: SSEEvent, data: dict, retry: int = 3000) -> str:
    """Format a Server-Sent Event message."""
    lines = [
        f"event: {event.value}",
        f"data: {json.dumps(data, ensure_ascii=False)}",
        f"retry: {retry}",
        "",
        "",
    ]
    return "\n".join(lines)


async def sse_event_generator(event_queue) -> AsyncGenerator[str, None]:
    """Async generator yielding SSE-formatted events from a queue."""
    while True:
        msg = await event_queue.get()
        if msg is None:  # Sentinel to stop
            break
        event_type = msg.get("event", "step_update")
        data = msg.get("data", {})
        yield format_sse(SSEEvent(event_type), data)
        event_queue.task_done()
