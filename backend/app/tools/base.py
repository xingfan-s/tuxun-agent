import asyncio
import functools
import structlog
from typing import Callable, Any

logger = structlog.get_logger()

TOOL_TIMEOUTS = {
    "search_place": 8,
    "geocode": 5,
    "reverse_geocode": 5,
    "search_nearby": 8,
    "get_streetview": 15,
    "extract_exif": 3,
    "search_landmark": 10,
    "reverse_image_search": 12,
}


def with_timeout(tool_name: str):
    """Decorator that wraps a sync tool function with an async timeout."""
    timeout = TOOL_TIMEOUTS.get(tool_name, 10)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> dict:
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(func, *args, **kwargs),
                    timeout=timeout,
                )
                return {"status": "success", "data": result, "tool": tool_name}
            except asyncio.TimeoutError:
                logger.warning("tool_timeout", tool=tool_name, timeout=timeout)
                return {"status": "timeout", "data": None, "tool": tool_name,
                        "error": f"Tool {tool_name} timed out after {timeout}s"}
            except Exception as e:
                logger.error("tool_error", tool=tool_name, error=str(e))
                return {"status": "failed", "data": None, "tool": tool_name,
                        "error": str(e)}
        return wrapper
    return decorator


class ToolResult:
    """Standardized tool result container."""
    def __init__(self, tool_name: str, status: str, data: Any = None,
                 error: str = None, duration_ms: int = 0):
        self.tool_name = tool_name
        self.status = status
        self.data = data
        self.error = error
        self.duration_ms = duration_ms

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "status": self.status,
            "data": self.data,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }
