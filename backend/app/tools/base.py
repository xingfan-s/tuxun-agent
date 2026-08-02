"""Base classes and utilities for tools.

ToolResult provides a standardized container for tool execution results.
"""

from dataclasses import dataclass
from time import monotonic
from typing import Any, Literal


ToolStatus = Literal[
    "success", "unavailable", "timeout", "invalid_input", "upstream_error",
    "empty_result", "failed", "skipped",
]


@dataclass
class ToolResult:
    """Standardized tool result container."""
    tool_name: str
    status: ToolStatus
    data: Any = None
    error: str | None = None
    duration_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "status": self.status,
            "data": self.data,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


@dataclass
class ToolBudget:
    """Per-task deterministic call and elapsed-time budget."""

    max_calls: int
    max_elapsed_seconds: float
    started_at: float = 0.0
    calls: int = 0

    def __post_init__(self) -> None:
        if not self.started_at:
            self.started_at = monotonic()

    @property
    def elapsed_seconds(self) -> float:
        return monotonic() - self.started_at

    def consume(self) -> bool:
        if self.calls >= self.max_calls or self.elapsed_seconds >= self.max_elapsed_seconds:
            return False
        self.calls += 1
        return True
