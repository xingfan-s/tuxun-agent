from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


EventType = Literal[
    "step_update",
    "progress",
    "tool_warning",
    "result",
    "error",
    "keepalive",
    "reasoning_summary",
]


class TaskEvent(BaseModel):
    """Replayable event envelope shared by the API and frontend."""

    id: int = Field(ge=1)
    task_id: str
    type: EventType
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data: dict[str, Any] = Field(default_factory=dict)
