from datetime import datetime, timezone
from typing import Literal
from pydantic import BaseModel, Field


class ToolStats(BaseModel):
    total_calls: int = 0
    success: int = 0
    timeout: int = 0
    failed: int = 0


class Result(BaseModel):
    address: str
    country: str
    province: str | None = None
    city: str | None = None
    district: str | None = None
    lat: float
    lng: float
    confidence: float = Field(ge=0, le=1)
    reasoning: str
    tokens_used: int = 0
    total_elapsed_ms: int = 0
    tool_stats: ToolStats | None = None


StepType = Literal[
    "safety_check", "exif", "vision", "clue_extraction",
    "tool_call", "reasoning", "final"
]


class StepResult(BaseModel):
    """
    data contract by type:

    safety_check:    {"passed": bool, "face_count": int, "scene": str, "sensitive_text": bool}
    exif:            {"gps": {"lat": float, "lng": float} | None, "datetime": str | None}
    vision:          {"description": str, "raw_output": str}
    clue_extraction: {"clues": dict}
    tool_call:       {"tool_name": str, "input": dict, "output": dict | None,
                      "status": "success"|"timeout"|"failed", "duration_ms": int}
    reasoning:       {"thought": str, "action": str, "action_input": dict | None}
    final:           {}
    """
    step: int
    type: StepType
    label: str
    status: Literal["running", "done", "error"]
    data: dict = Field(default_factory=dict)
    elapsed_ms: int = 0


TaskStatusType = Literal["uploaded", "analyzing", "done", "failed", "rejected"]


class TaskStatus(BaseModel):
    task_id: str
    status: TaskStatusType
    progress: int = Field(default=0, ge=0, le=100)
    steps: list[StepResult] = Field(default_factory=list)
    result: Result | None = None
    error: str | None = None
    error_recoverable: bool | None = None
    safety_reason: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class UploadResponse(BaseModel):
    task_id: str
    status: TaskStatusType
    created_at: datetime


class ErrorResponse(BaseModel):
    error: str
    reason: str | None = None
