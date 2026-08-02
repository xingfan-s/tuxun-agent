from datetime import datetime, timezone
from typing import Any, Literal
from pydantic import BaseModel, Field
from app.schemas.evidence import Evidence


class ToolStats(BaseModel):
    total_calls: int = 0
    success: int = 0
    timeout: int = 0
    failed: int = 0
    unavailable: int = 0
    budget_skipped: int = 0
    invalid_input: int = 0
    upstream_error: int = 0
    empty_result: int = 0


class Result(BaseModel):
    address: str = ""
    country: str = ""
    province: str | None = None
    city: str | None = None
    district: str | None = None
    lat: float | None = Field(default=None, ge=-90, le=90)
    lng: float | None = Field(default=None, ge=-180, le=180)
    coord_system: Literal["WGS84", "GCJ-02", "BD-09", "unknown"] = "WGS84"
    precision_level: Literal["country", "province", "city", "district", "road", "poi", "unknown"] = "unknown"
    uncertainty_radius_m: float | None = Field(default=None, ge=0)
    confidence: float | None = Field(default=None, ge=0, le=1)
    confidence_kind: Literal["calibrated", "ranking_score", "unknown"] = "ranking_score"
    reasoning: str = ""
    evidence: list[Evidence] = Field(default_factory=list)
    top_hypotheses: list[dict[str, Any]] = Field(default_factory=list)
    tokens_used: int = 0
    model_calls: int = 0
    model_usage: dict[str, dict[str, int]] = Field(default_factory=dict)
    estimated_cost: float = 0.0
    total_elapsed_ms: int = 0
    tool_stats: ToolStats | None = None


StepType = Literal[
    "safety_check", "exif", "vision", "vision_macro", "vision_detail",
    "clue_extraction", "ocr", "ocr_fusion", "geoclip", "geoclip_anchor",
    "clip_search", "anchor_search", "search_strategy", "tool_call",
    "reasoning", "verification", "final", "fine_localize", "result_enrichment"
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
    data: dict[str, Any] = Field(default_factory=dict)
    elapsed_ms: int = 0


TaskStatusType = Literal[
    "uploaded", "queued", "analyzing", "done", "failed", "rejected", "cancelled", "expired"
]


class TaskStatus(BaseModel):
    task_id: str
    status: TaskStatusType
    progress: int = Field(default=0, ge=0, le=100)
    steps: list[StepResult] = Field(default_factory=list)
    result: Result | None = None
    error: str | None = None
    error_recoverable: bool | None = None
    safety_reason: str | None = None
    cancel_requested: bool = False
    last_event_id: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class UploadResponse(BaseModel):
    task_id: str
    status: TaskStatusType
    created_at: datetime


class ErrorResponse(BaseModel):
    error: str
    reason: str | None = None
