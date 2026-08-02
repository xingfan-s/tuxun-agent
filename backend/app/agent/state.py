from typing import TypedDict, Callable
from langchain_core.messages import BaseMessage
from app.tools.base import ToolBudget


class AgentState(TypedDict):
    task_id: str
    image_path: str
    image_base64: str

    # Safety
    safety_passed: bool
    safety_reason: str | None

    # Stage outputs
    exif_data: dict | None
    vision_raw: str | None       # vision_detail output (方案三: detail pass)
    vision_region: str | None    # vision_macro output (方案三: macro pass)
    clues: dict | None           # structured clues with reliability weights (方案一)
    ocr_data: dict | None
    ocr_fused_queries: list[dict] | None  # OCR+vision context fused search queries (方案五)

    # Search strategy (方案四: knowledge base forward)
    search_strategy: dict | None

    # Multi-hypothesis tracking (方案一)
    hypotheses: list[dict]
    excluded_provinces: list[str]

    # ReAct loop
    messages: list[BaseMessage]
    tool_calls: list[dict]
    loop_count: int
    failed_tools: set[str]
    last_redirect_at: int
    tool_budget: ToolBudget

    # Result
    result: dict | None
    error: str | None

    # Adversarial verification (方案二)
    verification_passed: bool | None
    verification_feedback: str | None
    verification_history: list[str]

    # GeoCLIP
    geoclip_result: dict | None
    geoclip_anchors: list[dict] | None  # pre-search results around GeoCLIP coords

    # CLIP + FAISS image similarity search
    clip_result: dict | None

    # SSE callback
    stream_callback: Callable | None

    # Approximate usage accounting used when providers do not expose usage
    # metadata in streaming responses.
    tokens_used: int
    model_calls: int
    model_usage: dict[str, dict]
