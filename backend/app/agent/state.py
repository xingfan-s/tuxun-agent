from typing import TypedDict, Callable
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    task_id: str
    image_path: str
    image_base64: str

    # Safety
    safety_passed: bool
    safety_reason: str | None

    # Stage outputs
    exif_data: dict | None
    vision_raw: str | None
    clues: dict | None

    # ReAct loop
    messages: list[BaseMessage]
    tool_calls: list[dict]
    loop_count: int
    failed_tools: set[str]
    last_redirect_at: int

    # Result
    result: dict | None
    error: str | None

    # SSE callback
    stream_callback: Callable | None
