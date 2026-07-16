import asyncio
import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse
from app.schemas.task import TaskStatus, UploadResponse
from app.services.agent_service import task_store, AgentService
from app.utils.image import validate_image, save_upload, delete_image

router = APIRouter()

_agent_service: AgentService | None = None


def get_agent_service() -> AgentService:
    global _agent_service
    if _agent_service is None:
        _agent_service = AgentService()
    return _agent_service


@router.on_event("startup")
async def startup_agent():
    agent = get_agent_service()
    await agent.start()


@router.on_event("shutdown")
async def shutdown_agent():
    if _agent_service:
        await _agent_service.stop()


@router.post("/upload", response_model=UploadResponse, status_code=201)
async def upload_image(file: UploadFile = File(...)):
    """Upload an image for geo-location analysis."""
    content = await file.read()

    valid, error_msg = validate_image(file.content_type or "", len(content))
    if not valid:
        raise HTTPException(status_code=422, detail=error_msg)

    file_path, _ = save_upload(content, file.filename or "image.jpg")
    task_id = uuid.uuid4().hex[:12]

    task_store.create(task_id, file_path)

    return UploadResponse(
        task_id=task_id,
        status="uploaded",
        created_at=task_store.get(task_id).created_at,
    )


@router.get("/task/{task_id}", response_model=TaskStatus)
async def get_task(task_id: str):
    """Get task status and result."""
    task = task_store.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return task


@router.get("/task/{task_id}/stream")
async def stream_task(task_id: str):
    """SSE stream for real-time task progress."""
    task = task_store.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")

    event_queue: asyncio.Queue = asyncio.Queue()

    if task.status in ("done", "failed", "rejected"):
        if task.status == "done" and task.result:
            await event_queue.put({
                "event": "result", "data": task.result.model_dump(),
            })
        elif task.status == "rejected":
            await event_queue.put({
                "event": "error",
                "data": {"message": task.safety_reason or "安全预检未通过", "step": 0, "recoverable": False},
            })
        elif task.status == "failed":
            await event_queue.put({
                "event": "error",
                "data": {"message": task.error or "分析失败", "step": -1, "recoverable": False},
            })
        await event_queue.put(None)
    else:
        image_path = task_store.get_image(task_id)
        if image_path is None:
            raise HTTPException(status_code=404, detail="图片不存在")

        agent_service = get_agent_service()
        await agent_service.enqueue(task_id, image_path, event_queue)

    from app.utils.sse import sse_event_generator
    return StreamingResponse(
        sse_event_generator(event_queue),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/task/{task_id}/image")
async def get_task_image(task_id: str):
    """Get the uploaded image (only available during analysis)."""
    task = task_store.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    if task.status == "rejected":
        raise HTTPException(status_code=404, detail="图片不可用")

    image_path = task_store.get_image(task_id)
    if image_path is None or not Path(image_path).exists():
        raise HTTPException(status_code=404, detail="图片不存在或已被清理")

    return FileResponse(image_path, media_type="image/jpeg")


@router.delete("/task/{task_id}", status_code=204)
async def delete_task(task_id: str):
    """Delete a task and its image."""
    image_path = task_store.get_image(task_id)
    if image_path:
        delete_image(image_path)
    task_store.remove(task_id)
    return None
