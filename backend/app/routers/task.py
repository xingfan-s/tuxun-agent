import uuid
from app.utils.logging import structlog
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Header
from fastapi.responses import StreamingResponse, FileResponse
from app.schemas.task import TaskStatus, UploadResponse
from app.services.agent_service import AgentService
from app.services.event_bus import task_event_bus
from app.services.task_repository import task_store
from app.utils.image import validate_image, save_upload, delete_image
from app.config import get_settings

logger = structlog.get_logger()

router = APIRouter()

_agent_service: AgentService | None = None


@router.get("/config")
async def get_public_config():
    """Return public runtime config for the frontend."""
    from app.config import get_settings
    settings = get_settings()
    return {
        "amap_api_key": settings.amap_web_key or "",
        "map_service": settings.map_service,
    }


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
    try:
        limit = get_settings().max_file_size_mb * 1024 * 1024
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = await file.read(min(1024 * 1024, limit + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > limit:
                break
        content = b"".join(chunks)
    except Exception as e:
        logger.error("upload_read_failed", error_type=type(e).__name__)
        raise HTTPException(status_code=500, detail=f"读取文件失败: {str(e)}")

    valid, error_msg = validate_image(file.content_type or "", len(content), content)
    if not valid:
        raise HTTPException(status_code=422, detail=error_msg)

    try:
        file_path, _ = save_upload(content, file.filename or "image.jpg", file.content_type)
    except OSError as e:
        logger.error("upload_save_failed", error_type=type(e).__name__)
        raise HTTPException(status_code=500, detail=f"保存文件失败（磁盘或权限问题）")
    except Exception as e:
        logger.error("upload_save_failed", error_type=type(e).__name__)
        raise HTTPException(status_code=500, detail=f"保存文件失败: {str(e)}")

    task_id = uuid.uuid4().hex[:12]
    task_store.create(task_id, file_path)
    logger.info("upload_success", task_id=task_id, size=len(content))

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


@router.post("/task/{task_id}/start", response_model=TaskStatus)
async def start_task(task_id: str):
    """Idempotently enqueue a task; repeated calls return current state."""
    if task_store.get(task_id) is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    task = await get_agent_service().start_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="图片不存在")
    return task


@router.get("/task/{task_id}/stream")
async def stream_task(task_id: str, last_event_id: int = Header(default=0, alias="Last-Event-ID")):
    """SSE stream for real-time task progress."""
    task = task_store.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")

    subscriber_id, subscribed_queue = task_event_bus.subscribe(task_id, max(0, last_event_id))
    if task.status in ("done", "failed", "rejected", "cancelled", "expired"):
        task_event_bus._put_nowait(subscribed_queue, None)
    from app.utils.sse import sse_event_generator
    return StreamingResponse(
        sse_event_generator(
            subscribed_queue,
            on_close=lambda: task_event_bus.unsubscribe(task_id, subscriber_id),
        ),
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
    """Request cancellation; the worker owns final cleanup."""
    task = task_store.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    image_path = task_store.get_image(task_id)
    was_pending = task.status in ("uploaded", "queued")
    cancelled = task_store.request_cancel(task_id)
    if cancelled and was_pending:
        get_agent_service()._publish(task_id, {
            "event": "error",
            "data": {"message": "任务已取消", "cancelled": True, "recoverable": False},
        })
        task_event_bus.close(task_id)
    if image_path and was_pending:
        delete_image(image_path)
    return None
