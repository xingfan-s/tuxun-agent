import asyncio
import time
import structlog
from app.config import get_settings
from app.schemas.task import TaskStatus, Result, ToolStats
from app.agent.graph import build_graph
from app.agent.state import AgentState
from app.utils.image import compress_for_vision, delete_image

logger = structlog.get_logger()


class TaskStore:
    """In-memory task storage."""
    def __init__(self):
        self._tasks: dict[str, TaskStatus] = {}
        self._images: dict[str, str] = {}

    def create(self, task_id: str, image_path: str) -> TaskStatus:
        status = TaskStatus(task_id=task_id, status="uploaded")
        self._tasks[task_id] = status
        self._images[task_id] = image_path
        return status

    def get(self, task_id: str) -> TaskStatus | None:
        return self._tasks.get(task_id)

    def update(self, task_id: str, **kwargs):
        if task_id in self._tasks:
            for k, v in kwargs.items():
                setattr(self._tasks[task_id], k, v)

    def get_image(self, task_id: str) -> str | None:
        return self._images.get(task_id)

    def remove(self, task_id: str):
        self._tasks.pop(task_id, None)
        self._images.pop(task_id, None)


task_store = TaskStore()


class AgentService:
    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        self._running = False
        self._graph = build_graph()

    async def start(self):
        settings = get_settings()
        self._running = True
        for i in range(settings.worker_pool_size):
            worker = asyncio.create_task(self._worker(i))
            self._workers.append(worker)
        logger.info("agent_service_started", workers=settings.worker_pool_size)

    async def stop(self):
        self._running = False
        for _ in self._workers:
            await self._queue.put(None)
        for w in self._workers:
            w.cancel()
        logger.info("agent_service_stopped")

    async def enqueue(self, task_id: str, image_path: str, event_queue: asyncio.Queue):
        await self._queue.put((task_id, image_path, event_queue))

    async def _worker(self, worker_id: int):
        logger.info("worker_started", worker_id=worker_id)
        while self._running:
            try:
                task_id, image_path, event_queue = await self._queue.get()
                if task_id is None:
                    break
                await self._run_agent(task_id, image_path, event_queue)
                self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("worker_error", worker_id=worker_id, error=str(e))

    async def _run_agent(self, task_id: str, image_path: str, event_queue: asyncio.Queue):
        settings = get_settings()
        t_start = time.time()

        async def stream_callback(msg):
            await event_queue.put(msg)

        try:
            task_store.update(task_id, status="analyzing")
            image_base64 = compress_for_vision(image_path)

            initial_state: AgentState = {
                "task_id": task_id,
                "image_path": image_path,
                "image_base64": image_base64,
                "safety_passed": False,
                "safety_reason": None,
                "exif_data": None,
                "vision_raw": None,
                "clues": None,
                "messages": [],
                "tool_calls": [],
                "loop_count": 0,
                "failed_tools": set(),
                "last_redirect_at": 0,
                "result": None,
                "error": None,
                "stream_callback": stream_callback,
            }

            final_state = await self._graph.ainvoke(initial_state)

            safety_passed = final_state.get("safety_passed", False)

            if not safety_passed:
                safety_reason = final_state.get("safety_reason", "安全预检未通过")
                task_store.update(task_id, status="rejected", safety_reason=safety_reason)
                await event_queue.put({
                    "event": "error",
                    "data": {"message": safety_reason, "step": 0, "recoverable": False},
                })
                delete_image(image_path)
                return

            result_data = final_state.get("result", {})
            total_elapsed = int((time.time() - t_start) * 1000)

            tool_stats_data = result_data.get("tool_stats", {})
            final_result = Result(
                address=result_data.get("address", ""),
                country=result_data.get("country", ""),
                province=result_data.get("province"),
                city=result_data.get("city"),
                district=result_data.get("district"),
                lat=result_data.get("lat", 0),
                lng=result_data.get("lng", 0),
                confidence=result_data.get("confidence", 0),
                reasoning=result_data.get("reasoning", ""),
                tokens_used=result_data.get("tokens_used", 0),
                total_elapsed_ms=total_elapsed,
                tool_stats=ToolStats(**tool_stats_data) if tool_stats_data else None,
            )

            task_store.update(task_id, status="done", result=final_result, progress=100)

            await event_queue.put({
                "event": "result",
                "data": final_result.model_dump(),
            })

            if settings.safety_delete_image_after_done:
                delete_image(image_path)

        except Exception as e:
            logger.error("agent_run_error", task_id=task_id, error=str(e))
            task_store.update(task_id, status="failed", error=str(e))
            await event_queue.put({
                "event": "error",
                "data": {"message": f"分析失败：{str(e)}", "step": -1, "recoverable": False},
            })
            delete_image(image_path)
        finally:
            await event_queue.put(None)
