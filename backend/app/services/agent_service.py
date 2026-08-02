import asyncio
import time
from app.utils.logging import structlog
from app.config import get_settings
from app.schemas.task import TaskStatus, Result, ToolStats
from app.geolocation.result_validation import normalize_result
from app.services.event_bus import task_event_bus
from app.services.task_repository import task_store
from app.utils.image import compress_for_vision, delete_image, cleanup_expired_uploads
from app.tools.base import ToolBudget
from app.metrics import metrics
from app.safety.text_check import OcrCapabilityError

logger = structlog.get_logger()


def _build_result(result_data: dict, total_elapsed: int) -> Result:
    tool_stats_data = result_data.get("tool_stats", {})
    allowed = {
        key: result_data[key] for key in (
            "address", "country", "province", "city", "district", "lat", "lng",
            "coord_system", "precision_level", "uncertainty_radius_m", "confidence",
            "confidence_kind", "reasoning", "evidence", "top_hypotheses",
        ) if key in result_data
    }
    return Result(
        **allowed,
        tokens_used=int(result_data.get("tokens_used", 0) or 0),
        total_elapsed_ms=total_elapsed,
        tool_stats=ToolStats(**{
            key: tool_stats_data.get(key, 0)
            for key in ("total_calls", "success", "timeout", "failed", "unavailable",
                        "budget_skipped", "invalid_input", "upstream_error", "empty_result")
        }) if tool_stats_data else ToolStats(),
        model_calls=int(result_data.get("model_calls", 0) or 0),
        model_usage=result_data.get("model_usage", {}) or {},
        estimated_cost=float(result_data.get("estimated_cost", 0) or 0),
    )


class AgentService:
    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        self._running = False
        self._graph = None
        self._graph_error: str | None = None
        try:
            from app.agent.graph import build_graph
            self._graph = build_graph()
        except Exception as exc:
            self._graph_error = f"{type(exc).__name__}: {exc}"
            logger.warning("agent_graph_unavailable", error=self._graph_error)
        self._cleanup_task: asyncio.Task | None = None

    async def start(self):
        if self._running:
            return
        settings = get_settings()
        self._running = True
        for i in range(settings.worker_pool_size):
            worker = asyncio.create_task(self._worker(i))
            self._workers.append(worker)
        # Periodic TTL cleanup of expired tasks
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        logger.info("agent_service_started", workers=settings.worker_pool_size)

    async def stop(self):
        if not self._running:
            return
        self._running = False
        for _ in self._workers:
            await self._queue.put(None)
        await asyncio.gather(*self._workers, return_exceptions=True)
        if self._cleanup_task:
            self._cleanup_task.cancel()
            await asyncio.gather(self._cleanup_task, return_exceptions=True)
        self._workers.clear()
        logger.info("agent_service_stopped")

    async def _periodic_cleanup(self, interval: int = 300):
        """Run TTL expiry cleanup every `interval` seconds."""
        while self._running:
            try:
                task_store.expire_old_tasks()
                settings = get_settings()
                removed = cleanup_expired_uploads(
                    settings.upload_dir,
                    settings.upload_ttl_seconds,
                    task_store.active_image_paths(),
                )
                if removed:
                    logger.info("upload_cleanup_completed", removed=removed)
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("task_cleanup_failed", error=str(exc))

    async def start_task(self, task_id: str):
        """Atomically transition uploaded -> queued and enqueue once."""
        image_path = task_store.get_image(task_id)
        task = task_store.get(task_id)
        if task is None or image_path is None:
            return None
        if task_store.claim_start(task_id):
            await self._queue.put((task_id, image_path))
            metrics.increment("tuxun_tasks_started_total")
            metrics.gauge("tuxun_task_queue_length", self._queue.qsize())
            self._publish(task_id, {"event": "progress", "data": {"progress": 1, "status": "queued"}})
        return task_store.get(task_id)

    async def enqueue(self, task_id: str, image_path: str | None = None, event_queue=None):
        """Backward-compatible idempotent entry point."""
        return await self.start_task(task_id)

    async def _worker(self, worker_id: int):
        logger.info("worker_started", worker_id=worker_id)
        while self._running:
            item = None
            try:
                item = await self._queue.get()
                if item is None:
                    break
                task_id, image_path = item
                metrics.gauge("tuxun_task_queue_length", self._queue.qsize())
                await self._run_agent(task_id, image_path)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("worker_error", worker_id=worker_id, error=str(e))
            finally:
                self._queue.task_done()

    def _publish(self, task_id: str, message: dict):
        event_type = message.get("event", "step_update")
        # Raw cumulative LLM output is never sent to the browser.
        if event_type == "llm_token":
            return
        event = task_event_bus.publish(task_id, event_type, message.get("data") or {})
        metrics.increment("tuxun_sse_events_total", event_type=str(event_type))
        task_store.record_event(event)

    async def _run_agent(self, task_id: str, image_path: str):
        settings = get_settings()
        t_start = time.time()

        async def stream_callback(msg):
            if task_store.is_cancel_requested(task_id):
                raise asyncio.CancelledError()
            self._publish(task_id, msg)

        try:
            if self._graph is None:
                raise RuntimeError(f"Agent 图不可用: {self._graph_error or 'unknown error'}")
            from app.agent.state import AgentState
            if task_store.is_cancel_requested(task_id):
                raise asyncio.CancelledError()
            task_store.update(task_id, status="analyzing")
            self._publish(task_id, {"event": "progress", "data": {"progress": 2, "status": "analyzing"}})
            image_base64 = compress_for_vision(image_path)

            initial_state: AgentState = {
                "task_id": task_id,
                "image_path": image_path,
                "image_base64": image_base64,
                "safety_passed": False,
                "safety_reason": None,
                "exif_data": None,
                "vision_raw": None,
                "vision_region": None,
                "clues": None,
                "ocr_data": None,
                "ocr_fused_queries": None,
                "search_strategy": None,
                "hypotheses": [],
                "excluded_provinces": [],
                "messages": [],
                "tool_calls": [],
                "loop_count": 0,
                "failed_tools": set(),
                "last_redirect_at": 0,
                "result": None,
                "error": None,
                "geoclip_result": None,
                "geoclip_anchors": None,
                "clip_result": None,
                "verification_passed": None,
                "verification_feedback": None,
                "verification_history": [],
                "stream_callback": stream_callback,
                "tokens_used": 0,
                "model_calls": 0,
                "model_usage": {},
                "tool_budget": ToolBudget(settings.max_total_tool_calls, settings.max_tool_elapsed_seconds),
            }

            async with asyncio.timeout(settings.task_timeout_seconds):
                final_state = await self._graph.ainvoke(initial_state)

            safety_passed = final_state.get("safety_passed", False)

            if not safety_passed:
                safety_reason = final_state.get("safety_reason", "安全预检未通过")
                task_store.update(task_id, status="rejected", safety_reason=safety_reason,
                                  error=safety_reason, error_recoverable=False, progress=100)
                self._publish(task_id, {
                    "event": "error",
                    "data": {"message": safety_reason, "reason": safety_reason, "recoverable": False},
                })
                delete_image(image_path)
                metrics.increment("tuxun_task_terminal_total", status="rejected")
                metrics.observe("tuxun_task_duration_seconds", time.time() - t_start, status="rejected")
                return

            result_data = normalize_result(final_state.get("result") or {})
            total_elapsed = int((time.time() - t_start) * 1000)

            final_result = _build_result(result_data, total_elapsed)

            task_store.update(task_id, status="done", result=final_result, progress=100)
            metrics.increment("tuxun_task_terminal_total", status="done")
            metrics.observe("tuxun_task_duration_seconds", time.time() - t_start, status="done")
            metrics.increment("tuxun_model_calls_total", final_result.model_calls)

            self._publish(task_id, {
                "event": "result",
                "data": final_result.model_dump(mode="json"),
            })

            if settings.safety_delete_image_after_done:
                delete_image(image_path)

        except asyncio.CancelledError:
            task_store.update(task_id, status="cancelled", progress=100, error="任务已取消")
            self._publish(task_id, {"event": "error", "data": {"message": "任务已取消", "cancelled": True, "recoverable": False}})
            delete_image(image_path)
            metrics.increment("tuxun_task_terminal_total", status="cancelled")
            metrics.observe("tuxun_task_duration_seconds", time.time() - t_start, status="cancelled")
        except OcrCapabilityError:
            message = "安全检查服务暂不可用，请稍后重试"
            task_store.update(task_id, status="failed", progress=100, error=message,
                              error_recoverable=True)
            self._publish(task_id, {
                "event": "error",
                "data": {
                    "message": message,
                    "code": "safety_ocr_unavailable",
                    "recoverable": True,
                },
            })
            delete_image(image_path)
            metrics.increment("tuxun_task_terminal_total", status="failed")
            metrics.observe("tuxun_task_duration_seconds", time.time() - t_start, status="failed")
        except asyncio.TimeoutError:
            task_store.update(task_id, status="failed", progress=100, error="任务超时",
                              error_recoverable=False)
            self._publish(task_id, {"event": "error", "data": {"message": "任务超时", "code": "timeout", "recoverable": False}})
            delete_image(image_path)
            metrics.increment("tuxun_task_terminal_total", status="timeout")
            metrics.observe("tuxun_task_duration_seconds", time.time() - t_start, status="timeout")
        except Exception as e:
            import traceback
            logger.error("agent_run_error", task_id=task_id, error=str(e),
                        traceback=traceback.format_exc())
            task_store.update(task_id, status="failed", error="分析失败，请稍后重试",
                              error_recoverable=False)
            self._publish(task_id, {
                "event": "error",
                "data": {"message": "分析失败，请稍后重试", "step": -1, "recoverable": False},
            })
            delete_image(image_path)
            metrics.increment("tuxun_task_terminal_total", status="failed")
            metrics.observe("tuxun_task_duration_seconds", time.time() - t_start, status="failed")
        finally:
            task_event_bus.close(task_id)
