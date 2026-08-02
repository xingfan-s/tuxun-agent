"""In-process task repository used by the local deployment profile."""

from __future__ import annotations

import logging
import time

from app.schemas.task import StepResult, TaskStatus
from app.services.event_bus import task_event_bus

logger = logging.getLogger(__name__)

_TASK_TTL_SECONDS = 30 * 60
TERMINAL_STATUSES = {"done", "failed", "rejected", "cancelled", "expired"}


class TaskRepository:
    """Task snapshots plus compare-and-set lifecycle transitions."""

    def __init__(self):
        self._tasks: dict[str, TaskStatus] = {}
        self._images: dict[str, str] = {}
        self._completed_at: dict[str, float] = {}

    def create(self, task_id: str, image_path: str) -> TaskStatus:
        status = TaskStatus(task_id=task_id, status="uploaded")
        self._tasks[task_id] = status
        self._images[task_id] = image_path
        return status

    def get(self, task_id: str) -> TaskStatus | None:
        return self._tasks.get(task_id)

    def update(self, task_id: str, **kwargs) -> None:
        task = self._tasks.get(task_id)
        if task is None:
            return
        for key, value in kwargs.items():
            setattr(task, key, value)
        if task.status in TERMINAL_STATUSES:
            self._completed_at.setdefault(task_id, time.time())

    def get_image(self, task_id: str) -> str | None:
        return self._images.get(task_id)

    def active_image_paths(self) -> set[str]:
        return {
            path for task_id, path in self._images.items()
            if (self._tasks.get(task_id) and self._tasks[task_id].status not in TERMINAL_STATUSES)
        }

    def claim_start(self, task_id: str) -> bool:
        """Atomically claim uploaded -> queued within the event-loop thread."""
        task = self._tasks.get(task_id)
        if task is None or task.status != "uploaded":
            return False
        task.status = "queued"
        return True

    def record_event(self, event: dict) -> None:
        task = self._tasks.get(event.get("task_id"))
        if task is None:
            return
        task.last_event_id = int(event["id"])
        data = event.get("data") or {}
        if event.get("type") == "progress":
            task.progress = max(0, min(100, int(data.get("progress", task.progress))))
        elif event.get("type") == "step_update":
            try:
                incoming = StepResult.model_validate(data)
            except Exception as exc:
                logger.warning("invalid task step event: %s", exc)
                return
            for index, current in enumerate(task.steps):
                if current.step == incoming.step and current.type == incoming.type:
                    task.steps[index] = incoming
                    break
            else:
                task.steps.append(incoming)

    def request_cancel(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if task is None or task.status in TERMINAL_STATUSES:
            return False
        task.cancel_requested = True
        if task.status in ("uploaded", "queued"):
            task.status = "cancelled"
            task.error = "任务已取消"
            task.error_recoverable = False
            task.progress = 100
            self._completed_at.setdefault(task_id, time.time())
        return True

    def is_cancel_requested(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        return bool(task and task.cancel_requested)

    def remove(self, task_id: str) -> None:
        self._tasks.pop(task_id, None)
        self._images.pop(task_id, None)
        self._completed_at.pop(task_id, None)
        task_event_bus.remove(task_id)

    def expire_old_tasks(self, ttl: float = _TASK_TTL_SECONDS) -> None:
        from app.utils.image import delete_image

        now = time.time()
        expired_ids = [
            task_id
            for task_id, completed_at in self._completed_at.items()
            if now - completed_at > ttl
        ]
        for task_id in expired_ids:
            image_path = self._images.pop(task_id, None)
            if image_path:
                delete_image(image_path)
            self._tasks.pop(task_id, None)
            self._completed_at.pop(task_id, None)
            task_event_bus.remove(task_id)
        if expired_ids:
            logger.info("expired %s task snapshots", len(expired_ids))


task_store = TaskRepository()
