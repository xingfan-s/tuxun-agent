"""Replayable, non-blocking in-process task event bus.

Each subscriber gets its own queue. A bounded history allows reconnects with
Last-Event-ID while a disconnected consumer can never block the worker.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from app.metrics import metrics


@dataclass
class _Channel:
    next_id: int = 1
    history: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=256))
    subscribers: dict[str, asyncio.Queue] = field(default_factory=dict)


class TaskEventBus:
    def __init__(self, history_size: int = 256, subscriber_queue_size: int = 256):
        self.history_size = history_size
        self.subscriber_queue_size = subscriber_queue_size
        self._channels: dict[str, _Channel] = {}
        self._subscriber_counter = 0

    def _channel(self, task_id: str) -> _Channel:
        channel = self._channels.get(task_id)
        if channel is None:
            channel = _Channel(history=deque(maxlen=self.history_size))
            self._channels[task_id] = channel
        return channel

    def publish(self, task_id: str, event_type: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        channel = self._channel(task_id)
        event = {
            "id": channel.next_id,
            "task_id": task_id,
            "type": event_type,
            "occurred_at": datetime.now(timezone.utc).isoformat(),
            "data": data or {},
        }
        channel.next_id += 1
        channel.history.append(event)
        for queue in tuple(channel.subscribers.values()):
            self._put_nowait(queue, event)
        return event

    @staticmethod
    def _put_nowait(queue: asyncio.Queue, item: Any) -> None:
        try:
            queue.put_nowait(item)
        except asyncio.QueueFull:
            # Drop the oldest event for this subscriber. The history remains
            # authoritative and can be replayed on the next reconnect.
            try:
                queue.get_nowait()
                queue.task_done()
            except asyncio.QueueEmpty:
                pass
            try:
                queue.put_nowait(item)
            except asyncio.QueueFull:
                pass

    def subscribe(self, task_id: str, last_event_id: int = 0) -> tuple[str, asyncio.Queue]:
        channel = self._channel(task_id)
        self._subscriber_counter += 1
        subscriber_id = f"s{self._subscriber_counter}"
        queue: asyncio.Queue = asyncio.Queue(maxsize=self.subscriber_queue_size)
        for event in channel.history:
            if event["id"] > last_event_id:
                self._put_nowait(queue, event)
        channel.subscribers[subscriber_id] = queue
        metrics.increment("tuxun_sse_subscriptions_total", reconnect=str(last_event_id > 0).lower())
        metrics.gauge("tuxun_sse_subscribers", len(channel.subscribers), task_id=task_id)
        return subscriber_id, queue

    def unsubscribe(self, task_id: str, subscriber_id: str) -> None:
        channel = self._channels.get(task_id)
        if channel:
            channel.subscribers.pop(subscriber_id, None)
            metrics.gauge("tuxun_sse_subscribers", len(channel.subscribers), task_id=task_id)

    def close(self, task_id: str) -> None:
        channel = self._channels.get(task_id)
        if not channel:
            return
        for queue in tuple(channel.subscribers.values()):
            self._put_nowait(queue, None)

    def last_id(self, task_id: str) -> int:
        channel = self._channels.get(task_id)
        return channel.next_id - 1 if channel else 0

    def remove(self, task_id: str) -> None:
        self.close(task_id)
        self._channels.pop(task_id, None)


task_event_bus = TaskEventBus()
