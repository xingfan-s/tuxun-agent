from app.services.event_bus import TaskEventBus
from app.services.task_repository import TaskRepository


def test_subscribers_receive_replay_independently():
    bus = TaskEventBus(history_size=8, subscriber_queue_size=8)
    bus.publish("task", "progress", {"progress": 10})
    first_id, first = bus.subscribe("task", 0)
    second_id, second = bus.subscribe("task", 0)
    assert first.get_nowait()["id"] == second.get_nowait()["id"] == 1
    bus.publish("task", "progress", {"progress": 20})
    assert first.get_nowait()["data"]["progress"] == 20
    assert second.get_nowait()["data"]["progress"] == 20
    bus.unsubscribe("task", first_id)
    bus.unsubscribe("task", second_id)


def test_task_start_claim_is_idempotent():
    store = TaskRepository()
    store.create("task", "image.jpg")
    assert store.claim_start("task") is True
    assert store.claim_start("task") is False
    assert store.get("task").status == "queued"


def test_task_snapshot_records_progress_and_steps():
    store = TaskRepository()
    store.create("task", "image.jpg")
    store.record_event({
        "id": 1,
        "task_id": "task",
        "type": "progress",
        "data": {"progress": 20},
    })
    store.record_event({
        "id": 2,
        "task_id": "task",
        "type": "step_update",
        "data": {
            "step": 1,
            "type": "result_enrichment",
            "label": "结果丰富化",
            "status": "done",
            "data": {},
            "elapsed_ms": 10,
        },
    })
    task = store.get("task")
    assert task.progress == 20
    assert task.last_event_id == 2
    assert task.steps[0].type == "result_enrichment"
