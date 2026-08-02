import sys

import pytest


def test_openai_client_is_compatible_with_httpx():
    from openai import OpenAI

    client = OpenAI(api_key="test", base_url="http://127.0.0.1:1/v1")
    client.close()


def test_torch_runtime_loads_on_windows():
    if sys.platform != "win32":
        pytest.skip("Windows native runtime compatibility check")

    import torch

    assert torch.zeros(1).item() == 0


def test_clip_consumers_share_process_model_load_lock():
    from app.tools import clip_search, geoclip

    assert clip_search.MODEL_LOAD_LOCK is geoclip.MODEL_LOAD_LOCK


def test_clip_search_respects_transformers_offline_mode(monkeypatch):
    from app.tools import clip_search

    monkeypatch.delenv("MODEL_OFFLINE", raising=False)
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    assert clip_search._model_offline() is True


def test_clip_embedder_singleton_is_thread_safe(monkeypatch):
    import time
    from concurrent.futures import ThreadPoolExecutor
    from app.tools import clip_search

    init_count = 0

    class FakeEmbedder:
        def __init__(self):
            nonlocal init_count
            init_count += 1
            time.sleep(0.02)

    monkeypatch.setattr(clip_search, "_embedder", None)
    monkeypatch.setattr(clip_search, "CLIPEmbedder", FakeEmbedder)

    with ThreadPoolExecutor(max_workers=4) as executor:
        instances = list(executor.map(lambda _: clip_search.get_embedder(), range(4)))

    assert init_count == 1
    assert all(instance is instances[0] for instance in instances)
