from types import SimpleNamespace

import pytest


def test_result_builder_does_not_duplicate_model_metadata():
    from app.services.agent_service import _build_result

    result = _build_result({
        "address": "海南省三亚市",
        "model_calls": 3,
        "model_usage": {"qwen": {"calls": 3}},
        "estimated_cost": 0.25,
        "tokens_used": 42,
    }, total_elapsed=1500)

    assert result.address == "海南省三亚市"
    assert result.model_calls == 3
    assert result.model_usage == {"qwen": {"calls": 3}}
    assert result.estimated_cost == 0.25
    assert result.tokens_used == 42
    assert result.total_elapsed_ms == 1500


def test_hypothesis_sanitizer_rejects_narrative_and_restores_selected_score():
    from app.agent.nodes import _sanitize_hypotheses

    hypotheses = [
        {"province": "海南", "score": 0.05, "supporting_evidence": []},
        {"province": "进一步调查华南沿海地区，特别是海南省的可能性。", "score": 0.5},
        {"province": "广东", "score": 0.37, "supporting_evidence": []},
    ]

    sanitized = _sanitize_hypotheses(hypotheses, "海南省", 0.6)

    assert [item["province"] for item in sanitized] == ["海南", "广东"]
    assert sanitized[0]["score"] == 0.6


def test_result_builder_preserves_budget_skip_stats():
    from app.services.agent_service import _build_result

    result = _build_result({
        "address": "海南省三亚市",
        "tool_stats": {"total_calls": 2, "unavailable": 0, "budget_skipped": 2},
    }, total_elapsed=10)

    assert result.tool_stats is not None
    assert result.tool_stats.unavailable == 0
    assert result.tool_stats.budget_skipped == 2


@pytest.mark.asyncio
async def test_geoclip_node_invokes_lazy_prediction(monkeypatch):
    import app.agent.nodes as nodes

    called = []

    def predict(image_path: str, top_k: int):
        called.append((image_path, top_k))
        return []

    monkeypatch.setattr(nodes, "predict_location", predict)
    monkeypatch.setattr(nodes, "geoclip_load_status", lambda: "unavailable:RuntimeError")
    monkeypatch.setattr(nodes, "get_settings", lambda: SimpleNamespace(geoclip_enabled=True))

    state = {"image_path": "sample.jpg", "stream_callback": None}
    await nodes.geoclip_node(state)

    assert called == [("sample.jpg", 10)]
    assert state["geoclip_result"] is None


def test_near_duplicate_match_supplies_wgs84_coordinates():
    from app.agent.nodes import _apply_near_duplicate_match

    result = {"address": "中国·海南省", "province": "海南省", "evidence": []}
    state = {
        "clip_result": {
            "matches": [{"lat": 18.2094, "lon": 109.4943, "similarity": 0.9999}],
            "top_provinces": [{
                "lat": 18.2094,
                "lon": 109.4943,
                "province": "海南省",
                "city": "三亚市",
            }],
        },
    }

    _apply_near_duplicate_match(result, state)

    assert result["lat"] == 18.2094
    assert result["lng"] == 109.4943
    assert result["coord_system"] == "WGS84"
    assert result["address"] == "中国·海南省·三亚市"
    assert result["evidence"][0]["source"] == "near_duplicate"


@pytest.mark.asyncio
async def test_result_enrichment_prefers_city_and_preserves_coordinate_system(monkeypatch):
    import app.agent.nodes as nodes
    from app.tools.map import GeoResult

    class MapService:
        def __init__(self):
            self.queries = []

        async def geocode(self, query):
            self.queries.append(query)
            return [GeoResult(18.2528, 109.5119, "三亚市", province="海南省", city="三亚市")]

        async def search_nearby(self, *args, **kwargs):
            return []

        async def reverse_geocode(self, lat, lng):
            return None

    service = MapService()
    monkeypatch.setattr(nodes, "create_map_service", lambda: service)
    state = {
        "result": {
            "address": "中国·海南省",
            "province": "海南省",
            "city": "三亚市",
            "confidence": 0.6,
            "lat": None,
            "lng": None,
            "evidence": [],
        },
        "stream_callback": None,
    }

    await nodes.result_enrichment_node(state)

    assert service.queries[0] == "海南省三亚市"
    assert state["result"]["lat"] == 18.2528
    assert state["result"]["lng"] == 109.5119
    assert state["result"]["coord_system"] == "WGS84"
