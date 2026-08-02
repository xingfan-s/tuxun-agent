from app.evaluation import evaluate_predictions, fit_isotonic
from app.schemas.evidence import normalize_evidence
from app.tools.base import ToolBudget
from app.utils.logging import redact_sensitive


def test_isotonic_calibration_is_monotonic():
    calibrator = fit_isotonic([0.1, 0.2, 0.3, 0.4], [0, 1, 0, 1])
    values = [calibrator.predict(score) for score in [0.1, 0.2, 0.3, 0.4]]
    assert values == sorted(values)
    assert all(0 <= value <= 1 for value in values)


def test_evaluation_reports_hierarchical_and_distance_metrics():
    report = evaluate_predictions([{
        "prediction": {"country": "中国", "province": "浙江", "city": "杭州", "lat": 30.27, "lng": 120.15},
        "truth": {"country": "中国", "province": "浙江", "city": "杭州", "lat": 30.28, "lng": 120.16},
        "elapsed_ms": 500,
        "cost": 0.01,
    }])
    assert report["top1"]["city"] == 1.0
    assert report["within_km"]["25"] == 1.0
    assert report["latency_ms"]["p95"] == 500


def test_evidence_normalization_uses_public_contract():
    items = normalize_evidence([{"source": "ocr", "weight": 1.4, "clue": "road sign"}])
    assert items[0]["direction"] == "support"
    assert items[0]["reliability"] == 1.0
    assert items[0]["summary"] == "road sign"


def test_tool_budget_stops_additional_calls():
    budget = ToolBudget(max_calls=1, max_elapsed_seconds=60)
    assert budget.consume() is True
    assert budget.consume() is False


def test_log_redaction_removes_keys_and_api_tokens():
    assert redact_sensitive({"api_key": "secret"})["api_key"] == "[REDACTED]"
    assert "sk-example" not in redact_sensitive("failure sk-example12345")
