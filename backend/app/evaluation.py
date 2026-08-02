"""Repeatable evaluation and interpretable score calibration helpers."""

from bisect import bisect_right
from dataclasses import dataclass
from math import asin, cos, radians, sin, sqrt
from statistics import median
from typing import Any, Iterable
import json
from pathlib import Path

EVALUATION_SLICES = {
    "landmark", "street", "nature", "ocr_strong", "ocr_weak", "night",
    "low_resolution", "non_china",
}


@dataclass(frozen=True)
class IsotonicCalibrator:
    thresholds: tuple[float, ...]
    probabilities: tuple[float, ...]

    def predict(self, score: float) -> float:
        if not self.thresholds:
            return max(0.0, min(1.0, float(score)))
        index = max(0, min(len(self.probabilities) - 1, bisect_right(self.thresholds, float(score)) - 1))
        return self.probabilities[index]

    def to_dict(self) -> dict:
        return {"method": "isotonic", "version": 1, "thresholds": list(self.thresholds), "probabilities": list(self.probabilities)}


def fit_isotonic(scores: Iterable[float], labels: Iterable[int | bool]) -> IsotonicCalibrator:
    """Fit pair-adjacent-violators isotonic regression."""
    pairs = sorted((float(score), 1.0 if bool(label) else 0.0) for score, label in zip(scores, labels))
    if not pairs:
        return IsotonicCalibrator((), ())
    blocks: list[list[float]] = []
    for score, label in pairs:
        blocks.append([score, score, label, 1.0])
        while len(blocks) >= 2:
            left, right = blocks[-2], blocks[-1]
            if left[2] / left[3] <= right[2] / right[3]:
                break
            left[1] = right[1]
            left[2] += right[2]
            left[3] += right[3]
            blocks.pop()
    return IsotonicCalibrator(
        tuple(block[0] for block in blocks),
        tuple(round(max(0.0, min(1.0, block[2] / block[3])), 6) for block in blocks),
    )


def load_calibrator(path: str) -> IsotonicCalibrator | None:
    """Load a versioned calibration artifact generated from an isolated set."""
    if not path:
        return None
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("method") != "isotonic" or payload.get("version", 1) != 1:
            return None
        thresholds = tuple(float(value) for value in payload["thresholds"])
        probabilities = tuple(float(value) for value in payload["probabilities"])
        if not thresholds or len(thresholds) != len(probabilities):
            return None
        return IsotonicCalibrator(thresholds, probabilities)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    value = sin(d_lat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon / 2) ** 2
    return radius * 2 * asin(min(1.0, sqrt(value)))


def _location(item: dict[str, Any]) -> tuple[float, float] | None:
    try:
        lat = float(item.get("lat"))
        lng = float(item.get("lng", item.get("lon")))
    except (TypeError, ValueError):
        return None
    return (lat, lng) if -90 <= lat <= 90 and -180 <= lng <= 180 else None


def evaluate_predictions(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Compute hierarchical Top-1/Top-3, distance, latency and cost metrics."""
    rows = list(rows)
    total = len(rows)
    if not total:
        return {"count": 0, "top1": {}, "top3": {}, "distance_km_median": None,
                "within_km": {"25": 0.0, "50": 0.0, "100": 0.0},
                "latency_ms": {"p50": None, "p95": None}, "cost": {"total": 0.0, "mean": 0.0}}
    matches = {level: [0, 0] for level in ("country", "province", "city")}
    distances: list[float] = []
    latencies: list[float] = []
    costs: list[float] = []
    within = {25: 0, 50: 0, 100: 0}
    for row in rows:
        prediction = row.get("prediction") or row.get("predicted") or {}
        truth = row.get("truth") or row.get("label") or {}
        candidates = row.get("candidates") or [prediction]
        for level in matches:
            expected = str(truth.get(level) or "").strip().casefold()
            values = [str(item.get(level) or "").strip().casefold() for item in candidates]
            if expected and values and values[0] == expected:
                matches[level][0] += 1
            if expected and expected in values[:3]:
                matches[level][1] += 1
        predicted_location, truth_location = _location(prediction), _location(truth)
        if predicted_location and truth_location:
            distance = haversine_km(*predicted_location, *truth_location)
            distances.append(distance)
            for threshold in within:
                within[threshold] += int(distance <= threshold)
        for key, target in (("elapsed_ms", latencies), ("cost", costs)):
            if row.get(key) is not None:
                try:
                    target.append(float(row[key]))
                except (TypeError, ValueError):
                    pass

    def percentile(values: list[float], quantile: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        index = (len(ordered) - 1) * quantile
        lo, hi = int(index), min(len(ordered) - 1, int(index) + 1)
        return round(ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo), 3)

    return {
        "count": total,
        "top1": {level: round(value[0] / total, 4) for level, value in matches.items()},
        "top3": {level: round(value[1] / total, 4) for level, value in matches.items()},
        "distance_km_median": round(median(distances), 3) if distances else None,
        "within_km": {str(key): round(value / total, 4) for key, value in within.items()},
        "latency_ms": {"p50": percentile(latencies, 0.5), "p95": percentile(latencies, 0.95)},
        "cost": {"total": round(sum(costs), 6), "mean": round(sum(costs) / len(costs), 6) if costs else 0.0},
        "coverage": {"with_coordinates": len(distances), "with_latency": len(latencies)},
    }


def validate_dataset_isolation(rows: Iterable[dict[str, Any]], index_entries: Iterable[dict[str, Any]]) -> list[str]:
    """Detect evaluation leakage by hash and stable source identifiers."""
    index_entries = list(index_entries)
    hashes = {str(item.get("image_hash") or item.get("sha256") or "") for item in index_entries}
    source_ids = {(str(item.get("source") or ""), str(item.get("source_id") or item.get("url") or "")) for item in index_entries}
    violations: list[str] = []
    seen_slices: set[str] = set()
    for position, row in enumerate(rows, 1):
        identity = row.get("identity") or row.get("truth") or row
        image_hash = str(identity.get("image_hash") or identity.get("sha256") or "")
        source_key = (str(identity.get("source") or ""), str(identity.get("source_id") or identity.get("url") or ""))
        if image_hash and image_hash in hashes:
            violations.append(f"row {position}: image hash exists in retrieval index")
        if all(source_key) and source_key in source_ids:
            violations.append(f"row {position}: source identity exists in retrieval index")
        slice_name = str(row.get("slice") or "")
        if slice_name:
            seen_slices.add(slice_name)
        if slice_name and slice_name not in EVALUATION_SLICES:
            violations.append(f"row {position}: unsupported slice {slice_name}")
    missing = sorted(EVALUATION_SLICES - seen_slices)
    if missing:
        violations.append("missing slices: " + ", ".join(missing))
    return violations


def evaluate_by_slice(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("slice") or "unspecified"), []).append(row)
    return {name: evaluate_predictions(items) for name, items in sorted(grouped.items())}


def evaluate_ablations(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Compare signal ablations (for example ``without_clip``) reproducibly."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("ablation") or "full"), []).append(row)
    return {name: evaluate_predictions(items) for name, items in sorted(grouped.items())}
