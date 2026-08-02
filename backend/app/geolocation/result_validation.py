from typing import Any

from app.geolocation.coordinates import bd09_to_wgs84, gcj02_to_wgs84, validate_coordinate
from app.schemas.evidence import normalize_evidence

_COORD_SYSTEMS = {"WGS84", "GCJ-02", "BD-09", "unknown"}

_PRECISION_RADII_M = {
    "country": 1_500_000,
    "province": 300_000,
    "city": 25_000,
    "district": 10_000,
    "road": 2_000,
    "poi": 250,
}


def _has_unique_evidence(result: dict[str, Any]) -> bool:
    unique_markers = {
        "near_duplicate", "road_sign", "unique_landmark", "unique_storefront",
        "verified_poi", "verified_address",
    }
    for item in result.get("evidence") or []:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", item.get("type", ""))).lower()
        if source in unique_markers or any(marker in source for marker in unique_markers):
            return True
        if item.get("unique") is True or item.get("verifiable_unique") is True:
            return True
    return False


def _infer_precision(result: dict[str, Any]) -> str:
    if result.get("district"):
        return "district"
    if result.get("city"):
        return "city"
    if result.get("province"):
        return "province"
    if result.get("country"):
        return "country"
    return "unknown"


def normalize_result(result: dict[str, Any]) -> dict[str, Any]:
    """Apply deterministic result semantics before Pydantic validation."""
    normalized = dict(result)
    lat = normalized.get("lat")
    lng = normalized.get("lng")
    coord_system = normalized.get("coord_system", "WGS84")
    if coord_system not in _COORD_SYSTEMS:
        coord_system = "unknown"
    valid_input = validate_coordinate(lat, lng)
    if valid_input and coord_system == "GCJ-02":
        lat, lng = gcj02_to_wgs84(lat, lng)
        normalized["lat"], normalized["lng"] = lat, lng
        normalized["coord_system"] = "WGS84"
    elif valid_input and coord_system == "BD-09":
        lat, lng = bd09_to_wgs84(lat, lng)
        normalized["lat"], normalized["lng"] = lat, lng
        normalized["coord_system"] = "WGS84"
    elif valid_input and coord_system != "WGS84":
        lat, lng = None, None
    if not validate_coordinate(lat, lng) or (lat == 0 and lng == 0):
        normalized["lat"] = None
        normalized["lng"] = None
        normalized["coord_system"] = "unknown"
    else:
        normalized.setdefault("coord_system", "WGS84")

    precision = normalized.get("precision_level")
    if precision not in {"country", "province", "city", "district", "road", "poi", "unknown"}:
        precision = None
    if precision in {"road", "poi"} and not _has_unique_evidence(normalized):
        precision = None
    normalized["precision_level"] = precision or _infer_precision(normalized)

    if normalized.get("confidence_kind") not in {"calibrated", "ranking_score", "unknown"}:
        normalized["confidence_kind"] = "ranking_score"
    normalized.setdefault("confidence_kind", "ranking_score")
    normalized.setdefault("top_hypotheses", [])
    normalized["evidence"] = normalize_evidence(normalized.get("evidence", []))
    if normalized.get("uncertainty_radius_m") is None:
        normalized["uncertainty_radius_m"] = _PRECISION_RADII_M.get(
            normalized.get("precision_level"), None
        )
    return normalized
