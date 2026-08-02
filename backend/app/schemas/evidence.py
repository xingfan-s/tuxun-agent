"""Public evidence and tool status contracts."""

from typing import Any, Literal

from pydantic import BaseModel, Field


EvidenceDirection = Literal["support", "contradict", "context"]
EvidenceLocality = Literal["global", "country", "province", "city", "district", "road", "poi"]
ToolStatus = Literal[
    "success", "unavailable", "timeout", "invalid_input", "upstream_error",
    "empty_result", "failed", "skipped",
]


class Evidence(BaseModel):
    """A traceable signal contributing to a location hypothesis."""

    source: str = Field(min_length=1, max_length=64)
    direction: EvidenceDirection = "support"
    locality: EvidenceLocality = "global"
    reliability: float = Field(default=0.0, ge=0.0, le=1.0)
    raw_score: float | None = None
    calibrated_contribution: float | None = None
    summary: str = Field(default="", max_length=500)
    unique: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


def normalize_evidence(items: Any) -> list[dict[str, Any]]:
    """Normalize legacy model dictionaries to the public evidence shape."""
    if not isinstance(items, list):
        return []
    normalized: list[dict[str, Any]] = []
    valid_localities = {"global", "country", "province", "city", "district", "road", "poi"}
    for item in items:
        if not isinstance(item, dict):
            continue
        direction = item.get("direction", "support")
        if direction not in {"support", "contradict", "context"}:
            direction = "context"
        locality = item.get("locality", "global")
        if locality not in valid_localities:
            locality = "global"
        try:
            reliability = max(0.0, min(1.0, float(item.get("reliability", item.get("weight", 0.0)) or 0.0)))
        except (TypeError, ValueError):
            reliability = 0.0
        raw_score = item.get("raw_score", item.get("score"))
        try:
            raw_score = float(raw_score) if raw_score is not None else None
        except (TypeError, ValueError):
            raw_score = None
        contribution = item.get("calibrated_contribution")
        try:
            contribution = float(contribution) if contribution is not None else None
        except (TypeError, ValueError):
            contribution = None
        normalized.append({
            "source": str(item.get("source") or item.get("type") or "unknown")[:64],
            "direction": direction,
            "locality": locality,
            "reliability": reliability,
            "raw_score": raw_score,
            "calibrated_contribution": contribution,
            "summary": str(item.get("summary") or item.get("clue") or item.get("reason") or "")[:500],
            "unique": bool(item.get("unique") or item.get("verifiable_unique")),
            "metadata": item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
        })
    return normalized
