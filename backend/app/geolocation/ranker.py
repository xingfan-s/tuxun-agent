"""
Candidate scoring system (v2.0 Phase 3).

Fuses 5 signal sources with fixed weights to produce a ranked list
of candidate provinces/cities before the ReAct verification loop.

Weight allocation (v2.2 — adaptive CLIP weight based on DB size):
  When DB < 5000 images (current):
    GeoCLIP     25%  — GPS prediction (global model)
    CLIP search  15%  — small DB, limited coverage, low hit rate
    OCR          30%  — license plates, phone codes (most definitive)
    Qwen vision  20%  — macro region + architecture + climate context
    Knowledge    10%  — vegetation / script / climate rules

  When DB >= 5000 images (after GeoComp/OSV-5M ingestion):
    GeoCLIP     20%  — reduced as CLIP+FAISS becomes more reliable
    CLIP search  30%  — large street-level DB, high coverage, good hit rate
    OCR          25%  — still strong but less dominant
    Qwen vision  15%  — macro region + architecture + climate context
    Knowledge    10%  — vegetation / script / climate rules (province-level)

Usage:
    from app.geolocation.ranker import score_candidates
    ranked = score_candidates(geoclip_result, clip_result, ocr_data, clues, vision_region)
    # ranked[:20] → top 20 candidates for downstream processing
"""

import json
from app.config import get_settings
from app.evaluation import load_calibrator

from app.tools.china_knowledge import (
    normalize_province_name,
    get_provinces_for_region,
    get_china_script_region_rules,
    get_china_climate_rules,
    match_city_features,
    get_city_fingerprint,
)
from app.tools.china_knowledge import (
    ARCHITECTURE_FINGERPRINTS,
    VEGETATION_RANGES,
)


# ============================================================
# Weight configuration
# ============================================================

def _get_adaptive_weights():
    """Return traceable fusion weights from runtime configuration."""
    if _W_OVERRIDE is not None:
        return tuple(_W_OVERRIDE)
    settings = get_settings()
    weights = (
        settings.rank_weight_geoclip,
        settings.rank_weight_clip,
        settings.rank_weight_ocr,
        settings.rank_weight_vision,
        settings.rank_weight_knowledge,
    )
    total = sum(weights)
    if total <= 0:
        raise ValueError("candidate fusion weights must have a positive sum")
    return tuple(value / total for value in weights)


# Module-level weights initialized at import time
# Re-computed each time score_candidates() is called
_W_OVERRIDE = None  # Allow override for testing


# ============================================================
# Signal extraction helpers
# ============================================================

def _extract_geo_signals(geoclip_result: dict | None, w_geo: float = 0.25) -> dict[str, dict]:
    """Extract province-level GeoCLIP signals.

    Returns: {province_short: {"score": float, "prob": float, "lat": float, "lon": float}}
    """
    signals: dict[str, dict] = {}
    if not geoclip_result:
        return signals
    for pred in geoclip_result.get("top_provinces", []):
        province = normalize_province_name(pred.get("province", ""))
        if not province:
            continue
        prob = pred.get("probability", 0)
        score = w_geo * prob
        if province not in signals or score > signals[province]["score"]:
            signals[province] = {
                "score": score,
                "prob": prob,
                "lat": pred.get("lat"),
                "lon": pred.get("lon"),
            }
    return signals


def _extract_clip_signals(clip_result: dict | None, w_clip: float = 0.25) -> dict[str, dict]:
    """Extract province-level CLIP similarity signals.

    Returns: {province_short: {"score": float, "similarity": float}}
    """
    signals: dict[str, dict] = {}
    if not clip_result:
        return signals
    for pred in clip_result.get("top_provinces", []):
        province = normalize_province_name(pred.get("province", ""))
        if not province:
            continue
        similarity = float(pred.get("similarity", pred.get("score", 0.0)))
        similarity = max(-1.0, min(1.0, similarity))
        # IP search over L2-normalized vectors is cosine similarity. Keep it
        # as an explicitly uncalibrated ranking signal.
        similarity = max(0.0, similarity)
        score = w_clip * similarity
        if province not in signals or score > signals[province]["score"]:
            signals[province] = {
                "score": score,
                "similarity": similarity,
            }
    return signals


def _extract_ocr_signals(ocr_data: dict | None, w_ocr: float = 0.25) -> dict[str, dict]:
    """Extract OCR-based definitive signals.

    Returns: {province_short: {"score": float, "source": str}}
    """
    signals: dict[str, dict] = {}
    if not ocr_data:
        return signals

    # License plates (strongest OCR signal)
    for plate in ocr_data.get("license_plates", []):
        province = normalize_province_name(plate.get("province", ""))
        if not province or province == "未知":
            continue
        confidence = plate.get("confidence", "partial_plate")
        multiplier = 1.0 if confidence == "full_plate" else 0.7
        score = w_ocr * multiplier
        key = f"OCR车牌:{province}"
        signals[key] = {"score": score, "province": province, "source": "license_plate"}

    # Phone area codes
    for phone in ocr_data.get("phone_area_codes", []):
        province = normalize_province_name(phone.get("province", ""))
        if not province or province == "未知":
            continue
        score = w_ocr * 0.9
        key = f"OCR区号:{province}"
        if key not in signals:
            signals[key] = {"score": score, "province": province, "source": "phone_code"}

    # Highway numbers
    for hwy in ocr_data.get("highways", []):
        hwy_provinces = hwy.get("provinces", [])
        if not hwy_provinces:
            continue
        score_each = w_ocr * 0.6 / len(hwy_provinces)
        for p in hwy_provinces:
            province = normalize_province_name(p)
            if not province:
                continue
            key = f"OCR公路:{province}"
            if key not in signals:
                signals[key] = {"score": score_each, "province": province, "source": "highway"}

    return signals


def _extract_vision_signals(clues: dict, vision_region: str | None, w_vision: float = 0.15) -> dict[str, dict]:
    """Extract vision-based signals from macro region and architecture clues.

    Returns: {province_short: {"score": float, "source": str}}
    """
    signals: dict[str, dict] = {}

    # Macro region (broad signal, split across all provinces in region)
    if vision_region and vision_region != "无法判断":
        region_provinces = get_provinces_for_region(vision_region)
        if region_provinces:
            score_each = w_vision * 0.6 / len(region_provinces)
            for p in region_provinces:
                key = f"视觉区域:{p}"
                signals[key] = {"score": score_each, "province": p, "source": "vision_region"}

    # Architecture-specific matches (narrower, higher per-province weight)
    arch = clues.get("architecture", "")
    if arch:
        for af in ARCHITECTURE_FINGERPRINTS:
            if any(kw in arch for kw in af.get("keywords", [])):
                matched_provinces = af.get("provinces", [])
                if matched_provinces:
                    score_each = w_vision * 0.4 / len(matched_provinces)
                    for p in matched_provinces:
                        province = normalize_province_name(p)
                        if not province:
                            continue
                        key = f"视觉建筑:{province}"
                        existing = signals.get(key, {}).get("score", 0)
                        signals[key] = {
                            "score": max(existing, score_each),
                            "province": province,
                            "source": "architecture",
                        }

    return signals


def _extract_kb_signals(clues: dict, w_kb: float = 0.10) -> dict[str, dict]:
    """Extract knowledge-base signals from vegetation, script, and climate rules.

    Returns: {province_short: {"score": float, "source": str}}
    """
    signals: dict[str, dict] = {}
    veg = clues.get("vegetation", [])
    veg_str = " ".join(veg) if veg else ""
    scripts = clues.get("script", [])

    # Vegetation ranges
    if veg_str:
        for vr in VEGETATION_RANGES:
            if "provinces" in vr and any(kw in veg_str for kw in vr.get("keywords", [])):
                matched = vr["provinces"]
                score_each = w_kb * 0.5 / len(matched)
                for p in matched:
                    province = normalize_province_name(p)
                    if not province:
                        continue
                    key = f"KB植被:{province}"
                    signals[key] = {"score": score_each, "province": province, "source": "vegetation"}

    # Script rules
    for script in scripts:
        for rule in get_china_script_region_rules():
            if rule["script"] == script:
                matched = rule.get("expected_provinces", [])
                score_each = w_kb * 0.5 / len(matched)
                for p in matched:
                    province = normalize_province_name(p)
                    if not province:
                        continue
                    key = f"KB文字:{province}"
                    existing = signals.get(key, {}).get("score", 0)
                    signals[key] = {
                        "score": max(existing, score_each),
                        "province": province,
                        "source": "script",
                    }

    # Climate rules — check for lat-based exclusions
    for rule in get_china_climate_rules():
        cond = rule.get("condition", "")
        if cond == "desert_landscape":
            if any(kw in veg_str for kw in ["沙漠", "荒漠", "戈壁", "沙丘"]):
                expected = rule.get("expected_provinces", [])
                score_each = w_kb * 0.2 / len(expected) if expected else 0
                for p in expected:
                    province = normalize_province_name(p)
                    if not province:
                        continue
                    key = f"KB气候:{province}"
                    existing = signals.get(key, {}).get("score", 0)
                    signals[key] = {
                        "score": max(existing, score_each),
                        "province": province,
                        "source": "climate",
                    }

    return signals


def _extract_city_signals(clues: dict, vision_raw: str = "", w_vision: float = 0.15) -> dict[str, dict]:
    """Extract city-level fingerprint match signals (v2.1).

    Matches visual clues against CITY_FINGERPRINTS for granular
    city-level distinction within provinces.

    Returns: {province_short: {"score": float, "city": str, "source": str}}
    """
    signals: dict[str, dict] = {}
    if not clues and not vision_raw:
        return signals

    combined = vision_raw + " " + json.dumps(clues, ensure_ascii=False) if clues else vision_raw
    if len(combined) < 10:
        return signals

    from app.tools.china_knowledge import CITY_FINGERPRINTS

    for cf in CITY_FINGERPRINTS:
        city = cf["city"]
        province = normalize_province_name(cf["province"])
        if not province:
            continue
        matches, matched_kw = match_city_features(combined, city)
        if matches >= 2:  # At least 2 keyword matches for signal
            # More matches = higher confidence
            score = w_vision * 0.3 * min(matches / 5.0, 1.0)
            key = f"城市指纹:{city}"
            signals[key] = {
                "score": round(score, 4),
                "province": province,
                "city": city,
                "source": "city_fingerprint",
                "matched_keywords": matched_kw[:5],
            }

    return signals


# ============================================================
# Main scoring function
# ============================================================

def score_candidates(
    geoclip_result: dict | None = None,
    clip_result: dict | None = None,
    ocr_data: dict | None = None,
    clues: dict | None = None,
    vision_region: str | None = None,
    vision_raw: str = "",
    excluded_provinces: list[str] | None = None,
) -> list[dict]:
    """Score all Chinese provinces by fusing 5 signal sources.

    Args:
        geoclip_result: GeoCLIP GPS prediction results
        clip_result: CLIP+FAISS similarity search results
        ocr_data: OCR extracted text data
        clues: Structured visual clues from clue_extract_node
        vision_region: Macro region from vision_macro_node
        vision_raw: Raw vision detail output for city fingerprint matching
        excluded_provinces: List of provinces to exclude (score = 0)

    Returns:
        List of candidate dicts sorted by score descending:
        [{"province": str, "city": str|None, "score": float, "signals": [...]}, ...]
    """
    clues = clues or {}
    excluded = set(excluded_provinces or [])

    # v2.2: Adaptive weights based on CLIP+FAISS database size
    W_GEO, W_CLIP, W_OCR, W_VISION, W_KB = _get_adaptive_weights()

    # Collect signals from all sources
    all_signals: dict[str, dict] = {}  # province → accumulated

    def _add(province: str, score: float, source: str, detail: dict | None = None):
        if province in excluded:
            return
        if province not in all_signals:
            all_signals[province] = {"score": 0.0, "signals": [], "city": None}
        all_signals[province]["score"] += score
        all_signals[province]["signals"].append({
            "source": source,
            "score": round(score, 4),
            "direction": "support",
            "locality": "city" if (detail or {}).get("city") else "province",
            "reliability": round(min(1.0, max(0.0, score)), 4),
            **(detail or {}),
        })

    # 1. GeoCLIP
    geo = _extract_geo_signals(geoclip_result, W_GEO)
    for prov, info in geo.items():
        _add(prov, info["score"], "geoclip", {"prob": info["prob"]})
        # Use first GeoCLIP match as city hint
        if prov in all_signals and all_signals[prov].get("city") is None and info.get("city"):
            all_signals[prov]["city"] = info.get("city")

    # 2. CLIP
    clip = _extract_clip_signals(clip_result, W_CLIP)
    for prov, info in clip.items():
        _add(prov, info["score"], "clip", {"similarity": info["similarity"]})

    # 3. OCR
    ocr = _extract_ocr_signals(ocr_data, W_OCR)
    for key, info in ocr.items():
        prov = info["province"]
        _add(prov, info["score"], info["source"])
        if info["source"] == "license_plate" and ocr_data:
            for plate in ocr_data.get("license_plates", []):
                pn = normalize_province_name(plate.get("province", ""))
                if pn == prov:
                    city = plate.get("city", "")
                    if city and city != "未知" and prov in all_signals:
                        all_signals[prov]["city"] = city

    # 4. Qwen Vision
    vision = _extract_vision_signals(clues, vision_region, W_VISION)
    for key, info in vision.items():
        _add(info["province"], info["score"], info["source"])

    # 5. Knowledge Base
    kb = _extract_kb_signals(clues, W_KB)
    for key, info in kb.items():
        _add(info["province"], info["score"], info["source"])

    # 6. City Fingerprints
    city_signals = _extract_city_signals(clues, vision_raw, W_VISION)
    for key, info in city_signals.items():
        _add(info["province"], info["score"], info["source"])
        if info.get("city") and info["province"] in all_signals and not all_signals[info["province"]].get("city"):
            all_signals[info["province"]]["city"] = info["city"]

    # Build ranked list
    candidates = []
    calibrator = load_calibrator(get_settings().calibration_path)
    for province, data in all_signals.items():
        # Clamp score to [0, 0.98]
        score = min(data["score"], 0.98)
        # Sort signals by contribution
        data["signals"].sort(key=lambda s: s["score"], reverse=True)
        calibrated_score = calibrator.predict(score) if calibrator else None
        candidates.append({
            "province": province,
            "city": data.get("city"),
            "score": round(calibrated_score if calibrated_score is not None else score, 4),
            "raw_score": round(score, 4),
            "confidence_kind": "calibrated" if calibrator else "ranking_score",
            "signals": data["signals"],
        })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


def score_candidates_with_priorities(
    geoclip_result: dict | None = None,
    clip_result: dict | None = None,
    ocr_data: dict | None = None,
    clues: dict | None = None,
    vision_region: str | None = None,
    vision_raw: str = "",
    excluded_provinces: list[str] | None = None,
    top_k: int = 20,
) -> tuple[list[dict], dict]:
    """Score candidates and return both the ranked list and a summary dict.

    Returns:
        (candidates, summary) where summary contains:
        - ranked_candidates: top_k list
        - primary_province: highest scoring
        - secondary_provinces: next 2
        - top_signal_sources: which sources contributed most
    """
    candidates = score_candidates(
        geoclip_result, clip_result, ocr_data, clues, vision_region, vision_raw, excluded_provinces,
    )

    top = candidates[:top_k] if candidates else []

    # Analyze which sources contributed most to the top candidate
    top_sources = []
    if top:
        sources = {}
        for s in top[0].get("signals", []):
            src = s["source"]
            sources[src] = sources.get(src, 0) + s["score"]
        top_sources = sorted(sources.items(), key=lambda x: x[1], reverse=True)

    summary = {
        "ranked_candidates": top,
        "primary_province": top[0]["province"] if top else None,
        "secondary_provinces": [c["province"] for c in top[1:3]] if len(top) >= 3 else [],
        "top_signal_sources": top_sources,
        "total_scored": len(candidates),
    }

    return candidates, summary
