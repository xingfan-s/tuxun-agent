import time
import json
import contextvars
from app.utils.logging import structlog, redact_sensitive
import asyncio
from openai import OpenAI
from langgraph.graph import END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from app.config import get_settings
from app.agent.state import AgentState
from app.agent.prompts import (
    VISION_MACRO_PROMPT, VISION_DETAIL_PROMPT,
    CLUE_EXTRACTION_PROMPT, REACT_SYSTEM_PROMPT,
    RESULT_SYNTHESIS_PROMPT, ADVERSARIAL_VERIFY_PROMPT,
)
from app.safety import run_safety_check
from app.tools.exif import extract_exif
from app.tools.search import search_place
from app.tools.landmark import search_landmark
from app.tools.reverse_image import reverse_image_search, reverse_image_search_async
from app.tools.ocr_china import extract_china_clues
from app.tools.map import create_map_service
from app.utils.image import encode_image_for_ocr
from app.tools.geoclip import predict_location, get_load_status as geoclip_load_status
from app.tools.clip_search import search_similar_images, search_similar_images_tool
from app.tools.china_knowledge import (
    get_china_vegetation_lat_ranges,
    get_china_architecture_region_rules,
    get_china_script_region_rules,
    get_china_climate_rules,
    normalize_province_name,
    contains_province,
    get_clue_reliability_weights,
    generate_search_strategy,
    update_hypothesis_score,
    get_provinces_for_region,
)
from app.geolocation.ranker import score_candidates_with_priorities
from app.geolocation.coordinates import validate_coordinate
from app.tools.base import ToolBudget

logger = structlog.get_logger()
_LLM_USAGE: contextvars.ContextVar[dict | None] = contextvars.ContextVar("llm_usage", default=None)

_VALID_PROVINCES = {
    "北京", "天津", "上海", "重庆", "河北", "山西", "内蒙古", "辽宁", "吉林", "黑龙江",
    "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南", "湖北", "湖南", "广东",
    "广西", "海南", "四川", "贵州", "云南", "西藏", "陕西", "甘肃", "青海", "宁夏",
    "新疆", "台湾", "香港", "澳门",
}


def _canonical_province_name(value: object) -> str | None:
    """Return a standard province short name, rejecting narrative text."""
    text = str(value or "").strip()
    if not text:
        return None
    normalized = normalize_province_name(text)
    if normalized in _VALID_PROVINCES:
        return normalized
    matches = [name for name in _VALID_PROVINCES if name in text]
    return matches[0] if len(matches) == 1 else None


def _sanitize_hypotheses(hypotheses: list[dict], selected_province: object,
                         selected_score: object = None) -> list[dict]:
    """Validate, merge and order province hypotheses for result display."""
    merged: dict[str, dict] = {}
    for hypothesis in hypotheses or []:
        province = _canonical_province_name(hypothesis.get("province"))
        if not province:
            continue
        candidate = dict(hypothesis)
        candidate["province"] = province
        existing = merged.get(province)
        if existing is None:
            merged[province] = candidate
            continue
        if float(candidate.get("score", 0) or 0) > float(existing.get("score", 0) or 0):
            existing.update(candidate)
        existing["supporting_evidence"] = (
            list(existing.get("supporting_evidence", []))
            + list(candidate.get("supporting_evidence", []))
        )
        existing["contradicting_evidence"] = (
            list(existing.get("contradicting_evidence", []))
            + list(candidate.get("contradicting_evidence", []))
        )

    selected = _canonical_province_name(selected_province)
    if selected:
        selected_value = float(selected_score or 0)
        candidate = merged.setdefault(selected, {
            "province": selected,
            "city": None,
            "score": selected_value,
            "supporting_evidence": [],
            "contradicting_evidence": [],
            "source": "final_result",
        })
        candidate["score"] = max(float(candidate.get("score", 0) or 0), selected_value)

    return sorted(
        merged.values(),
        key=lambda item: (
            item.get("province") == selected,
            float(item.get("score", 0) or 0),
        ),
        reverse=True,
    )


def _estimate_tokens(text: str | None) -> int:
    """Conservative fallback when a provider omits usage in streaming mode."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def _record_usage(state: AgentState, text: str | None, model: str | None = None) -> None:
    usage_hint = _LLM_USAGE.get()
    _LLM_USAGE.set(None)
    tokens = int((usage_hint or {}).get("total_tokens") or (usage_hint or {}).get("completion_tokens") or _estimate_tokens(text))
    state["tokens_used"] = int(state.get("tokens_used", 0)) + tokens
    state["model_calls"] = int(state.get("model_calls", 0)) + 1
    model_name = model or get_settings().qwen_model
    model_usage = state.setdefault("model_usage", {})
    entry = model_usage.setdefault(model_name, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    entry["calls"] += 1
    entry["prompt_tokens"] += int((usage_hint or {}).get("prompt_tokens", 0) or 0)
    entry["completion_tokens"] += int((usage_hint or {}).get("completion_tokens", tokens) or tokens)
    entry["total_tokens"] += int((usage_hint or {}).get("total_tokens", tokens) or tokens)

TOOL_MAP = {
    "search_place": search_place,
    "search_landmark": search_landmark,
    "reverse_image_search": reverse_image_search,
    "extract_china_clues": extract_china_clues,
    "search_similar_images": search_similar_images_tool,
}

MAP_TOOLS = {"geocode", "reverse_geocode", "search_nearby"}

TOOL_TIMEOUTS = {
    "search_place": 15, "geocode": 15, "reverse_geocode": 15,
    "search_nearby": 20,
    "search_landmark": 15, "reverse_image_search": 25,
    "extract_china_clues": 30,
    "search_similar_images": 20,
}


async def push_step(state: AgentState, step_num: int, step_type: str,
                     label: str, status: str, data: dict, elapsed_ms: int, progress: int):
    """Push a step update via SSE callback."""
    cb = state.get("stream_callback")
    if cb:
        public_data = dict(data)
        if step_type == "vision_detail":
            public_data.pop("raw_output", None)
        if step_type == "reasoning":
            action = public_data.get("action", "analysis")
            public_data.pop("thought", None)
            public_data["summary"] = {
                "redirect": "正在修正候选方向",
                "final_answer": "候选验证完成",
            }.get(action, "正在评估候选证据")
        await cb({
            "event": "step_update",
            "data": {
                "step": step_num, "type": step_type, "label": label,
                "status": status, "data": public_data, "elapsed_ms": elapsed_ms,
            }
        })
        await cb({"event": "progress", "data": {"progress": progress}})


def _get_llm_client():
    settings = get_settings()
    return OpenAI(api_key=settings.qwen_api_key, base_url=settings.qwen_base_url, timeout=120.0)


_ROLE_MAP = {"human": "user", "ai": "assistant", "system": "system"}


async def _llm_chat(messages: list, model: str = None, temperature: float = 0.1,
                   max_tokens: int = 2000, stream_callback=None) -> str:
    settings = get_settings()

    formatted = []
    for m in messages:
        if isinstance(m, dict):
            formatted.append(m)
        else:
            role = _ROLE_MAP.get(getattr(m, "type", None), "user")
            formatted.append({"role": role, "content": m.content})

    # ---- Streaming path (v2.1) ----
    if stream_callback:
        try:
            from openai import AsyncOpenAI
            aclient = AsyncOpenAI(
                api_key=settings.qwen_api_key,
                base_url=settings.qwen_base_url,
                timeout=120.0,
            )
            stream = await aclient.chat.completions.create(
                model=model or settings.qwen_model,
                messages=formatted,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            full_text: list[str] = []
            token_count = 0
            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    full_text.append(delta.content)
                    token_count += 1
                    # Never send raw model output to the browser. The UI only
                    # receives a coarse stage summary while full text stays in
                    # the agent state.
                    if token_count == 1:
                        await stream_callback({
                            "event": "reasoning_summary",
                            "data": {"text": "正在整理候选证据", "phase": "model"},
                        })
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage:
                    _LLM_USAGE.set({
                        "prompt_tokens": getattr(chunk_usage, "prompt_tokens", 0),
                        "completion_tokens": getattr(chunk_usage, "completion_tokens", 0),
                        "total_tokens": getattr(chunk_usage, "total_tokens", 0),
                    })
            if full_text:
                await stream_callback({
                    "event": "reasoning_summary",
                    "data": {
                        "text": "模型分析阶段完成",
                        "phase": "model",
                        "tokens_estimate": _estimate_tokens("".join(full_text)),
                    },
                })
            return "".join(full_text)
        except Exception as e:
            _log_openai_error("llm_stream_error", e, model or settings.qwen_model)
            logger.warning("llm_stream_fallback", error=str(e)[:100])
            # Fall through to non-streaming path

    # ---- Non-streaming path (original) ----
    client = _get_llm_client()

    def _call():
        response = client.chat.completions.create(
            model=model or settings.qwen_model,
            messages=formatted,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        usage = getattr(response, "usage", None)
        if usage:
            _LLM_USAGE.set({
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            })
        return response.choices[0].message.content

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_call), timeout=60,
        )
    except asyncio.TimeoutError:
        logger.error("llm_chat_timeout", model=model or settings.qwen_model)
        raise
    except Exception as e:
        _log_openai_error("llm_chat_error", e, model or settings.qwen_model)
        raise


async def _llm_vision(image_base64: str, prompt: str, max_tokens: int = 2000) -> str:
    settings = get_settings()
    client = _get_llm_client()

    def _call():
        response = client.chat.completions.create(
            model=settings.qwen_vl_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                ]
            }],
            temperature=0.1,
            max_tokens=max_tokens,
        )
        usage = getattr(response, "usage", None)
        if usage:
            _LLM_USAGE.set({
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            })
        return response.choices[0].message.content

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_call), timeout=90,
        )
    except asyncio.TimeoutError:
        logger.error("llm_vision_timeout", model=settings.qwen_vl_model)
        raise
    except Exception as e:
        _log_openai_error("llm_vision_error", e, settings.qwen_vl_model)
        raise


def _log_openai_error(event: str, error: Exception, model: str):
    """Log OpenAI errors with full API details when available."""
    try:
        from openai import APIError
        if isinstance(error, APIError):
            sc = getattr(error, 'status_code', None)
            logger.error(event, model=model, status_code=sc,
                        error_type=type(error).__name__, message=redact_sensitive(error))
            return
    except ImportError:
        pass
    logger.error(event, model=model, error=redact_sensitive(error))


# ============================================================
# Node 0: Safety Check
# ============================================================

async def safety_check_node(state: AgentState) -> AgentState:
    t0 = time.time()
    result = await asyncio.to_thread(run_safety_check, state["image_base64"])
    elapsed = int((time.time() - t0) * 1000)

    await push_step(state, 0, "safety_check", "安全预检",
                    "done", result, elapsed, 3)

    state["safety_passed"] = result["passed"]
    state["safety_reason"] = result.get("reason")
    return state


# ============================================================
# Node 1: EXIF Extraction
# ============================================================

async def exif_extract_node(state: AgentState) -> AgentState:
    t0 = time.time()
    exif_data = await asyncio.to_thread(extract_exif.invoke, {"image_path": state["image_path"]})
    elapsed = int((time.time() - t0) * 1000)

    await push_step(state, 1, "exif", "EXIF 提取",
                    "done", exif_data, elapsed, 8)

    state["exif_data"] = exif_data
    return state


# ============================================================
# Node 2: Vision Macro (方案三: Pass 1 - 快速区域分类)
# ============================================================

async def vision_macro_node(state: AgentState) -> AgentState:
    """Quick regional classification with minimal tokens (max_tokens=50)."""
    t0 = time.time()
    try:
        region = await _llm_vision(state["image_base64"], VISION_MACRO_PROMPT, max_tokens=50)
        region = region.strip()
        _record_usage(state, region, get_settings().qwen_vl_model)
    except Exception as e:
        logger.error("vision_macro_failed", error=str(e))
        region = "无法判断"

    valid_regions = {"东北", "华北", "西北", "西南", "华南", "华东", "华中"}
    region = region if region in valid_regions else "无法判断"

    elapsed = int((time.time() - t0) * 1000)
    state["vision_region"] = region

    await push_step(state, 2, "vision_macro", "宏观区域分类",
                    "done", {"region": region}, elapsed, 12)
    return state


# ============================================================
# Node 2.1: Country Check (when macro region is 无法判断)
# ============================================================



# ============================================================
# Node 2.5: GeoCLIP (v2.0 Phase 1 - image-to-GPS prediction)
# ============================================================

async def geoclip_node(state: AgentState) -> AgentState:
    """Run GeoCLIP to predict GPS coordinates from the image."""
    settings = get_settings()
    t0 = time.time()

    if not settings.geoclip_enabled:
        state["geoclip_result"] = None
        await push_step(state, 6, "geoclip", "GeoCLIP 地理预测",
                        "done", {"status": "disabled"}, 0, 15)
        return state

    try:
        top_coords = await asyncio.to_thread(
            predict_location, state["image_path"], 10,
        )
    except Exception as e:
        logger.warning("geoclip_inference_failed", error=str(e))
        state["geoclip_result"] = None
        elapsed = int((time.time() - t0) * 1000)
        await push_step(state, 6, "geoclip", "GeoCLIP 地理预测",
                        "done", {"status": "error", "error": str(e)[:100]},
                        elapsed, 15)
        return state

    if not top_coords:
        state["geoclip_result"] = None
        elapsed = int((time.time() - t0) * 1000)
        load_status = geoclip_load_status()
        status = "unavailable" if load_status.startswith("unavailable:") else "no_results"
        await push_step(state, 6, "geoclip", "GeoCLIP 地理预测",
                        "done", {"status": status, "capability": load_status}, elapsed, 15)
        return state

    # Reverse geocode top 5 coordinates concurrently to get province/city
    map_service = create_map_service()

    async def _reverse_one(coord):
        try:
            geo_result = await map_service.reverse_geocode(coord["lat"], coord["lon"])
            if geo_result:
                info = vars(geo_result) if hasattr(geo_result, '__dict__') else geo_result
                province = (info.get("province", "") or
                           info.get("state", "") or
                           info.get("region", ""))
                city = info.get("city", "") or info.get("town", "")
                if province:
                    return {
                        "province": province,
                        "city": city or "",
                        "lat": coord["lat"],
                        "lon": coord["lon"],
                        "probability": coord["probability"],
                    }
        except Exception:
            pass
        return None

    geo_results = await asyncio.gather(*[_reverse_one(c) for c in top_coords[:5]])
    top_provinces = [r for r in geo_results if r is not None]

    elapsed = int((time.time() - t0) * 1000)

    geoclip_result = {
        "top_coords": top_coords[:5],
        "top_provinces": top_provinces,
        "inference_ms": elapsed,
    }
    state["geoclip_result"] = geoclip_result

    await push_step(state, 6, "geoclip", "GeoCLIP 地理预测",
                    "done", {
                        "top_predictions": [
                            f"{p.get('province', '?')} ({p['probability']:.4f})"
                            for p in top_provinces[:5]
                        ],
                        "inference_ms": elapsed,
                    }, elapsed, 15)
    return state


# ============================================================
# Node 2.7: Unified Anchor Pre-Search (CLIP + GeoCLIP + 以图搜图)
# ============================================================

async def anchor_search_node(state: AgentState) -> AgentState:
    """Pre-search around coordinates from 3 sources to provide
    concrete geographic anchors before the ReAct loop.

    Priority: (1) CLIP+FAISS match coords (real GPS),
              (2) GeoCLIP predictions (model GPS, fallback),
              (3) Reverse image search (Bing best_guess geocoded).

    Reverse image search runs in parallel with anchor geocoding.
    """
    settings = get_settings()
    t0 = time.time()
    map_service = create_map_service()

    # ---- Collect coordinate sources ----
    clues = state.get("clues") or {}
    ocr_data = state.get("ocr_data") or {}

    coord_sources: list[dict] = []

    # Source 1: CLIP matches (real GPS from geo-tagged DB)
    clip_result = state.get("clip_result")
    if clip_result and clip_result.get("matches"):
        for i, m in enumerate(clip_result["matches"][:3]):
            if m.get("lat") is not None and m.get("lon") is not None:
                coord_sources.append({
                    "lat": m["lat"], "lon": m["lon"],
                    "priority": 1, "source": f"CLIP相似图{i+1}(dist={m.get('distance', 0):.1f})",
                    "city": m.get("city", ""),
                })

    # Source 2: GeoCLIP (fallback if CLIP < 2 anchors)
    geoclip_result = state.get("geoclip_result")
    if len(coord_sources) < 3 and geoclip_result and geoclip_result.get("top_coords"):
        remaining = 3 - len(coord_sources)
        for c in geoclip_result["top_coords"][:remaining]:
            coord_sources.append({
                "lat": c["lat"], "lon": c["lon"],
                "priority": 2, "source": f"GeoCLIP(prob={c.get('probability', 0):.3f})",
            })

    # Source 3: OCR definitive clues — add city geocode anchors
    # (don't discard other anchors, just enrich the set — LLM will judge)
    ocr_province = ocr_city = None
    for plate in ocr_data.get("license_plates", []):
        p = plate.get("province", "")
        c = plate.get("city", "")
        if p and p != "未知":
            ocr_province = p
            ocr_city = c if c != "未知" else None
            break
    if not ocr_province:
        for phone in ocr_data.get("phone_area_codes", []):
            p = phone.get("province", "")
            c = phone.get("city", "")
            if p and p != "未知":
                ocr_province = p
                ocr_city = c if c != "未知" else None
                break

    if ocr_city:
        coord_sources.append({
            "lat": None, "lon": None,  # geocode will resolve
            "priority": 0, "source": f"OCR锁定({ocr_city})",
            "geocode_target": ocr_city,
        })
    elif ocr_province:
        coord_sources.append({
            "lat": None, "lon": None,
            "priority": 0, "source": f"OCR锁定({ocr_province})",
            "geocode_target": ocr_province,
        })

    # ---- Build architecture-aware POI keywords per anchor ----
    arch = clues.get("architecture", "")

    def _poi_keywords_for_clues() -> list[str]:
        """Return prioritized POI search keywords based on visual/OCR clues."""
        keywords = ["景点"]  # default
        if arch:
            if "徽派" in arch or "马头墙" in arch:
                keywords = ["古村落", "徽派建筑", "古镇"]
            elif "土楼" in arch:
                keywords = ["土楼", "客家围屋"]
            elif "窑洞" in arch:
                keywords = ["窑洞", "黄土高原"]
            elif "骑楼" in arch:
                keywords = ["骑楼老街", "老城区"]
            elif "藏式" in arch or "碉房" in arch:
                keywords = ["藏式建筑", "寺庙", "雪山"]
            elif "傣式" in arch or "竹楼" in arch or "干栏式" in arch:
                keywords = ["傣族园", "热带植物园"]
            elif "江南" in arch or "白墙" in arch:
                keywords = ["水乡古镇", "园林"]
            elif "俄式" in arch or "洋葱头" in arch:
                keywords = ["俄式建筑", "教堂"]
            elif "石库门" in arch or "里弄" in arch:
                keywords = ["石库门", "老弄堂", "老城厢"]
        if ocr_city:
            keywords.insert(0, ocr_city)
        return keywords

    # ---- Anchor geocoding helper ----
    poi_keywords = _poi_keywords_for_clues()

    async def _search_one_anchor(idx: int, cs: dict) -> dict | None:
        lat, lon = cs["lat"], cs["lon"]

        # OCR anchor: resolve via geocode first
        geo_target = cs.get("geocode_target")
        if geo_target and (lat is None or lon is None):
            try:
                geo_results = await map_service.geocode(geo_target)
                if geo_results:
                    g = vars(geo_results[0])
                    lat = g.get("lat")
                    lon = g.get("lng") if g.get("lng") is not None else g.get("lon")
                    if lat is None or lon is None:
                        return None
                else:
                    return None
            except Exception:
                return None

        try:
            geo = await map_service.reverse_geocode(lat, lon)
            if geo is None:
                return None
            info = vars(geo) if hasattr(geo, '__dict__') else geo
            province = info.get("province", "") or info.get("state", "")
            city = info.get("city", "") or info.get("town", "") or cs.get("city", "")
            district = info.get("district", "") or info.get("county", "")
            display = info.get("display_name", "")

            # Use architecture-aware POI keywords for better anchor context
            for kw in poi_keywords[:2]:
                pois = await map_service.search_nearby(lat, lon, keyword=kw, radius=5000)
                poi_names = [p.name for p in pois[:5]] if pois else []
                if poi_names:
                    break

            urban_pois = await map_service.search_nearby(lat, lon, keyword="广场", radius=3000)
            urban_names = [p.name for p in urban_pois[:3]] if urban_pois else []

            return {
                "rank": idx + 1, "lat": lat, "lon": lon,
                "priority": cs["priority"], "source": cs["source"],
                "province": province, "city": city,
                "district": district, "display_name": display,
                "landmarks": poi_names, "urban_pois": urban_names,
            }
        except Exception as e:
            logger.warning("anchor_search_failed", idx=idx, error=str(e))
            return None

    # ---- Reverse image search -> extract location hints ----
    async def _reverse_image_locations() -> list[dict]:
        """Run reverse image search and extract geocodable location hints.

        Filters results for location relevance: keeps scenic spots, landmarks,
        travel destinations, government/edu domains; discards shopping, social
        media, generic product pages.
        """
        import re as _re

        def _location_score(r: dict) -> int:
            """Score a search result for location relevance (0-10).
            Used to filter out non-location results before geocoding.
            """
            score = 0
            src = r.get("source", "")
            title = r.get("title", "").strip()
            snippet = r.get("snippet", "").strip()
            combined = f"{title} {snippet}".lower()

            # Structured location data from Bing → inherently high relevance
            if src in ("bing_best_guess", "bing_travel", "bing_attraction"):
                return 10
            if src == "bing_entity":
                return 8  # Bing entities are usually well-categorized

            # Location keywords in title/snippet
            loc_kw = ["市", "省", "县", "区", "镇", "景区", "景点", "公园", "广场",
                      "旅游", "旅行", "攻略", "游记", "风景", "地标", "名胜",
                      "scenic", "landmark", "travel", "tourism", "destination",
                      "temple", "pagoda", "mountain", "lake", "river", "bridge"]
            score += sum(2 for kw in loc_kw if kw in combined)

            # Travel/photo domains
            travel_domains = ["mafengwo", "ctrip", "qunar", "tripadvisor", "travel",
                            "flickr", "panoramio", "500px", "tuchong", "lofter",
                            "lvyou", "youji", "photography"]
            url = r.get("url", "").lower()
            score += sum(3 for d in travel_domains if d in url)

            # Government/edu → often official location pages
            if ".gov.cn" in url or ".edu.cn" in url:
                score += 3

            # Penalize non-location content
            non_loc = ["价格", "购买", "淘宝", "京东", "shop", "price", "buy",
                      "social", "facebook", "twitter", "weibo", "微信",
                      "video", "movie", "music", "game"]
            score -= sum(2 for kw in non_loc if kw in combined)

            # Generic similar-image results from Google Lens / Bing visual
            # → only keep if they mention specific locations
            if src in ("google_lens", "google_lens_related",
                      "bing_visual", "bing_pages", "yandex"):
                # These engines return visually similar images — need location text
                if score < 4:
                    return 0

            return max(0, score)

        locations: list[tuple[str, str]] = []  # (text, source_label)
        try:
            results = await reverse_image_search_async(state["image_base64"])
            for r in results:
                src = r.get("source", "")
                title = r.get("title", "").strip()
                snippet = r.get("snippet", "").strip()

                # Score location relevance
                rel = _location_score(r)
                if rel < 3 and src not in ("bing_best_guess", "bing_travel",
                                           "bing_attraction", "bing_entity"):
                    continue  # skip non-location results

                text = ""
                label = ""

                if src == "bing_best_guess":
                    text = title.replace("Bing 推测: ", "").strip()
                    label = "Bing推测"
                elif src == "bing_travel":
                    text = title.replace("目的地: ", "").strip()
                    label = "Bing旅行"
                elif src == "bing_attraction":
                    text = title.replace("景点: ", "").strip()
                    label = "Bing景点"
                elif src == "bing_entity":
                    name = title.strip()
                    if name and len(name) >= 2:
                        skip = {"tree", "car", "food", "person", "animal", "building"}
                        if not any(kw in name.lower() for kw in skip):
                            text = name
                            label = "Bing实体"
                elif src in ("baidu_exact", "baidu_similar"):
                    m = _re.search(r'[\u4e00-\u9fff]{2,6}(?:市|省|县|区|镇|景区|公园|广场)', title)
                    if m:
                        text = m.group(0)
                        label = "百度匹配"
                elif src in ("google_lens", "google_lens_related"):
                    # Google Lens results: extract Chinese place names
                    m = _re.search(r'[\u4e00-\u9fff]{2,6}(?:市|省|县|区|镇|景区|公园|广场|旅游|景点)', combined=f"{title} {snippet}")
                    if m:
                        text = m.group(0)
                        label = "GoogleLens"
                elif src in ("bing_visual", "bing_pages"):
                    m = _re.search(r'[\u4e00-\u9fff]{2,8}(?:市|省|县|区|景区|公园|广场)', title + " " + snippet)
                    if m:
                        text = m.group(0)
                        label = "Bing相似"

                if text and len(text) >= 2:
                    locations.append((text, label))
        except Exception as e:
            logger.warning("reverse_image_anchor_failed", error=str(e))
            return []

        # Geocode each location hint (dedup by text)
        geocoded = []
        seen = set()
        for loc, label in locations:
            if loc in seen:
                continue
            seen.add(loc)
            if len(geocoded) >= 8:  # collect up to 8 before consensus merge
                break
            try:
                geo_results = await map_service.geocode(loc)
                if geo_results:
                    g = vars(geo_results[0])
                    lat = g.get("lat")
                    lon = g.get("lng") if g.get("lng") is not None else g.get("lon")
                    if lat is not None and lon is not None:
                        province = g.get("province", "") or g.get("state", "")
                        city = g.get("city", "") or g.get("town", "")
                        pois = await map_service.search_nearby(lat, lon, keyword="景点", radius=5000)
                        poi_names = [p.name for p in pois[:3]] if pois else []
                        geocoded.append({
                            "rank": 0, "lat": lat, "lon": lon,
                            "priority": 3, "source": f"以图搜图({label})->{loc}",
                            "province": province, "city": city,
                            "district": "", "display_name": g.get("display_name", loc),
                            "landmarks": poi_names, "urban_pois": [],
                        })
            except Exception:
                pass

        # ---- Multi-source overlap annotation (no auto-upgrade) ----
        # Group by (province, city) and annotate each anchor with how many
        # independent sources point to this same location. The LLM can judge
        # whether 2/4 or 3/4 consensus is meaningful vs coincidence.
        groups: dict[tuple[str, str], list[dict]] = {}
        for a in geocoded:
            key = (a["province"], a["city"]) if a["city"] else (a["province"], "")
            groups.setdefault(key, []).append(a)

        total_sources = len(groups)
        for key, items in groups.items():
            engines = []
            seen_src = set()
            for item in items:
                src = item["source"].replace("以图搜图(", "").replace(")", "").split("->")[0]
                if src not in seen_src:
                    engines.append(src)
                    seen_src.add(src)
            source_count = len(engines)
            # Annotate each anchor with consensus context
            suffix = f" | 搜图共{total_sources}源, {source_count}源指向此地({', '.join(engines)})"
            for item in items:
                item["source"] += suffix

        return geocoded[:5]  # cap at 5 anchors

    # ---- Early exit: no coordinates at all ----
    if not coord_sources:
        state["geoclip_anchors"] = None
        await push_step(state, 8, "anchor_search", "多源锚点预搜",
                        "done", {"status": "no_coords"}, 0, 18)
        return state

    # ---- Execute: geocode anchors + reverse image search in parallel ----
    geocode_anchors = await asyncio.gather(*[
        _search_one_anchor(i, cs) for i, cs in enumerate(coord_sources[:5])
    ])
    reverse_locations = await _reverse_image_locations()

    # Merge: CLIP/GeoCLIP first, then reverse image locations
    anchors = [a for a in geocode_anchors if a is not None]
    valid_anchors = anchors + reverse_locations
    for i, a in enumerate(valid_anchors):
        a["rank"] = i + 1

    state["geoclip_anchors"] = valid_anchors

    elapsed = int((time.time() - t0) * 1000)

    summary_lines = []
    for a in valid_anchors:
        prio = ["", "CLIP", "GeoCLIP", "搜图"][a.get("priority", 1)]
        loc = (f"{a.get('city') or a.get('province', '?')}"
               if (a.get('city') or a.get('province'))
               else f"({a['lat']:.3f},{a['lon']:.3f})")
        pois = a.get("landmarks", [])[:3]
        summary_lines.append(
            f"[{prio}] {loc}" + (f" -> {' / '.join(pois)}" if pois else "")
        )

    await push_step(state, 8, "anchor_search", "多源锚点预搜",
                    "done", {
                        "anchors": valid_anchors,
                        "sources": {
                            "clip": sum(1 for a in valid_anchors if a.get("priority") == 1),
                            "geoclip": sum(1 for a in valid_anchors if a.get("priority") == 2),
                            "reverse_image": len(reverse_locations),
                        },
                        "summary": summary_lines,
                        "count": len(valid_anchors),
                        "elapsed_ms": elapsed,
                    }, elapsed, 18)
    return state


# ============================================================
# Node 2.7: CLIP + FAISS Similar Image Search (v2.0 Phase 2)
# ============================================================

async def clip_search_node(state: AgentState) -> AgentState:
    """Search for visually similar geo-tagged images via CLIP + FAISS."""
    settings = get_settings()
    t0 = time.time()

    if not settings.clip_search_enabled:
        state["clip_result"] = None
        await push_step(state, 7, "clip_search", "CLIP 相似图检索",
                        "done", {"status": "disabled"}, 0, 17)
        return state

    # Skip if database is empty — avoids loading CLIP model uselessly
    from app.tools.clip_search import get_db
    try:
        db = get_db()
    except Exception as exc:
        state["clip_result"] = None
        await push_step(state, 7, "clip_search", "CLIP 相似图检索",
                        "done", {"status": "unavailable", "error": type(exc).__name__},
                        int((time.time() - t0) * 1000), 17)
        return state
    if db.count() == 0:
        state["clip_result"] = None
        await push_step(state, 7, "clip_search", "CLIP 相似图检索",
                        "done", {"status": "skipped", "reason": "empty_db"}, 0, 17)
        return state

    clip_result = await asyncio.to_thread(
        search_similar_images, state["image_path"], 5,
    )

    elapsed = int((time.time() - t0) * 1000)

    if clip_result is None:
        state["clip_result"] = None
        await push_step(state, 7, "clip_search", "CLIP 相似图检索",
                        "done", {"status": "error", "db_size": 0},
                        elapsed, 17)
        return state

    # Extract province info from top matches via reverse geocode
    matches = clip_result.get("matches", [])
    clip_provinces = []
    if matches:
        map_service = create_map_service()

        async def _reverse_match(match):
            try:
                geo = await map_service.reverse_geocode(match["lat"], match["lon"])
                if geo:
                    info = vars(geo) if hasattr(geo, '__dict__') else geo
                    return {
                        "province": info.get("province", ""),
                        "city": info.get("city", "") or info.get("town", ""),
                        "lat": match["lat"],
                        "lon": match["lon"],
                        "distance": match["distance"],
                        "similarity": match.get("similarity"),
                        "source": match.get("source", ""),
                    }
            except Exception:
                pass
            return None

        geo_results = await asyncio.gather(*[_reverse_match(m) for m in matches[:5]])
        clip_provinces = [r for r in geo_results if r is not None and r.get("province")]

    clip_result["top_provinces"] = clip_provinces
    state["clip_result"] = clip_result

    await push_step(state, 7, "clip_search", "CLIP 相似图检索",
                    "done", {
                        "matches_found": len(matches),
                        "db_size": clip_result.get("db_size", 0),
                        "top_match": matches[0] if matches else None,
                        "search_ms": clip_result.get("search_ms", 0),
                    }, elapsed, 17)
    return state


# ============================================================
# Node 4: Vision Detail (方案三: Pass 2 - 区域感知详细分析)
# ============================================================

async def vision_detail_node(state: AgentState) -> AgentState:
    """Region-aware blind visual analysis without retrieval anchoring."""
    t0 = time.time()
    macro_region = state.get("vision_region", "无法判断")

    prompt = VISION_DETAIL_PROMPT.format(macro_region=macro_region,
                                         geo_context="")
    try:
        vision_raw = await _llm_vision(state["image_base64"], prompt)
        _record_usage(state, vision_raw, get_settings().qwen_vl_model)
    except Exception as e:
        logger.error("vision_detail_failed", error=str(e))
        vision_raw = f"视觉分析暂时不可用。宏观区域: {macro_region}"

    elapsed = int((time.time() - t0) * 1000)

    await push_step(state, 3, "vision_detail", f"详细视觉分析（{macro_region}）",
                    "done", {"description": vision_raw[:200] + "...", "raw_output": vision_raw},
                    elapsed, 25)
    state["vision_raw"] = vision_raw
    return state


# ============================================================
# Node 4: Clue Extraction
# ============================================================

async def clue_extract_node(state: AgentState) -> AgentState:
    t0 = time.time()
    prompt = CLUE_EXTRACTION_PROMPT.format(vision_raw=state["vision_raw"])
    try:
        response = await _llm_chat([HumanMessage(content=prompt)], temperature=0.1, max_tokens=1000)
        _record_usage(state, response)
    except Exception as e:
        logger.error("clue_extract_failed", error=str(e))
        response = '{"raw": "线索提取暂时不可用", "parse_error": false, "top_3_clues": []}'

    try:
        clues = json.loads(response)
    except json.JSONDecodeError:
        clues = {"raw": response, "parse_error": True}

    elapsed = int((time.time() - t0) * 1000)

    await push_step(state, 4, "clue_extraction", "线索提取",
                    "done", {
                        "clues": clues,
                        "top_clues": clues.get("top_3_clues", []) if isinstance(clues, dict) else [],
                    }, elapsed, 35)
    state["clues"] = clues
    return state


# ============================================================
# Node 5: OCR + Context Fusion (方案五)
# ============================================================

async def ocr_extract_node(state: AgentState) -> AgentState:
    """Extract OCR independently so retrieval priors cannot influence it."""
    t0 = time.time()

    def _extract():
        ocr_base64 = encode_image_for_ocr(state["image_path"])
        return extract_china_clues.invoke({"image_base64": ocr_base64})

    ocr_data = await asyncio.to_thread(_extract)
    elapsed = int((time.time() - t0) * 1000)

    await push_step(state, 5, "ocr", "OCR 文字提取",
                    "done", {
                        "text_count": ocr_data.get("text_count", 0),
                        "plates": len(ocr_data.get("license_plates", [])),
                        "area_codes": len(ocr_data.get("phone_area_codes", [])),
                        "highways": len(ocr_data.get("highways", [])),
                        "summary": ocr_data.get("geolocation_summary", ""),
                    }, elapsed, 45)

    state["ocr_data"] = ocr_data
    return state


async def ocr_fusion_node(state: AgentState) -> AgentState:
    """Merge completed OCR into visual clues after both branches finish."""
    ocr_data = state.get("ocr_data") or {}

    # Merge OCR into clues
    clues = state.get("clues", {}) or {}
    if ocr_data.get("license_plates"):
        clues["license_plates"] = ocr_data["license_plates"]
    if ocr_data.get("phone_area_codes"):
        clues["phone_area_codes"] = ocr_data["phone_area_codes"]
    if ocr_data.get("highways"):
        clues["highways"] = ocr_data["highways"]
    if ocr_data.get("all_text"):
        clues["ocr_text"] = ocr_data["all_text"][:30]
    state["clues"] = clues

    # ---- 方案五: OCR-Context Fusion ----
    fused_queries = _fuse_ocr_with_context(ocr_data, state.get("vision_region"), clues)
    state["ocr_fused_queries"] = fused_queries

    if fused_queries:
        await push_step(state, 5, "ocr_fusion", "OCR上下文融合",
                        "done", {
                            "fused_queries": [q.get("query", "") for q in fused_queries[:5]],
                            "fusion_strategy": f"基于视觉区域({state.get('vision_region', '未知')})消歧",
                        }, 0, 48)
    return state


async def independent_signals_node(state: AgentState) -> AgentState:
    """Run independent, potentially expensive signals concurrently."""
    nodes = (
        ("exif", exif_extract_node),
        ("vision_macro", vision_macro_node),
        ("ocr", ocr_extract_node),
        ("geoclip", geoclip_node),
        ("clip", clip_search_node),
    )

    async def _run(name, node):
        try:
            await node(state)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("independent_signal_failed", signal=name, error_type=type(exc).__name__)
            await push_step(state, 2, name if name != "clip" else "clip_search",
                            f"{name} signal", "error",
                            {"status": "unavailable", "error_type": type(exc).__name__}, 0, 15)

    await asyncio.gather(*(_run(name, node) for name, node in nodes))
    return state


def _fuse_ocr_with_context(ocr_data: dict, vision_region: str | None,
                           clues: dict) -> list[dict]:
    """方案五: Generate region-constrained search queries from OCR text."""
    queries = []
    all_text = ocr_data.get("all_text", [])

    candidate_provinces = []
    if vision_region and vision_region != "无法判断":
        candidate_provinces = get_provinces_for_region(vision_region)

    plates = ocr_data.get("license_plates", [])
    phones = ocr_data.get("phone_area_codes", [])

    ocr_province = None
    ocr_city = None
    ocr_conf = 0.0
    if plates and plates[0].get("province") and plates[0].get("province") != "未知":
        ocr_province = plates[0]["province"]
        c = plates[0].get("city", "")
        ocr_city = c if c != "未知" else None
        ocr_conf = plates[0].get("ocr_confidence", 0.0)
    elif phones and phones[0].get("province") and phones[0].get("province") != "未知":
        ocr_province = phones[0]["province"]
        c = phones[0].get("city", "")
        ocr_city = c if c != "未知" else None

    # Don't bias search queries if OCR confidence is low
    _ocr_reliable = ocr_conf >= 0.75 if ocr_conf > 0 else True  # True if unknown

    meaningful = [t.strip() for t in all_text if len(t.strip()) >= 3
                  and not t.strip().isdigit()
                  and not t.strip().replace(".", "").replace("-", "").isdigit()]

    for text in meaningful[:8]:
        query = text
        if _ocr_reliable and ocr_city:
            query = f"{ocr_city} {text}"
        elif _ocr_reliable and ocr_province:
            query = f"{ocr_province} {text}"
        elif candidate_provinces and len(candidate_provinces) <= 3:
            query = f"{' '.join(candidate_provinces)} {text}"

        queries.append({
            "query": query,
            "original_text": text,
            "region_constraint": ocr_city or ocr_province or (
                "、".join(candidate_provinces[:3]) if candidate_provinces else "无"
            ),
            "suggested_tool": "geocode" if (_ocr_reliable and (ocr_city or ocr_province)) or len(text) >= 5
                              else "search_place",
        })

    if _ocr_reliable and ocr_province and not meaningful:
        queries.append({
            "query": ocr_city or ocr_province,
            "original_text": f"OCR定位: {ocr_city or ocr_province}",
            "region_constraint": ocr_province,
            "suggested_tool": "geocode",
        })

    return queries[:6]


def _build_hypotheses_from_candidates(ranked: list[dict],
                                       ocr_data: dict | None) -> list[dict]:
    """Convert ranked candidates into hypothesis dicts for the ReAct loop.

    Scoring is done by the ranker; this function only formats the output.
    OCR definitive lock-in overrides ranking when a full plate match exists.
    """
    hypotheses = []
    # Check if OCR has a definitive lock
    ocr_lock_province = None
    ocr_lock_city = None
    if ocr_data:
        for plate in ocr_data.get("license_plates", []):
            if plate.get("confidence") in ("full_plate",):
                p = normalize_province_name(plate.get("province", ""))
                if p and p != "未知":
                    ocr_lock_province = p
                    c = plate.get("city", "")
                    ocr_lock_city = c if c != "未知" else None
                    break
        if not ocr_lock_province:
            for phone in ocr_data.get("phone_area_codes", []):
                p = normalize_province_name(phone.get("province", ""))
                if p and p != "未知":
                    ocr_lock_province = p
                    c = phone.get("city", "")
                    ocr_lock_city = c if c != "未知" else None
                    break

    for c in ranked:
        province = c["province"]
        score = c["score"]
        signals = c.get("signals", [])
        evidence = [
            {
                "source": s["source"],
                "direction": "support",
                "locality": s.get("locality", "province"),
                "reliability": s.get("reliability", 0.0),
                "raw_score": s["score"],
                "calibrated_contribution": None,
                "summary": f"{s['source']}: {s['score']:.3f}",
            }
            for s in signals
        ]
        city = c.get("city")

        # If OCR locks to a specific province, boost it as strongest single signal
        # but NOT enough to auto-converge alone (needs multi-source consensus).
        if ocr_lock_province and province == ocr_lock_province:
            score = max(score, 0.60)
            city = city or ocr_lock_city
            evidence.insert(0, {
                "source": "ocr",
                "direction": "support",
                "locality": "city" if ocr_lock_city else "province",
                "reliability": 0.90,
                "raw_score": 0.60,
                "calibrated_contribution": None,
                "summary": f"OCR锁定: {ocr_lock_province}" + (f"·{ocr_lock_city}" if ocr_lock_city else ""),
            })

        hypotheses.append({
            "province": province,
            "city": city,
            "score": round(min(score, 0.95), 3),
            "raw_score": c.get("raw_score", score),
            "confidence_kind": c.get("confidence_kind", "ranking_score"),
            "supporting_evidence": evidence,
            "contradicting_evidence": [],
            "source": "candidate_ranker",
            "round_created": 0,
        })

    # Sort by score
    hypotheses.sort(key=lambda h: h["score"], reverse=True)

    # Fallback: if no hypotheses, seed with broad coverage
    if not hypotheses:
        from app.tools.china_knowledge import _get_all_provinces
        for p in _get_all_provinces()[:6]:
            hypotheses.append({
                "province": p, "city": None, "score": 0.1,
                "supporting_evidence": [],
                "contradicting_evidence": [],
                "source": "fallback",
                "round_created": 0,
            })

    return hypotheses[:10]


# ============================================================
# Node 6: Search Strategy (方案四: 知识库前移)
# ============================================================

async def search_strategy_node(state: AgentState) -> AgentState:
    """Generate search strategy from knowledge base (rule-based, no LLM call)."""
    t0 = time.time()

    clues = state.get("clues", {}) or {}
    ocr_data = state.get("ocr_data") or {}
    vision_region = state.get("vision_region")

    geoclip_result = state.get("geoclip_result")
    clip_result = state.get("clip_result")
    strategy = generate_search_strategy(clues, ocr_data, vision_region,
                                        geoclip_result, clip_result)
    state["search_strategy"] = strategy

    # ---- v2.0 Phase 3: Weighted candidate scoring ----
    excluded = strategy.get("excluded_provinces", [])
    all_candidates, candidate_summary = score_candidates_with_priorities(
        geoclip_result, clip_result, ocr_data, clues, vision_region,
        state.get("vision_raw", ""), excluded, top_k=get_settings().react_top_k,
    )
    ranked = candidate_summary["ranked_candidates"]
    strategy["ranked_candidates"] = ranked
    strategy["primary_province"] = candidate_summary["primary_province"]
    strategy["top_signal_sources"] = candidate_summary["top_signal_sources"]

    # Build hypotheses from ranked candidates (formatting only)
    hypotheses = _build_hypotheses_from_candidates(ranked, ocr_data)
    state["hypotheses"] = hypotheses
    state["excluded_provinces"] = excluded

    elapsed = int((time.time() - t0) * 1000)

    await push_step(state, 9, "search_strategy", "搜索策略生成",
                    "done", {
                        "primary_region": strategy.get("primary_region"),
                        "candidates": [f"{c['province']}={c['score']:.3f}" for c in ranked[:10]],
                        "ranked_candidates": [
                            {"province": c["province"], "city": c.get("city"),
                             "score": c["score"], "signals": c["signals"][:3]}
                            for c in ranked[:10]
                        ],
                        "excluded_count": len(excluded),
                        "suggested_first": strategy.get("suggested_first_action"),
                        "hypotheses": [{"province": h["province"], "score": h["score"]}
                                      for h in hypotheses[:5]],
                        "confidence_notes": strategy.get("confidence_notes", ""),
                    }, elapsed, 55)
    return state


# ============================================================
# JSON Parsing Helpers
# ============================================================

def _tool_call_signature(tool_name: str, tool_input: dict) -> str:
    """Create a stable signature for deduplicating tool calls."""
    # Normalize: sort keys, round floats, remove None values
    normalized = {}
    for k, v in sorted(tool_input.items()):
        if v is None:
            continue
        if isinstance(v, float):
            normalized[k] = round(v, 4)
        elif isinstance(v, str) and len(v) > 200:
            # Truncate long strings (e.g., base64) for comparison
            normalized[k] = v[:80]
        else:
            normalized[k] = v
    return f"{tool_name}:{json.dumps(normalized, sort_keys=True, ensure_ascii=False)}"

def _parse_decision(response: str) -> dict | None:
    import re
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*\}", response, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    return None


# ============================================================
# Cross-Validation Helpers (preserved)
# ============================================================

def _extract_location_from_result(tool_result: dict) -> list[float]:
    data = tool_result.get("data")
    if not data:
        return []
    lats = []
    items = data if isinstance(data, list) else [data]
    for item in items:
        if isinstance(item, dict):
            lat = item.get("lat")
        elif hasattr(item, "lat"):
            lat = item.lat
        else:
            continue
        if lat is not None and isinstance(lat, (int, float)):
            lats.append(float(lat))
    return lats


def _extract_province_from_result(tool_result: dict) -> str:
    data = tool_result.get("data", {}) if isinstance(tool_result.get("data"), dict) else None
    items = tool_result.get("data", []) if isinstance(tool_result.get("data"), list) else []
    raw = ""
    if items and isinstance(items[0], dict):
        raw = items[0].get("province", "") or items[0].get("display_name", "")
    elif data:
        raw = data.get("province", "") or data.get("display_name", "")
    return normalize_province_name(raw)


def _check_climate_mismatch(lats: list[float], clues: dict) -> str | None:
    if not lats or not clues:
        return None
    avg_lat = sum(lats) / len(lats)
    climate = clues.get("climate_zone", "")
    veg = clues.get("vegetation", [])
    hemi = clues.get("hemisphere", "")
    tropical_veg = {"棕榈树", "椰子", "热带雨林", "芭蕉", "香蕉树", "椰子树", "榕树"}
    if climate == "tropical" and abs(avg_lat) > 25:
        return f"搜索结果在纬度{avg_lat:.0f}°（温带），但视觉显示热带气候"
    if any(v in tropical_veg for v in veg) and abs(avg_lat) > 32:
        return f"搜索结果在纬度{avg_lat:.0f}°，但图中可见热带植被，不可能在这个纬度"
    if hemi == "north" and avg_lat < -5:
        return f"搜索结果在南半球（{avg_lat:.0f}°），但视觉提示北半球"
    if hemi == "south" and avg_lat > 5:
        return f"搜索结果在北半球（{avg_lat:.0f}°），但视觉提示南半球"
    veg_keywords_flat = " ".join(veg) if veg else ""
    for rule in get_china_vegetation_lat_ranges():
        if any(kw in veg_keywords_flat for kw in rule["keywords"]):
            if "max_lat" in rule and avg_lat > rule["max_lat"]:
                return rule["warning"].format(lat=avg_lat)
            if "min_lat" in rule and avg_lat < rule["min_lat"]:
                return rule["warning"].format(lat=avg_lat)
    return None


def _check_china_architecture_mismatch(lats: list[float], clues: dict, tool_result: dict) -> str | None:
    architecture = clues.get("architecture", "")
    if not architecture or not lats:
        return None
    result_province = _extract_province_from_result(tool_result)
    for rule in get_china_architecture_region_rules():
        if any(kw in architecture for kw in rule["keywords"]):
            if result_province and not contains_province(result_province, rule["expected_provinces"]):
                return rule["warning"] + f"，但搜索结果显示{result_province}"
    return None


def _check_china_text_region_mismatch(clues: dict, tool_result: dict) -> str | None:
    scripts = clues.get("script", [])
    if not scripts:
        return None
    result_province = _extract_province_from_result(tool_result)
    if not result_province:
        return None
    for rule in get_china_script_region_rules():
        if rule["script"] in scripts:
            if not contains_province(result_province, rule["expected_provinces"]):
                return rule["warning"].format(province=result_province)
    return None


def _check_china_climate_visual_mismatch(clues: dict, tool_result: dict) -> str | None:
    climate = clues.get("climate_zone", "")
    veg = clues.get("vegetation", [])
    veg_str = " ".join(veg) if veg else ""
    summary = clues.get("summary", "")
    combined_text = f"{veg_str} {summary} {climate}"
    result_province = _extract_province_from_result(tool_result)
    if not result_province:
        return None
    snow_kw = ["雪", "积雪", "雪地", "冰雪"]
    desert_kw = ["沙漠", "荒漠", "戈壁", "沙丘"]
    red_soil_kw = ["红土", "红壤", "红色土壤"]
    loess_kw = ["黄土", "黄土高原", "窑洞"]
    rice_kw = ["稻田", "水田", "水稻"]
    has_snow = any(kw in combined_text for kw in snow_kw)
    has_desert = any(kw in combined_text for kw in desert_kw)
    has_red = any(kw in combined_text for kw in red_soil_kw)
    has_loess = any(kw in combined_text for kw in loess_kw)
    has_rice = any(kw in combined_text for kw in rice_kw)
    for rule in get_china_climate_rules():
        cond = rule["condition"]
        if cond == "snow_visible" and has_snow:
            if contains_province(result_province, rule.get("excluded_provinces", [])):
                return rule["warning"].format(south_city=result_province)
        elif cond == "desert_landscape" and has_desert:
            if not contains_province(result_province, rule.get("expected_provinces", [])):
                return rule["warning"].format(province=result_province)
        elif cond == "red_soil" and has_red:
            if not contains_province(result_province, rule.get("expected_provinces", [])):
                return rule["warning"].format(province=result_province)
        elif cond == "loess_landscape" and has_loess:
            if not contains_province(result_province, rule.get("expected_provinces", [])):
                return rule["warning"].format(province=result_province)
        elif cond == "rice_paddy" and has_rice:
            if contains_province(result_province, rule.get("excluded_provinces", [])):
                return rule["warning"]
    if has_snow:
        southern = ["海南", "广东", "广西", "云南", "福建", "台湾"]
        for s in southern:
            if s in result_province:
                return f"图中可见积雪，但搜索结果指向{s}（该地区几乎从不下雪）"
    if has_desert:
        desert_regions = ["新疆", "内蒙古", "甘肃", "青海", "宁夏"]
        if not contains_province(result_province, desert_regions):
            return f"图中为荒漠景观，但搜索结果指向{result_province}（非荒漠地区）"
    return None


async def _cross_validate(state: AgentState, tool_name: str, tool_result: dict,
                          tool_entry: dict) -> dict | None:
    clues = state.get("clues") or {}
    lats = _extract_location_from_result(tool_result)
    if not lats:
        return None

    warning = _check_climate_mismatch(lats, clues)
    arch_warning = _check_china_architecture_mismatch(lats, clues, tool_result)
    text_warning = _check_china_text_region_mismatch(clues, tool_result)
    climate_warning = _check_china_climate_visual_mismatch(clues, tool_result)

    all_warnings = [w for w in [warning, arch_warning, text_warning, climate_warning] if w]
    if not all_warnings:
        return None

    combined_warning = "；".join(all_warnings)
    logger.info("cross_validate_mismatch", tool=tool_name, warnings=all_warnings)

    # Amap geo tools are authoritative for Chinese geography — don't redirect
    # based on visual clues alone, since visual interpretation can be wrong
    if tool_name in ("geocode", "reverse_geocode", "search_nearby"):
        logger.info("cross_validate_amap_skip", tool=tool_name,
                   warnings_count=len(all_warnings))
        return None

    prompt = f"""视觉线索: {json.dumps(clues, ensure_ascii=False)}
工具 "{tool_name}" 的搜索结果: {json.dumps(tool_result.get('data'), ensure_ascii=False, default=str)[:500]}

{chr(10).join(['⚠️ 矛盾检测: ' + w for w in all_warnings])}

请判断：搜索结果是否与视觉线索矛盾？如果矛盾，应该在哪座城市或省份重新搜索？输出JSON:
{{"consistent": true/false, "correct_region": "城市或省份名", "reasoning": "判断理由"}}"""

    try:
        response = await _llm_chat([HumanMessage(content=prompt)], temperature=0, max_tokens=300)
        decision = _parse_decision(response)
        if decision and not decision.get("consistent", True):
            return decision
    except Exception as e:
        logger.warning("cross_validate_error", error=str(e))
    return None


async def _apply_redirect(state: AgentState, original_tool: str,
                          redirect: dict, loop_count: int, max_loops: int):
    region = redirect.get("correct_region", "")
    if not region:
        return
    # Validate region looks like a real place name
    if len(region) > 20 or any(kw in region for kw in ["线索", "证据", "转向", "工具", "查询"]):
        logger.warning("cross_validate_redirect_invalid", region=region[:100])
        return

    logger.info("cross_validate_redirect", region=region, reason=redirect.get("reasoning", ""))

    # Update hypotheses: penalize current top, add corrected
    hypotheses = state.get("hypotheses", [])
    if hypotheses:
        hypotheses[0]["score"] = round(hypotheses[0]["score"] * 0.4, 3)
        hypotheses[0]["contradicting_evidence"].append({
            "clue": f"交叉验证矛盾: {redirect.get('reasoning', '')[:100]}",
            "weight": -0.3,
        })
        hypotheses.append({
            "province": region,
            "city": None,
            "score": 0.55,
            "supporting_evidence": [
                {"clue": f"纠正确认: {redirect.get('reasoning', '')[:100]}", "weight": 0.55}
            ],
            "contradicting_evidence": [],
            "source": "redirect",
            "round_created": loop_count,
        })
        hypotheses.sort(key=lambda h: h["score"], reverse=True)
        state["hypotheses"] = hypotheses

    map_service = create_map_service()
    geo_results = await _execute_tool("geocode", {"address": region}, state)
    geo_lat = geo_lng = None
    if geo_results["status"] == "success" and geo_results.get("data"):
        items = geo_results["data"] if isinstance(geo_results["data"], list) else [geo_results["data"]]
        if items:
            first = items[0]
            geo_lat = first.get("lat") if isinstance(first, dict) else (getattr(first, "lat", None))
            geo_lng = first.get("lng") if isinstance(first, dict) else (getattr(first, "lng", None))

    state["tool_calls"] = state.get("tool_calls", []) + [{
        "tool_name": "redirect", "status": "success",
        "input": {"original_tool": original_tool, "warning": redirect.get("reasoning", "")},
        "output": f"已纠正搜索范围 → {region} ({geo_lat}, {geo_lng})",
        "duration_ms": 0,
    }]
    state["messages"] = state.get("messages", []) + [
        AIMessage(content=f"[交叉验证] 矛盾→重新搜索: {region}。"),
    ]

    if original_tool in ("search_place", "search_nearby", "search_landmark") and geo_lat and geo_lng:
        tool_calls = state.get("tool_calls", [])
        tool_input = {}
        for entry in reversed(tool_calls):
            if entry.get("tool_name") == original_tool and not entry.get("redirected"):
                tool_input = entry.get("input", {})
                break
        keyword = tool_input.get("keyword") or tool_input.get("query") or tool_input.get("description") or region
        await push_step(state, 7, "tool_call", f"纠正搜索: search_nearby({region})",
                        "running", {"lat": geo_lat, "lng": geo_lng, "keyword": keyword}, 0,
                        70 + int(25 * loop_count / max_loops))
        result = await _execute_tool("search_nearby",
                                     {"lat": geo_lat, "lng": geo_lng, "keyword": keyword, "radius": 10000},
                                     state)
        state["tool_calls"] = state.get("tool_calls", []) + [{
            "tool_name": "search_nearby", "status": result["status"],
            "input": {"lat": geo_lat, "lng": geo_lng, "keyword": keyword, "radius": 10000},
            "output": result.get("data"), "error": result.get("error"),
            "duration_ms": 0, "redirected": True,
        }]


# ============================================================
# Node 7: ReAct Loop (方案一+六: 多假设追踪 + 结构化反思)
# ============================================================

async def react_loop_node(state: AgentState) -> AgentState:
    settings = get_settings()
    max_loops = settings.max_react_loops
    loop_count = state.get("loop_count", 0) + 1
    state["loop_count"] = loop_count

    if loop_count > max_loops:
        hypotheses = state.get("hypotheses", [])
        if hypotheses:
            top = hypotheses[0]
            # Extract coordinates from tool call history
            coords = _extract_coords_for_result(
                {"province": top["province"], "city": top.get("city", "")},
                state.get("tool_calls", []),
            )
            lat = coords[0] if coords else None
            lng = coords[1] if coords else None
            state["result"] = {
                "_action": "final_answer", "_reason": "max_loops_reached",
                "address": f"中国·{top['province']}",
                "country": "中国", "province": top["province"],
                "city": top.get("city", ""), "lat": lat, "lng": lng,
                "confidence": top["score"],
                "confidence_kind": top.get("confidence_kind", "ranking_score"),
                "reasoning": f"达到最大轮数，最佳假设: {top['province']}({top['score']:.2f})",
            }
        else:
            state["result"] = {"_action": "final_answer", "_reason": "max_loops_reached"}
        return state

    # ---- Format hypotheses for prompt (方案一) ----
    hypotheses = state.get("hypotheses", [])
    hypotheses_text = json.dumps(hypotheses[:5], ensure_ascii=False, indent=2) if hypotheses else "暂无假设"

    strategy = state.get("search_strategy", {})
    strategy_text = json.dumps({
        "primary_region": strategy.get("primary_region"),
        "candidates": strategy.get("candidate_provinces", [])[:8],
        "excluded": strategy.get("excluded_provinces", [])[:10],
        "suggested_first": strategy.get("suggested_first_action"),
        "notes": strategy.get("confidence_notes", ""),
    }, ensure_ascii=False, indent=2) if strategy else "无"

    # Build GeoCLIP context for the ReAct prompt
    geoclip_result = state.get("geoclip_result")
    geo_clip_context_text = "未使用"
    if geoclip_result and geoclip_result.get("top_provinces"):
        predictions = geoclip_result["top_provinces"][:5]
        lines = [
            f"- {p['province']}{('·'+p['city']) if p.get('city') else ''} "
            f"(prob={p['probability']:.4f}, coords={p['lat']:.4f},{p['lon']:.4f})"
            for p in predictions
        ]
        geo_clip_context_text = "\n".join(lines)

    # Build GeoCLIP anchor pre-search context
    geoclip_anchors = state.get("geoclip_anchors") or []
    if geoclip_anchors:
        anchor_lines = []
        for a in geoclip_anchors:
            loc = f"{a.get('province', '')}{a.get('city', '')}{a.get('district', '')}"
            loc = loc if loc else f"({a['lat']:.4f}, {a['lon']:.4f})"
            pois = a.get("landmarks", [])[:5]
            line = f"锚点{a['rank']}: ({a['lat']:.4f}, {a['lon']:.4f}) -> {loc}"
            if pois:
                line += f"\n  周边POI: {' | '.join(pois)}"
            line += f"\n  来源: {a.get('source', '?')}"
            anchor_lines.append(line)
        geoclip_anchor_context_text = "\n".join(anchor_lines)
    else:
        geoclip_anchor_context_text = "未执行锚点预搜（GeoCLIP 未启用或无有效预测）"

    tool_descriptions = "\n".join([
        "- search_place(query, region?): 搜索地点、店名、路牌文字",
        "- geocode(address): 地址→经纬度",
        "- reverse_geocode(lat, lng): 经纬度→地址",
        "- search_nearby(lat, lng, keyword, radius=5000): 周边POI搜索",
        "- search_landmark(description): 搜索著名地标",
        "- reverse_image_search(image_base64, context?): 多引擎以图搜图（百度+Google Lens+Bing+Yandex），返回图片在互联网上的出现位置，需配合其他工具确认具体地点",
        "- extract_china_clues(image_base64): OCR提取图中车牌、电话区号、公路编号等中国线索",
        "- search_similar_images(image_path): CLIP+FAISS本地图片库检索，返回视觉相似且已知GPS的图片，用于发现视觉相似的已知地点",
    ])

    tool_results_raw = json.dumps(state.get("tool_calls", []), ensure_ascii=False, indent=2)
    # Truncate to prevent prompt exceeding 30720 token limit (Qwen)
    if len(tool_results_raw) > 4000:
        tool_results_text = tool_results_raw[:4000] + f"\n... (截断，共{len(state.get('tool_calls', []))}条工具结果)"
    else:
        tool_results_text = tool_results_raw
    failed_tools_text = ", ".join(state.get("failed_tools", set())) or "无"

    ocr_data = state.get("ocr_data") or {}
    ocr_data_text = json.dumps({
        "geolocation_summary": ocr_data.get("geolocation_summary", "无"),
        "license_plates": ocr_data.get("license_plates", []),
        "phone_area_codes": ocr_data.get("phone_area_codes", []),
        "highways": ocr_data.get("highways", []),
    }, ensure_ascii=False, indent=2)

    messages = [
        SystemMessage(content=REACT_SYSTEM_PROMPT.format(
            tool_descriptions=tool_descriptions,
            clues=json.dumps(state.get("clues", {}), ensure_ascii=False, indent=2),
            ocr_data=ocr_data_text,
            exif_data=json.dumps(state.get("exif_data", {}), ensure_ascii=False),
            tool_results=tool_results_text,
            failed_tools=failed_tools_text,
            hypotheses=hypotheses_text,
            search_strategy=strategy_text,
            geo_clip_context=geo_clip_context_text,
            geo_clip_anchor_context=geoclip_anchor_context_text,
            loop_count=loop_count,
            max_loops=max_loops,
        )),
        HumanMessage(content="请根据当前线索和假设列表决定下一步动作。"),
    ]

    t0 = time.time()
    try:
        # v2.1: Stream ReAct reasoning tokens to frontend
        cb = state.get("stream_callback")
        response = await _llm_chat(messages, temperature=0.1, max_tokens=1500,
                                   stream_callback=cb)
        _record_usage(state, response)
    except Exception as e:
        logger.error("react_llm_failed", error=str(e), loop=loop_count)
        state["error"] = f"LLM调用失败(第{loop_count}轮): {str(e)[:200]}"
        return state
    elapsed = int((time.time() - t0) * 1000)

    decision = _parse_decision(response)
    if decision is None:
        logger.warning("react_parse_error", response=response[:200])
        state["messages"] = state.get("messages", []) + [
            AIMessage(content=response),
            HumanMessage(content="你的上一轮输出无法解析为JSON。请严格按照JSON格式输出你的决策。"),
        ]
        return state

    # ---- 方案六: Handle redirect ----
    if decision.get("action") == "redirect":
        new_target = decision.get("new_target")
        reason = decision.get("reason", "")
        logger.info("react_redirect", target=new_target, reason=reason)

        # Validate: reject None or empty targets
        if not new_target or not isinstance(new_target, str):
            logger.warning("react_redirect_invalid", target=str(new_target)[:100])
            state["messages"] = state.get("messages", []) + [
                AIMessage(content=json.dumps({"action": "redirect_rejected",
                    "reason": f"new_target 无效: {new_target}"})),
            ]
            return state

        # Validate: if new_target looks like reasoning text, ignore redirect
        if len(new_target) > 20 or any(kw in new_target for kw in ["线索", "证据", "转向", "工具", "查询"]):
            logger.warning("react_redirect_invalid", target=new_target[:100])
            state["messages"] = state.get("messages", []) + [
                AIMessage(content=json.dumps({"action": "redirect_rejected",
                    "reason": "new_target 不是有效的地名"})),
            ]
            return state

        state["hypotheses"] = [{
            "province": new_target, "city": None, "score": 0.50,
            "supporting_evidence": [{"clue": f"方向修正: {reason[:150]}", "weight": 0.50}],
            "contradicting_evidence": [],
            "source": "redirect", "round_created": loop_count,
        }]
        state["last_redirect_at"] = loop_count
        state["messages"] = state.get("messages", []) + [
            AIMessage(content=f"[方向修正] {reason} → {new_target}"),
        ]
        await push_step(state, 7, "reasoning", f"推理第{loop_count}轮（方向修正）",
                        "done", {"thought": reason, "action": "redirect", "new_target": new_target},
                        elapsed, 70 + int(25 * loop_count / max_loops))
        return state

    # ---- Handle final answer ----
    if "address" in decision or decision.get("action") == "final_answer":
        state["result"] = decision
        final_province = decision.get("province", "")
        if final_province and state.get("hypotheses"):
            for h in state["hypotheses"]:
                if h["province"] == final_province:
                    h["score"] = max(h["score"], float(decision.get("confidence", 0.8) or 0.8))
                    break
        await push_step(state, 7, "reasoning", f"推理第{loop_count}轮",
                        "done", {"thought": decision.get("reasoning", ""), "action": "final_answer"},
                        elapsed, 70 + int(25 * loop_count / max_loops))
        return state

    # ---- v2.1: Extract actions (single or parallel) ----
    raw_actions = decision.get("actions")
    if raw_actions and isinstance(raw_actions, list) and len(raw_actions) > 0:
        actions = raw_actions[:3]  # Max 3 parallel calls
    else:
        actions = [decision]  # Single action (backward compatible)

    # ---- Validate all actions before execution ----
    valid_actions: list[dict] = []
    skipped_msgs: list[str] = []
    for act in actions:
        act_name = (act.get("action") or act.get("tool_name")
                    or act.get("tool") or act.get("tool_call") or "")
        act_input = (act.get("action_input") or act.get("tool_input")
                     or act.get("query") or {})
        if isinstance(act_input, str):
            act_input = {"query": act_input}

        if not act_name or act_name in ("final_answer", "redirect"):
            continue

        if act_name in state.get("failed_tools", set()):
            skipped_msgs.append(f"⚠️ {act_name} 之前已失败，已跳过")
            state["tool_calls"] = state.get("tool_calls", []) + [{
                "tool_name": act_name, "status": "skipped", "reason": "previously_failed",
            }]
            continue

        act_sig = _tool_call_signature(act_name, act_input)
        is_dup = False
        for prior in state.get("tool_calls", []):
            if prior.get("_sig") == act_sig:
                skipped_msgs.append(f"⚠️ {act_name}({json.dumps(act_input, ensure_ascii=False)[:80]}) 重复调用，已跳过")
                state["tool_calls"] = state.get("tool_calls", []) + [{
                    "tool_name": act_name, "status": "skipped",
                    "reason": "duplicate — identical call already made",
                }]
                is_dup = True
                break
        if is_dup:
            continue

        valid_actions.append({
            "name": act_name,
            "input": act_input,
            "sig": act_sig,
            "target": act.get("target_hypothesis", ""),
        })

    if not valid_actions:
        # All actions were invalid — tell LLM to try something different
        state["messages"] = state.get("messages", []) + [
            HumanMessage(content="\n".join(skipped_msgs)
                         + "\n所有工具调用都被跳过。请选择一个不同的工具或不同的参数。"),
        ]
        await push_step(state, 7, "reasoning", f"推理第{loop_count}轮（工具跳过）",
                        "done", {"thought": "所有工具调用无效", "skipped": skipped_msgs},
                        elapsed, 70 + int(25 * loop_count / max_loops))
        return state

    # Notify about skipped tools alongside valid ones
    if skipped_msgs:
        state["messages"] = state.get("messages", []) + [
            AIMessage(content="\n".join(skipped_msgs)),
        ]

    # ---- Execute valid actions in parallel ----
    parallel_label = ", ".join(a["name"] for a in valid_actions)
    await push_step(state, 7, "tool_call",
                    f"并行调用({len(valid_actions)}): {parallel_label}",
                    "running", {"actions": [{"tool": a["name"], "input": a["input"]}
                                           for a in valid_actions]},
                    0, 70 + int(25 * loop_count / max_loops))

    async def _exec_one(act: dict) -> dict:
        t0 = time.time()
        result = await _execute_tool(act["name"], act["input"], state)
        elapsed = int((time.time() - t0) * 1000)
        return {
            "tool_name": act["name"],
            "status": result["status"],
            "input": act["input"],
            "output": result.get("data"),
            "error": result.get("error"),
            "duration_ms": elapsed,
            "target_hypothesis": act["target"],
            "_sig": act["sig"],
        }

    tool_entries = await asyncio.gather(*[_exec_one(a) for a in valid_actions])

    # ---- Process results for each tool ----
    for entry in tool_entries:
        tool_name2 = entry["tool_name"]
        tool_result2 = {"status": entry["status"], "data": entry.get("output"),
                       "error": entry.get("error")}
        target = entry.get("target_hypothesis", "")

        state["tool_calls"] = state.get("tool_calls", []) + [entry]

        # Update hypotheses
        if target and hypotheses:
            result_province = _extract_province_from_result(tool_result2)
            is_supporting = (
                entry["status"] == "success"
                and (not result_province or target in result_province
                     or result_province in target)
            )
            for i, h in enumerate(hypotheses):
                if h["province"] == target or (
                    result_province and h["province"] in result_province
                ):
                    hypotheses[i] = update_hypothesis_score(
                        h, tool_result2, is_supporting, loop_count, tool_name2,
                    )
            hypotheses.sort(key=lambda h: h["score"], reverse=True)
            state["hypotheses"] = hypotheses

        # City fingerprint boost
        if entry["status"] == "success" and hypotheses:
            result_province = _extract_province_from_result(tool_result2)
            if result_province and tool_name2 in ("geocode", "reverse_geocode",
                                                   "search_nearby", "search_place"):
                from app.tools.china_knowledge import match_city_features, get_city_fingerprint
                vision_raw = state.get("vision_raw", "")
                clues = state.get("clues", {}) or {}
                combined = vision_raw + " " + json.dumps(clues, ensure_ascii=False)
                for h in hypotheses[:3]:
                    city = h.get("city", "")
                    if not city:
                        cf = get_city_fingerprint(h.get("province", ""))
                        city = cf["city"] if cf else None
                    if city:
                        matches, matched_kw = match_city_features(combined, city)
                        if matches >= 2:
                            bonus = 0.03 * min(matches, 5)
                            h["score"] = round(min(h["score"] + bonus, 0.92), 3)
                            h.setdefault("supporting_evidence", []).append({
                                "clue": f"城市指纹匹配({city}): {', '.join(matched_kw[:3])}",
                                "weight": round(bonus, 3),
                            })

        # Track failed tools
        if entry["status"] in ("unavailable", "timeout", "upstream_error", "failed"):
            state["failed_tools"] = state.get("failed_tools", set()) | {tool_name2}
            cb = state.get("stream_callback")
            if cb:
                error_code = entry.get("error", "")
                if error_code == "tool_budget_exceeded":
                    warning_message = "已达到本次分析的工具调用上限，后续工具验证已停止"
                else:
                    warning_message = f"工具 {tool_name2} 暂不可用：{error_code or entry['status']}"
                await cb({
                    "event": "tool_warning",
                    "data": {
                        "tool": tool_name2, "reason": entry["status"],
                        "message": warning_message,
                    }
                })

        # Cross-validate
        if entry["status"] == "success":
            redirect = await _cross_validate(state, tool_name2, tool_result2, entry)
            if redirect:
                await _apply_redirect(state, tool_name2, redirect, loop_count, max_loops)

        await push_step(state, 7, "tool_call", f"工具结果: {tool_name2}",
                        "done", entry, entry["duration_ms"],
                        70 + int(25 * loop_count / max_loops))

    # ---- 方案六: Structured Reflection (every 3 rounds) ----
    if loop_count % 3 == 0:
        state["last_redirect_at"] = loop_count
        hyp_text = json.dumps(hypotheses[:3], ensure_ascii=False, indent=2) if hypotheses else "无"
        reflection_prompt = f"""你是推理方向审阅专家。结构化评估当前推理状态。

当前假设: {hyp_text}
线索: {json.dumps(state.get('clues', {}), ensure_ascii=False)}
最近工具: {json.dumps(state.get('tool_calls', [])[-3:], ensure_ascii=False)}

请用JSON格式回答（只输出JSON）：
{{"on_track": true/false, "top_hypothesis_plausible": true/false,
 "should_consider_alternative": true/false,
 "alternative_province": "省份名 | null",
 "should_change_region": true/false,
 "suggested_new_direction": "具体建议 | null",
 "reasoning": "评估理由"}}"""

        try:
            reflection_raw = await _llm_chat(
                [HumanMessage(content=reflection_prompt)],
                temperature=0.1, max_tokens=400,
            )
            reflection = _parse_decision(reflection_raw)
            if reflection and not reflection.get("on_track") and reflection.get("should_change_region"):
                new_dir = _canonical_province_name(reflection.get("alternative_province"))
                if not new_dir:
                    new_dir = _canonical_province_name(reflection.get("suggested_new_direction"))
                if new_dir:
                    alt_found = False
                    for h in hypotheses:
                        if h["province"] == new_dir:
                            h["score"] = min(h["score"] + 0.20, 0.85)
                            alt_found = True
                            break
                    if not alt_found:
                        hypotheses.append({
                            "province": new_dir, "city": None, "score": 0.50,
                            "supporting_evidence": [
                                {"clue": f"反思推荐: {reflection.get('reasoning', '')[:150]}", "weight": 0.50}
                            ],
                            "contradicting_evidence": [],
                            "source": "reflection", "round_created": loop_count,
                        })
                    hypotheses.sort(key=lambda h: h["score"], reverse=True)
                    state["hypotheses"] = hypotheses
            state["messages"] = state.get("messages", []) + [
                AIMessage(content=json.dumps({
                    "type": "structured_reflection", "round": loop_count,
                    "on_track": reflection.get("on_track") if reflection else "unknown",
                    "reasoning": reflection.get("reasoning", "") if reflection else "",
                }, ensure_ascii=False))
            ]
        except Exception as e:
            logger.warning("reflection_error", error=str(e))

    # ---- Auto-finalize on strong consensus (方案一) ----
    # Requires: (a) top score > 0.80, (b) gap > 0.20, AND
    #            (c) evidence from >= 2 independent signal sources
    # (prevents OCR-alone dominance from auto-converging on wrong answer)
    if hypotheses and len(hypotheses) >= 2:
        top = hypotheses[0]
        second = hypotheses[1]
        # Count distinct evidence sources
        sources = set()
        for e in top.get("supporting_evidence", []):
            explicit_source = str(e.get("source", ""))
            if explicit_source:
                sources.add(explicit_source)
                continue
            clue = e.get("clue", e.get("summary", ""))
            # Check geoclip before clip — "geoclip" contains "clip" as substring
            if "OCR" in clue or "ocr" in clue:
                sources.add("ocr")
            elif "GeoCLIP" in clue or "geoclip" in clue:
                sources.add("geoclip")
            elif "CLIP" in clue or "clip" in clue or "相似图" in clue:
                sources.add("clip")
            elif "视觉" in clue or "vision" in clue or "架构" in clue or "植被" in clue:
                sources.add("vision")
            elif "城市指纹" in clue or "city" in clue.lower():
                sources.add("city_fingerprint")
            elif "工具" in clue or "tool" in clue.lower() or "验证" in clue:
                sources.add("tool_result")
            elif "纠正" in clue or "redirect" in clue.lower():
                sources.add("redirect")
        multi_source = len(sources) >= 2

        if top["score"] > 0.80 and (top["score"] - second["score"]) > 0.20 and multi_source:
            logger.info("hypothesis_consensus", top=top["province"], score=top["score"])

            # Extract coordinates from tool call history
            coords = _extract_coords_for_result(
                {"province": top["province"], "city": top.get("city", "")},
                state.get("tool_calls", []),
            )
            lat = coords[0] if coords else None
            lng = coords[1] if coords else None

            state["result"] = {
                "_action": "final_answer", "_reason": "hypothesis_consensus",
                "address": f"中国·{top['province']}" + (f"·{top['city']}" if top.get("city") else ""),
                "country": "中国", "province": top["province"],
                "city": top.get("city", ""), "lat": lat, "lng": lng,
                "confidence": top["score"],
                "confidence_kind": top.get("confidence_kind", "ranking_score"),
                "reasoning": f"多假设收敛: {top['province']}({top['score']:.2f}) vs {second['province']}({second['score']:.2f})",
            }

    return state


# ============================================================
# Tool Execution
# ============================================================

async def _execute_tool(tool_name: str, tool_input: dict, state: AgentState) -> dict:
    settings = get_settings()
    timeout = min(TOOL_TIMEOUTS.get(tool_name, settings.tool_timeout_seconds), settings.max_tool_elapsed_seconds)
    budget = state.get("tool_budget")
    if budget is None:
        budget = ToolBudget(settings.max_total_tool_calls, settings.max_tool_elapsed_seconds)
        state["tool_budget"] = budget
    if not budget.consume():
        return {"status": "unavailable", "error": "tool_budget_exceeded"}
    # Copy to avoid mutating the caller's dict (which gets stored in tool_calls)
    tool_input = dict(tool_input)

    try:
        if tool_name == "reverse_image_search":
            if get_settings().reverse_image_service.lower() in {"none", "disabled", "off"}:
                return {"status": "skipped", "data": [{
                    "title": "以图搜图已禁用",
                    "url": "",
                    "snippet": "REVERSE_IMAGE_SERVICE=none",
                    "source": "disabled",
                }]}
            tool_input["image_base64"] = state["image_base64"]
            # Use async version directly to avoid nested ThreadPoolExecutor
            result = await asyncio.wait_for(
                reverse_image_search_async(
                    tool_input.get("image_base64", ""),
                    tool_input.get("context", ""),
                ),
                timeout=timeout,
            )
            return {"status": "empty_result" if not result else "success", "data": result}
        elif tool_name == "extract_china_clues":
            tool_input["image_base64"] = encode_image_for_ocr(state["image_path"])
        elif tool_name == "search_similar_images":
            tool_input["image_path"] = state["image_path"]

        if tool_name in TOOL_MAP:
            result = await asyncio.wait_for(
                asyncio.to_thread(TOOL_MAP[tool_name].invoke, tool_input),
                timeout=timeout,
            )
            return {"status": "empty_result" if not result else "success", "data": result}

        elif tool_name in MAP_TOOLS:
            map_service = create_map_service()
            if tool_name == "geocode":
                address = str(tool_input.get("address", "")).strip()
                if not address:
                    return {"status": "invalid_input", "error": "address is required"}
                result = await asyncio.wait_for(
                    map_service.geocode(address),
                    timeout=timeout,
                )
                data = [vars(r) for r in result]
                return {"status": "empty_result" if not data else "success", "data": data}
            elif tool_name == "reverse_geocode":
                lat = tool_input.get("lat")
                lng = tool_input.get("lng")
                if not validate_coordinate(lat, lng):
                    return {"status": "invalid_input", "error": "lat and lng are required"}
                result = await asyncio.wait_for(
                    map_service.reverse_geocode(lat, lng), timeout=timeout,
                )
                data = [vars(result)] if result else []
                return {"status": "empty_result" if not data else "success", "data": data}
            elif tool_name == "search_nearby":
                lat = tool_input.get("lat")
                lng = tool_input.get("lng")
                keyword = str(tool_input.get("keyword", "")).strip()
                if not validate_coordinate(lat, lng) or not keyword:
                    return {"status": "invalid_input", "error": "lat, lng and keyword are required"}
                result = await asyncio.wait_for(
                    map_service.search_nearby(lat, lng, keyword, tool_input.get("radius", 5000)),
                    timeout=timeout,
                )
                data = [vars(r) for r in result]
                return {"status": "empty_result" if not data else "success", "data": data}

        return {"status": "unavailable", "error": f"unknown_tool:{tool_name}"}

    except asyncio.TimeoutError:
        return {"status": "timeout", "error": f"超时 ({timeout}s)"}
    except Exception as e:
        status_code = getattr(e, "status_code", None)
        status = "upstream_error" if status_code or isinstance(e, (ConnectionError, OSError)) else "failed"
        return {"status": status, "error": f"{type(e).__name__}: {str(e)[:200]}"}


def _result_looks_valid(result: dict) -> bool:
    """Check if result fields look like actual place names, not LLM reasoning text."""
    if not result or not isinstance(result, dict):
        return False
    province = result.get("province", "") or ""
    address = result.get("address", "") or ""
    # Province name should be short (real province names are ≤10 chars in Chinese)
    if len(province) > 20:
        return False
    # Address should not contain reasoning keywords
    reasoning_kw = ["线索", "工具查询", "证据", "转向", "方向修正", "应转向", "强烈支持"]
    for kw in reasoning_kw:
        if kw in province or kw in address:
            return False
    # Province should look like a Chinese place name
    if province and not any(
        province.endswith(suffix) for suffix in ["省", "市", "区", "县", "自治区", "自治州"]
    ):
        # Might still be valid (like "海南"), but if it's very long, reject
        if len(province) > 10:
            return False
    return True


def _extract_coords_for_result(result: dict, tool_calls: list) -> tuple | None:
    """Extract lat/lng from tool calls that match the result's province/city."""
    if not tool_calls:
        return None
    province = (result.get("province", "") or "").replace("省", "").replace("市", "")
    city = (result.get("city", "") or "").replace("市", "")
    # Search tool calls in reverse order (most recent first) for geo results
    for entry in reversed(tool_calls):
        if entry.get("status") != "success":
            continue
        tool = entry.get("tool_name", "")
        if tool not in ("geocode", "reverse_geocode", "search_nearby"):
            continue
        data = entry.get("output")
        if not data:
            continue
        items = data if isinstance(data, list) else [data]
        for item in items:
            if not isinstance(item, dict):
                continue
            lat = item.get("lat")
            lng = item.get("lng") or item.get("lon")
            if lat is None or lng is None:
                continue
            # Check if the geo result matches our target province/city
            item_province = item.get("province", "") or ""
            item_city = item.get("city", "") or ""
            item_address = item.get("address", "") or item.get("display_name", "") or ""
            if province and (province in item_province or province in item_city
                           or province in item_address):
                return (float(lat), float(lng))
            if city and (city in item_city or city in item_address):
                return (float(lat), float(lng))
    # No match found — return last successful geo result's coords as fallback
    for entry in reversed(tool_calls):
        if entry.get("status") != "success":
            continue
        if entry.get("tool_name") not in ("geocode", "reverse_geocode", "search_nearby"):
            continue
        data = entry.get("output")
        if not data:
            continue
        items = data if isinstance(data, list) else [data]
        for item in items:
            if isinstance(item, dict) and item.get("lat") and (item.get("lng") or item.get("lon")):
                return (float(item["lat"]), float(item.get("lng") or item.get("lon")))
    return None


def _apply_near_duplicate_match(result: dict, state: AgentState) -> None:
    """Use an almost exact indexed image match as deterministic GPS evidence."""
    clip_result = state.get("clip_result") or {}
    matches = clip_result.get("matches") or []
    if not matches:
        return

    match = matches[0]
    similarity = float(match.get("similarity") or 0)
    lat, lng = match.get("lat"), match.get("lon", match.get("lng"))
    if similarity < 0.995 or not validate_coordinate(lat, lng):
        return

    result["lat"] = float(lat)
    result["lng"] = float(lng)
    result["coord_system"] = "WGS84"

    geo_match = next((item for item in clip_result.get("top_provinces") or []
                      if abs(float(item.get("lat", 999)) - float(lat)) < 1e-6
                      and abs(float(item.get("lon", 999)) - float(lng)) < 1e-6), None)
    if geo_match:
        province = geo_match.get("province") or result.get("province") or ""
        city = geo_match.get("city") or result.get("city") or ""
        result["country"] = result.get("country") or "中国"
        result["province"] = province
        result["city"] = city
        result["address"] = "·".join(part for part in (result["country"], province, city) if part)
        result["precision_level"] = "city" if city else "province"

    evidence = result.setdefault("evidence", [])
    if not any(item.get("source") == "near_duplicate" for item in evidence if isinstance(item, dict)):
        evidence.append({
            "source": "near_duplicate",
            "direction": "support",
            "reliability": 1.0,
            "summary": f"本地图像库近重复匹配（相似度 {similarity:.3f}）",
            "unique": True,
        })


# ============================================================
# Node 8: Result Synthesis
# ============================================================

async def result_synthesize_node(state: AgentState) -> AgentState:
    t0 = time.time()

    raw_result = state.get("result") or {}
    hypotheses = state.get("hypotheses", [])

    # Build fallback from top hypothesis (used when LLM fails or no address)
    top_hypothesis_result = None
    if hypotheses:
        top = hypotheses[0]
        top_hypothesis_result = {
            "address": f"中国·{top['province']}" + (f"·{top['city']}" if top.get("city") else ""),
            "country": "中国", "province": top["province"],
            "city": top.get("city", ""), "lat": None, "lng": None,
            "confidence": top.get("score", 0.1),
            "confidence_kind": top.get("confidence_kind", "ranking_score"),
            "evidence": top.get("supporting_evidence", []) + top.get("contradicting_evidence", []),
            "reasoning": f"最佳假设: {top['province']}({top.get('score', 0):.2f})",
        }

    # If max_loops and no address, use top hypothesis
    if raw_result.get("_reason") == "max_loops_reached" and "address" not in raw_result:
        if top_hypothesis_result:
            raw_result = top_hypothesis_result

    # If raw_result has no address at all, use hypothesis fallback
    if not raw_result.get("address") and top_hypothesis_result:
        raw_result = top_hypothesis_result
        raw_result["_reason"] = "llm_fallback"

    # If STILL no address — try extracting from tool_call history first,
    # then fall back to geoclip coordinates.
    if not raw_result.get("address"):
        # Try extracting location from recent tool outputs (geocode/reverse_geocode)
        tool_calls = state.get("tool_calls", [])
        extracted = None
        for tc in reversed(tool_calls):
            if tc.get("status") != "success":
                continue
            data = tc.get("output")
            if not data:
                continue
            items = data if isinstance(data, list) else [data]
            for item in items:
                if not isinstance(item, dict):
                    continue
                # Prefer results with province/city/display_name
                display = item.get("display_name", "") or item.get("formatted_address", "")
                province = item.get("province", "") or item.get("state", "")
                city = item.get("city", "") or item.get("town", "")
                country = item.get("country", "")
                if display and (province or city or country):
                    # Found a usable location
                    address = "·".join(p for p in [country, province, city] if p) or display
                    extracted = {
                        "address": address,
                        "country": country, "province": province, "city": city,
                        "lat": item.get("lat"),
                        "lng": item.get("lng") if item.get("lng") is not None else item.get("lon"),
                        "confidence": 0.3,
                        "reasoning": f"从工具调用历史提取: {address}",
                        "_reason": "tool_history_extract",
                    }
                    break
            if extracted:
                break

        if extracted:
            raw_result = extracted
        else:
            # Geoclip fallback
            geoclip = state.get("geoclip_result") or {}
            top_coords = geoclip.get("top_coords", [])
            if top_coords:
                c = top_coords[0]
                raw_result = {
                    "address": f"推测坐标 ({c['lat']:.4f}, {c['lon']:.4f})",
                    "country": "", "province": "", "city": "",
                    "lat": c["lat"], "lng": c["lon"],
                    "confidence": round(c.get("probability", 0.1), 2),
                    "reasoning": f"GeoCLIP推测: ({c['lat']:.4f}, {c['lon']:.4f})",
                    "_reason": "geoclip_fallback",
                }
            else:
                vision = state.get("vision_raw", "")
                raw_result = {
                    "address": "无法确定",
                    "country": "", "province": "", "city": "",
                    "lat": None, "lng": None, "confidence": 0.05,
                    "reasoning": f"无法确定位置。{vision[:200] if vision else ''}",
                    "_reason": "no_signal_fallback",
                }

    verification_feedback = state.get("verification_feedback", "未执行对抗验证")

    fmt = {
        "clues": json.dumps(state.get("clues", {}), ensure_ascii=False),
        "exif_data": json.dumps(state.get("exif_data", {}), ensure_ascii=False),
        "reasoning_history": json.dumps(state.get("tool_calls", []), ensure_ascii=False),
    }

    if raw_result.get("_action") == "final_answer" and raw_result.get("_reason") == "max_loops_reached":
        prompt_text = RESULT_SYNTHESIS_PROMPT.format(
            **fmt,
            final_output="已达到最大推理轮数，基于已有线索给出最佳推测。",
            verification_feedback=verification_feedback,
        )
        try:
            response = await _llm_chat([HumanMessage(content=prompt_text)], temperature=0.1, max_tokens=1000)
            _record_usage(state, response)
            result = _parse_decision(response) or raw_result
        except Exception as e:
            logger.error("synthesize_llm_failed", error=str(e))
            result = raw_result
    elif raw_result.get("address"):
        # Already have a usable result (from hypothesis or ReAct), skip LLM call
        result = raw_result
    else:
        prompt_text = RESULT_SYNTHESIS_PROMPT.format(
            **fmt,
            final_output=json.dumps(raw_result, ensure_ascii=False),
            verification_feedback=verification_feedback,
        )
        try:
            response = await _llm_chat([HumanMessage(content=prompt_text)], temperature=0.1, max_tokens=1000)
            _record_usage(state, response)
            result = _parse_decision(response) or raw_result
        except Exception as e:
            logger.error("synthesize_llm_failed", error=str(e))
            result = raw_result if raw_result.get("address") else (top_hypothesis_result or raw_result)

    # ---- Validate result: if province/address looks like reasoning text, fall back ----
    if not _result_looks_valid(result) and top_hypothesis_result:
        logger.warning("result_validation_failed",
                       address=str(result.get("address", ""))[:100],
                       province=str(result.get("province", "")))
        result = top_hypothesis_result

    # A near-duplicate indexed image is stronger than inferred coordinates.
    _apply_near_duplicate_match(result, state)

    # ---- Fill missing coordinates from tool call history ----
    if result.get("lat") is None or result.get("lng") is None:
        tool_calls_all = state.get("tool_calls", [])
        coords = _extract_coords_for_result(result, tool_calls_all)
        if coords:
            result["lat"] = coords[0]
            result["lng"] = coords[1]

    tool_calls = state.get("tool_calls", [])
    total = len(tool_calls)
    success = sum(1 for t in tool_calls if t["status"] == "success")
    timeout = sum(1 for t in tool_calls if t["status"] == "timeout")
    budget_skipped = sum(
        1 for t in tool_calls
        if t["status"] == "unavailable" and t.get("error") == "tool_budget_exceeded"
    )
    unavailable = sum(
        1 for t in tool_calls
        if t["status"] == "unavailable" and t.get("error") != "tool_budget_exceeded"
    )
    invalid_input = sum(1 for t in tool_calls if t["status"] == "invalid_input")
    upstream_error = sum(1 for t in tool_calls if t["status"] == "upstream_error")
    empty_result = sum(1 for t in tool_calls if t["status"] == "empty_result")
    failed = sum(1 for t in tool_calls if t["status"] in {"failed", "upstream_error"})

    result["tool_stats"] = {
        "total_calls": total, "success": success,
        "timeout": timeout, "failed": failed,
        "unavailable": unavailable, "budget_skipped": budget_skipped,
        "invalid_input": invalid_input,
        "upstream_error": upstream_error, "empty_result": empty_result,
    }
    result["tokens_used"] = int(state.get("tokens_used", 0))
    result["model_calls"] = int(state.get("model_calls", 0))
    result["model_usage"] = state.get("model_usage", {})
    result["estimated_cost"] = round(
        result["tokens_used"] / 1000 * get_settings().model_cost_per_1k_tokens, 6
    )
    result["total_elapsed_ms"] = 0
    selected_province = _canonical_province_name(result.get("province"))
    ordered_hypotheses = _sanitize_hypotheses(
        hypotheses or [], selected_province, result.get("confidence"),
    )
    result["top_hypotheses"] = [
        {"province": h["province"], "score": h["score"],
         "evidence_count": len(h.get("supporting_evidence", [])),
         "selected": h.get("province") == selected_province}
        for h in ordered_hypotheses[:3]
    ]
    if not result.get("evidence") and hypotheses:
        result["evidence"] = (
            hypotheses[0].get("supporting_evidence", [])
            + hypotheses[0].get("contradicting_evidence", [])
        )
    if result.get("confidence_kind") is None and hypotheses:
        result["confidence_kind"] = hypotheses[0].get("confidence_kind", "ranking_score")
    result.setdefault("confidence_kind", "ranking_score")

    state["result"] = result

    elapsed = int((time.time() - t0) * 1000)
    await push_step(state, 9, "final", "结果整合", "done", {
        "address": result.get("address", ""),
        "confidence": result.get("confidence", 0),
    }, elapsed, 90)

    return state


# ============================================================
# Node 9: Adversarial Verification (方案二)
# ============================================================

async def adversarial_verify_node(state: AgentState) -> AgentState:
    """Verify the final answer by actively searching for contradictions."""
    result = state.get("result") or {}
    if not result.get("address"):
        state["verification_passed"] = True
        state["verification_feedback"] = "无答案需要验证"
        return state

    t0 = time.time()

    prompt = ADVERSARIAL_VERIFY_PROMPT.format(
        address=result.get("address", ""),
        province=result.get("province", ""),
        city=result.get("city", ""),
        lat=result.get("lat"),
        lng=result.get("lng"),
        confidence=result.get("confidence", 0),
        reasoning=result.get("reasoning", ""),
        clues=json.dumps(state.get("clues", {}), ensure_ascii=False, indent=2),
        ocr_data=json.dumps(state.get("ocr_data", {}), ensure_ascii=False, indent=2),
        tool_calls=json.dumps(state.get("tool_calls", [])[-5:], ensure_ascii=False, indent=2),
    )

    try:
        response = await _llm_chat([HumanMessage(content=prompt)], temperature=0, max_tokens=500)
        _record_usage(state, response)
        verification = _parse_decision(response)
    except Exception as e:
        logger.warning("verify_error", error=str(e))
        verification = None

    if verification is None:
        verification = {"valid": True, "contradictions": [], "confidence_adjustment": 0,
                       "suggested_correction": None}

    elapsed = int((time.time() - t0) * 1000)

    is_valid = verification.get("valid", True)
    contradictions = verification.get("contradictions", [])
    adjustment = float(verification.get("confidence_adjustment", 0) or 0)
    suggestion = verification.get("suggested_correction")

    state["verification_passed"] = is_valid
    state["verification_feedback"] = json.dumps(verification, ensure_ascii=False)

    # Adjust confidence
    if adjustment != 0:
        old_conf = result.get("confidence", 0)
        new_conf = max(0.05, min(0.98, old_conf + adjustment))
        result["confidence"] = round(new_conf, 3)
        if adjustment < 0:
            result["verification_notes"] = (
                f"验证发现矛盾: {'; '.join(contradictions[:3])}。"
                f"置信度从 {old_conf} 调整为 {new_conf}"
            )
        state["result"] = result

    await push_step(state, 10, "verification", "对抗验证",
                    "done", {
                        "valid": is_valid,
                        "contradictions": contradictions,
                        "adjustment": adjustment,
                        "suggested_correction": suggestion,
                    }, elapsed, 95)

    return state


# ============================================================
# Node 10: Result enrichment (never promotes precision without unique evidence)
# ============================================================

async def result_enrichment_node(state: AgentState) -> AgentState:
    """Add city context without claiming road/POI precision."""
    result = state.get("result") or {}
    if not result.get("address"):
        return state

    confidence = float(result.get("confidence") or 0)
    if confidence < 0.5:
        return state

    province = result.get("province", "") or ""
    city = result.get("city", "") or ""
    address = result.get("address", "") or ""

    # Need at least city-level granularity to narrow
    if not city and not address:
        return state

    t0 = time.time()

    try:
        map_service = create_map_service()

        lat, lng = result.get("lat"), result.get("lng")
        coord_system = result.get("coord_system") or "WGS84"
        district = result.get("district", "") or ""

        if lat is None or lng is None:
            # Prefer the most specific known locality. Province-only geocoding
            # returns an administrative center that can contradict the city.
            search_query = f"{province}{city}" if city else address
            geo_results = await map_service.geocode(search_query)
            if not geo_results and search_query != address:
                geo_results = await map_service.geocode(address)
            if not geo_results:
                return state

            geo_info = vars(geo_results[0])
            lat = geo_info.get("lat")
            lng = geo_info.get("lng")
            coord_system = geo_info.get("coord_system") or "WGS84"
            district = geo_info.get("district", "") or district

        # Step 2: Search for nearby landmarks to corroborate
        landmarks = await map_service.search_nearby(lat, lng, keyword="景点", radius=3000)
        landmark_names = [p.name for p in landmarks[:5]] if landmarks else []

        # Also search for urban features (squares, markets, stations)
        urban = await map_service.search_nearby(lat, lng, keyword="广场", radius=2000)
        urban_names = [p.name for p in urban[:3]] if urban else []

        transit = await map_service.search_nearby(lat, lng, keyword="地铁站", radius=2000)
        transit_names = [p.name for p in transit[:3]] if transit else []

        # Reverse geocoding describes the city representative point. It is
        # contextual enrichment, not proof that the photo was taken there.
        geo_detail = await map_service.reverse_geocode(lat, lng)
        if geo_detail:
            detail_info = vars(geo_detail)
            district = detail_info.get("district", "") or district

        # Step 4: Extract city fingerprint for additional cross-ref
        from app.tools.china_knowledge import get_city_fingerprint
        city_clean = city.rstrip("市") if city else ""
        fingerprint = get_city_fingerprint(city_clean) if city_clean else None
        fingerprint_features = fingerprint.get("features", [])[:3] if fingerprint else []

        # Step 5: Enrich result
        if district and not result.get("district"):
            result["district"] = district
        if lat is not None and lng is not None and (result.get("lat") is None or result.get("lng") is None):
            result["lat"] = lat
            result["lng"] = lng
            result["coord_system"] = coord_system

        # Add refinement evidence
        refinement_details = []
        if landmark_names:
            refinement_details.append(f"周边景点: {', '.join(landmark_names[:3])}")
        if urban_names:
            refinement_details.append(f"周边商圈: {', '.join(urban_names[:2])}")
        if transit_names:
            refinement_details.append(f"附近地铁: {', '.join(transit_names[:2])}")
        if fingerprint_features:
            refinement_details.append(f"城市特征: {'; '.join(fingerprint_features[:2])}")

        if refinement_details:
            old_reasoning = result.get("reasoning", "")
            summary = " | ".join(refinement_details)
            result["reasoning"] = old_reasoning + "\n\n[结果丰富化] " + summary
            result.setdefault("evidence", []).append({
                "source": "result_enrichment",
                "direction": "context",
                "raw_score": None,
                "calibrated_contribution": None,
                "summary": summary,
                "verifiable_unique": False,
            })

        state["result"] = result
        elapsed = int((time.time() - t0) * 1000)

        await push_step(state, 11, "result_enrichment", f"结果丰富化: {city or address}",
                        "done", {
                            "district": district,
                            "lat": lat, "lng": lng,
                            "landmarks": landmark_names[:5],
                            "urban_pois": urban_names,
                            "transit": transit_names,
                            "city_features_matched": fingerprint_features,
                        }, elapsed, 97)

    except Exception as e:
        logger.warning("result_enrichment_failed", error=str(e))
        # Preserve original result on failure

    return state


# Backward-compatible import for older graph snapshots.
fine_localize_node = result_enrichment_node


# ============================================================
# Verification Conditional Edge
# ============================================================

def _set_best_result_on_conflict(state: AgentState, hypotheses: list,
                                  history: list[str], latest_suggestion: str):
    """When verification ping-pongs, recalculate scores using ALL evidence from
    tool calls + verification history, then pick the best hypothesis."""
    if not hypotheses:
        return

    tool_calls = state.get("tool_calls", [])

    for h in hypotheses:
        province = h.get("province", "")
        # Start from base score (clamped)
        base = max(h.get("score", 0.1), 0.05)
        supporting = len(h.get("supporting_evidence", []))
        contradicting = len(h.get("contradicting_evidence", []))

        # ---- Bonus from tool calls that match this hypothesis ----
        tool_bonus = 0.0
        for tc in tool_calls:
            if tc.get("status") != "success":
                continue
            output = tc.get("output")
            if not output:
                continue
            # Extract province from tool output
            result_province = _extract_province_from_result({"data": output})
            if result_province and province in result_province:
                tool_name = tc.get("tool_name", "")
                if tool_name in ("geocode", "reverse_geocode", "search_nearby"):
                    tool_bonus += 0.15  # Amap geo results are authoritative
                else:
                    tool_bonus += 0.06

        # ---- Bonus from verification history that mentions this province ----
        verify_bonus = 0.0
        short_province = province.replace("壮族自治区", "").replace("回族自治区", "").replace("维吾尔自治区", "").replace("自治区", "").replace("省", "").replace("市", "")
        for entry in history + [latest_suggestion]:
            if short_province in entry:
                verify_bonus += 0.03  # small bonus for being mentioned in verification

        # Recalculate final score
        evidence_weight = min(supporting * 0.08 + tool_bonus, 0.30)
        penalty = contradicting * 0.10
        new_score = round(base + evidence_weight - penalty + verify_bonus, 3)
        new_score = max(0.05, min(0.85, new_score))
        h["score"] = new_score

        logger.info("conflict_recalc",
                   province=province,
                   base=base, supporting=supporting,
                   contradicting=contradicting,
                   tool_bonus=round(tool_bonus, 3),
                   final=round(new_score, 3))

    # Pick the best hypothesis
    best = max(
        hypotheses,
        key=lambda h: h.get("score", 0) + len(h.get("supporting_evidence", [])) * 0.02
    )

    # Build result with coordinates extracted from tool calls
    coords = _extract_coords_for_result(
        {"province": best["province"], "city": best.get("city", "")}, tool_calls,
    )
    lat = coords[0] if coords else None
    lng = coords[1] if coords else None

    conflict_note = "；".join([
        f"验证轮{i+1}: {h[:80]}" for i, h in enumerate(history + [latest_suggestion])
    ])
    all_scores = ", ".join([
        f"{h['province']}={h.get('score', 0):.2f}" for h in hypotheses[:5]
    ])

    state["result"] = {
        "_action": "final_answer",
        "_reason": "verification_conflict_resolved",
        "address": f"中国·{best['province']}" + (f"·{best['city']}" if best.get("city") else ""),
        "country": "中国",
        "province": best["province"],
        "city": best.get("city", ""),
        "lat": lat, "lng": lng,
        "confidence": best.get("score", 0.3),
        "reasoning": (
            f"存在矛盾证据，综合工具调用和验证历史后选择: {best['province']}"
            f"(score={best.get('score', 0):.2f})。"
            f"所有候选分数: {all_scores}。"
            f"验证历史: {conflict_note}"
        ),
    }
    logger.info("verification_conflict_resolved",
               chosen=best.get("province"), scores=all_scores)


def should_retry_after_verify(state: AgentState) -> str:
    """Determine whether to retry ReAct after verification failure.

    Tracks verification history to detect ping-pong loops (e.g. Guangdong→Fujian→Guangdong).
    """
    if state.get("verification_passed", True):
        return END

    feedback = state.get("verification_feedback", "")
    try:
        verification = json.loads(feedback) if isinstance(feedback, str) else feedback
    except (json.JSONDecodeError, TypeError):
        return END

    suggestion = verification.get("suggested_correction") if isinstance(verification, dict) else None
    contradictions = verification.get("contradictions", []) if isinstance(verification, dict) else []
    if not suggestion or not contradictions:
        return END

    # ---- Detect ping-pong: if this suggestion was already tried before, stop ----
    history: list[str] = state.get("verification_history", [])
    current_result_province = ""
    hypotheses = state.get("hypotheses", [])
    if hypotheses:
        current_result_province = hypotheses[0].get("province", "")

    # Check if suggestion essentially matches a previous round
    for prev_suggestion in history:
        # Same target → already tried this correction
        if suggestion in prev_suggestion or prev_suggestion in suggestion:
            logger.info("verification_pingpong_detected", suggestion=suggestion, history=history)
            _set_best_result_on_conflict(state, hypotheses, history, suggestion)
            return END

    # Check if we're flipping back to a province that was just rejected
    for prev_suggestion in history:
        # Extract province names and check for flip-back
        prev_provinces = _extract_provinces_from_text(prev_suggestion)
        current_provinces = _extract_provinces_from_text(suggestion)
        if prev_provinces & current_provinces:
            logger.info("verification_pingpong_province", current=current_provinces,
                       previous=prev_provinces)
            _set_best_result_on_conflict(state, hypotheses, history, suggestion)
            return END

    history.append(suggestion)
    state["verification_history"] = history
    # ----

    if hypotheses:
        hypotheses[0]["score"] = round(hypotheses[0]["score"] * 0.5, 3)
        for c in contradictions[:3]:
            hypotheses[0]["contradicting_evidence"].append({
                "clue": f"对抗验证驳回: {c}", "weight": -0.5,
            })

    # Extract clean province names (supports multiple: "广东或海南" → ["广东省", "海南省"])
    clean_provinces = _clean_suggestion_to_provinces(suggestion)
    # Insert corrections as top hypotheses (first one gets highest score)
    for idx, cp in enumerate(clean_provinces):
        score = 0.60 - idx * 0.05  # First: 0.60, Second: 0.55, etc.
        hypotheses.insert(idx, {
            "province": cp, "city": None, "score": score,
            "supporting_evidence": [
                {"clue": f"对抗验证纠正: {suggestion}", "weight": score}
            ],
            "contradicting_evidence": [],
            "source": "adversarial_verify",
            "round_created": state.get("loop_count", 0),
        })
    state["hypotheses"] = hypotheses

    # Inject forced review message
    contradiction_text = "\n".join([f"  {i+1}. {c}" for i, c in enumerate(contradictions[:3])])
    review_message = HumanMessage(content=f"""⚠️ 对抗验证驳回：你的上一轮答案被驳回，必须重新审查。

发现矛盾：
{contradiction_text}

纠正方向：{suggestion}

请针对以上矛盾，搜索 {suggestion} 相关的证据来验证纠正方向是否正确。如果纠正方向被证据支持，输出 final_answer；如果纠正方向也被否定，说明理由并给出新的推测。""")
    state["messages"] = state.get("messages", []) + [review_message]

    state["loop_count"] = min(state.get("loop_count", 0), 7)
    state["result"] = None
    state["verification_passed"] = None
    state["verification_feedback"] = None
    return "react_loop"


def _extract_provinces_from_text(text: str) -> set[str]:
    """Extract Chinese province names from a text for ping-pong detection."""
    provinces = {"广东", "福建", "广西", "海南", "云南", "贵州", "四川", "湖南", "湖北",
                 "江西", "浙江", "江苏", "安徽", "河南", "河北", "山东", "山西", "陕西",
                 "甘肃", "青海", "宁夏", "新疆", "西藏", "内蒙古", "辽宁", "吉林", "黑龙江",
                 "北京", "上海", "天津", "重庆", "台湾", "香港", "澳门"}
    found = set()
    for p in provinces:
        if p in text:
            found.add(p)
    return found


# Province short name → full administrative name
_PROVINCE_SHORT_TO_FULL: dict[str, str] = {
    "北京": "北京市", "上海": "上海市", "天津": "天津市", "重庆": "重庆市",
    "内蒙古": "内蒙古自治区", "广西": "广西壮族自治区",
    "西藏": "西藏自治区", "宁夏": "宁夏回族自治区", "新疆": "新疆维吾尔自治区",
}
for _p in ["广东", "福建", "海南", "云南", "贵州", "四川", "湖南", "湖北",
           "江西", "浙江", "江苏", "安徽", "河南", "河北", "山东", "山西", "陕西",
           "甘肃", "青海", "辽宁", "吉林", "黑龙江"]:
    _PROVINCE_SHORT_TO_FULL[_p] = _p + "省"


def _clean_suggestion_to_provinces(suggestion: str) -> list[str]:
    """Extract clean province names from verification suggestion text.

    '建议重新搜索广东省 | null' → ['广东省']
    '建议重新搜索广东省或海南省' → ['广东省', '海南省']
    '福建省泉州市石狮市' → ['福建省']
    """
    found = []
    for short in sorted(_PROVINCE_SHORT_TO_FULL.keys(), key=len, reverse=True):
        if short in suggestion:
            name = _PROVINCE_SHORT_TO_FULL[short]
            if name not in found:
                found.append(name)
    if found:
        return found
    return [suggestion[:20]]
