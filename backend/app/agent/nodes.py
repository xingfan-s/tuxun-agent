import time
import json
import structlog
from openai import OpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from app.config import get_settings
from app.agent.state import AgentState
from app.agent.prompts import (
    VISION_ANALYSIS_PROMPT, CLUE_EXTRACTION_PROMPT,
    REACT_SYSTEM_PROMPT, RESULT_SYNTHESIS_PROMPT,
)
from app.safety import run_safety_check
from app.tools.exif import extract_exif
from app.tools.search import search_place
from app.tools.landmark import search_landmark
from app.tools.reverse_image import reverse_image_search
from app.tools.map import create_map_service

logger = structlog.get_logger()

TOOL_MAP = {
    "search_place": search_place,
    "search_landmark": search_landmark,
    "reverse_image_search": reverse_image_search,
}

MAP_TOOLS = {"geocode", "reverse_geocode", "search_nearby", "get_streetview"}

TOOL_TIMEOUTS = {
    "search_place": 15, "geocode": 15, "reverse_geocode": 15,
    "search_nearby": 20, "get_streetview": 20, "extract_exif": 3,
    "search_landmark": 15, "reverse_image_search": 15,
}


async def push_step(state: AgentState, step_num: int, step_type: str,
                     label: str, status: str, data: dict, elapsed_ms: int, progress: int):
    """Push a step update via SSE callback."""
    cb = state.get("stream_callback")
    if cb:
        await cb({
            "event": "step_update",
            "data": {
                "step": step_num, "type": step_type, "label": label,
                "status": status, "data": data, "elapsed_ms": elapsed_ms,
            }
        })
        await cb({"event": "progress", "data": {"progress": progress}})


def _get_llm_client():
    settings = get_settings()
    return OpenAI(api_key=settings.qwen_api_key, base_url=settings.qwen_base_url)


_ROLE_MAP = {"human": "user", "ai": "assistant", "system": "system"}


def _llm_chat(messages: list, model: str = None, temperature: float = 0.1, max_tokens: int = 2000) -> str:
    settings = get_settings()
    client = _get_llm_client()

    formatted = []
    for m in messages:
        if isinstance(m, dict):
            formatted.append(m)
        else:
            role = _ROLE_MAP.get(getattr(m, "type", None), "user")
            formatted.append({"role": role, "content": m.content})

    response = client.chat.completions.create(
        model=model or settings.qwen_model,
        messages=formatted,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def _llm_vision(image_base64: str, prompt: str) -> str:
    settings = get_settings()
    client = _get_llm_client()
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
        max_tokens=2000,
    )
    return response.choices[0].message.content


async def safety_check_node(state: AgentState) -> AgentState:
    """Node 0: Safety check."""
    t0 = time.time()
    result = run_safety_check(state["image_base64"])
    elapsed = int((time.time() - t0) * 1000)

    await push_step(state, 0, "safety_check", "安全预检",
                    "done", result, elapsed, 5)

    state["safety_passed"] = result["passed"]
    state["safety_reason"] = result.get("reason")
    return state


async def exif_extract_node(state: AgentState) -> AgentState:
    """Node 1: EXIF extraction."""
    t0 = time.time()
    exif_data = extract_exif.invoke({"image_path": state["image_path"]})
    elapsed = int((time.time() - t0) * 1000)

    await push_step(state, 1, "exif", "EXIF 提取",
                    "done", exif_data, elapsed, 15)

    state["exif_data"] = exif_data
    return state


async def vision_analyze_node(state: AgentState) -> AgentState:
    """Node 2: Vision analysis with Qwen-VL."""
    t0 = time.time()
    vision_raw = _llm_vision(state["image_base64"], VISION_ANALYSIS_PROMPT)
    elapsed = int((time.time() - t0) * 1000)

    await push_step(state, 2, "vision", "视觉分析",
                    "done", {"description": vision_raw[:200] + "...", "raw_output": vision_raw},
                    elapsed, 30)

    state["vision_raw"] = vision_raw
    return state


async def clue_extract_node(state: AgentState) -> AgentState:
    """Node 3: Clue extraction."""
    t0 = time.time()
    prompt = CLUE_EXTRACTION_PROMPT.format(vision_raw=state["vision_raw"])
    response = _llm_chat([HumanMessage(content=prompt)], temperature=0.1, max_tokens=1000)

    try:
        clues = json.loads(response)
    except json.JSONDecodeError:
        clues = {"raw": response, "parse_error": True}

    elapsed = int((time.time() - t0) * 1000)

    await push_step(state, 3, "clue_extraction", "线索提取",
                    "done", {"clues": clues}, elapsed, 40)

    state["clues"] = clues
    return state


def _parse_decision(response: str) -> dict | None:
    """Robust JSON extraction from LLM response. Handles markdown code blocks
    and common key name variations."""
    import re

    # Try extracting from ```json ... ``` block
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Try extracting the first { } block
    m = re.search(r"\{.*\}", response, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    # Try raw parse
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    return None


def _extract_location_from_result(tool_result: dict) -> list[float]:
    """Extract lat/lng list from tool result data."""
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


def _check_climate_mismatch(lats: list[float], clues: dict) -> str | None:
    """Rule-based climate/vegetation consistency check. Returns warning or None."""
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
        return f"搜索结果在纬度{avg_lat:.0f}°，但图中可见热带植被（{set(veg) & tropical_veg}），不可能在这个纬度"
    if hemi == "north" and avg_lat < -5:
        return f"搜索结果在南半球（{avg_lat:.0f}°），但视觉提示北半球"
    if hemi == "south" and avg_lat > 5:
        return f"搜索结果在北半球（{avg_lat:.0f}°），但视觉提示南半球"
    return None


async def _cross_validate(state: AgentState, tool_name: str, tool_result: dict,
                          tool_entry: dict) -> dict | None:
    """Cross-validate tool results against visual clues.
    If inconsistent, ask LLM to judge and return corrected region.
    Returns None if consistent, or dict with corrected action."""
    clues = state.get("clues") or {}
    lats = _extract_location_from_result(tool_result)
    if not lats:
        return None

    warning = _check_climate_mismatch(lats, clues)
    if not warning:
        return None

    logger.info("cross_validate_mismatch", tool=tool_name, warning=warning)

    # Ask LLM to validate and redirect
    prompt = f"""视觉线索: {json.dumps(clues, ensure_ascii=False)}
工具 "{tool_name}" 的搜索结果: {json.dumps(tool_result.get('data'), ensure_ascii=False, default=str)[:500]}

{chr(10).join(['⚠️ 矛盾检测: ' + warning])}

请判断：搜索结果是否与视觉线索矛盾？如果矛盾，应该在哪座城市或省份重新搜索？输出JSON:
{{"consistent": true/false, "correct_region": "城市或省份名", "reasoning": "判断理由"}}"""

    try:
        response = _llm_chat([HumanMessage(content=prompt)], temperature=0, max_tokens=300)
        decision = _parse_decision(response)
        if decision and not decision.get("consistent", True):
            return decision
    except Exception as e:
        logger.warning("cross_validate_error", error=str(e))
    return None


async def _apply_redirect(state: AgentState, original_tool: str,
                          redirect: dict, loop_count: int, max_loops: int):
    """Apply LLM redirect: geocode the corrected region and search nearby."""
    region = redirect.get("correct_region", "")
    if not region:
        return

    logger.info("cross_validate_redirect", region=region, reason=redirect.get("reasoning", ""))

    # Step 1: geocode the corrected region
    map_service = create_map_service()
    geo_results = await _execute_tool("geocode", {"address": region}, state)
    geo_lat = geo_lng = None
    if geo_results["status"] == "success" and geo_results.get("data"):
        items = geo_results["data"] if isinstance(geo_results["data"], list) else [geo_results["data"]]
        if items:
            first = items[0]
            geo_lat = first.get("lat") if isinstance(first, dict) else (getattr(first, "lat", None))
            geo_lng = first.get("lng") if isinstance(first, dict) else (getattr(first, "lng", None))

    # Step 2: add correction info to state
    state["tool_calls"] = state.get("tool_calls", []) + [{
        "tool_name": "redirect",
        "status": "success",
        "input": {"original_tool": original_tool, "warning": redirect.get("reasoning", "")},
        "output": f"已纠正搜索范围 → {region} ({geo_lat}, {geo_lng})",
        "duration_ms": 0,
    }]
    state["messages"] = state.get("messages", []) + [
        AIMessage(content=f"[交叉验证] 视觉线索与搜索结果矛盾，LLM判断应重新搜索: {region}。原因: {redirect.get('reasoning', '')}"),
    ]

    # Step 3: re-search in corrected region
    if original_tool in ("search_place", "search_nearby", "search_landmark") and geo_lat and geo_lng:
        tool_input = state["tool_calls"][-2].get("input", {}) if len(state.get("tool_calls", [])) >= 2 else {}
        keyword = tool_input.get("keyword") or tool_input.get("query") or tool_input.get("description") or region
        await push_step(state, 4, "tool_call", f"纠正搜索: search_nearby({region})",
                        "running", {"lat": geo_lat, "lng": geo_lng, "keyword": keyword}, 0,
                        40 + int(50 * loop_count / max_loops))
        result = await _execute_tool("search_nearby",
                                     {"lat": geo_lat, "lng": geo_lng, "keyword": keyword, "radius": 10000},
                                     state)
        state["tool_calls"] = state.get("tool_calls", []) + [{
            "tool_name": "search_nearby", "status": result["status"],
            "input": {"lat": geo_lat, "lng": geo_lng, "keyword": keyword, "radius": 10000},
            "output": result.get("data"),
            "error": result.get("error"),
            "duration_ms": 0,
            "redirected": True,
        }]


async def react_loop_node(state: AgentState) -> AgentState:
    """Node 4: ReAct reasoning loop."""
    settings = get_settings()
    max_loops = settings.max_react_loops
    loop_count = state.get("loop_count", 0) + 1
    state["loop_count"] = loop_count

    if loop_count > max_loops:
        state["result"] = {"_action": "final_answer", "_reason": "max_loops_reached"}
        return state

    tool_descriptions = "\n".join([
        "- search_place(query, region?): 搜索地点、店名、路牌文字",
        "- geocode(address): 地址→经纬度",
        "- reverse_geocode(lat, lng): 经纬度→地址",
        "- search_nearby(lat, lng, keyword, radius=5000): 周边POI搜索",
        "- get_streetview(lat, lng): 拉取街景图",
        "- search_landmark(description): 搜索著名地标",
        "- reverse_image_search(image_base64, context?): 以图搜图",
    ])

    tool_results_text = json.dumps(state.get("tool_calls", []), ensure_ascii=False, indent=2)
    failed_tools_text = ", ".join(state.get("failed_tools", set())) or "无"

    messages = [
        SystemMessage(content=REACT_SYSTEM_PROMPT.format(
            tool_descriptions=tool_descriptions,
            clues=json.dumps(state.get("clues", {}), ensure_ascii=False, indent=2),
            exif_data=json.dumps(state.get("exif_data", {}), ensure_ascii=False),
            tool_results=tool_results_text,
            failed_tools=failed_tools_text,
            loop_count=loop_count,
            max_loops=max_loops,
        )),
        HumanMessage(content="请根据当前线索决定下一步动作。"),
    ]

    t0 = time.time()
    response = _llm_chat(messages, temperature=0.1, max_tokens=1500)
    elapsed = int((time.time() - t0) * 1000)

    decision = _parse_decision(response)
    if decision is None:
        logger.warning("react_parse_error", response=response[:200])
        state["messages"] = state.get("messages", []) + [
            AIMessage(content=response),
            HumanMessage(content="你的上一轮输出无法解析为JSON。请严格按照JSON格式输出你的决策，包含 action 和 action_input 字段。"),
        ]
        return state

    if "address" in decision or decision.get("action") == "final_answer":
        state["result"] = decision
        await push_step(state, 4, "reasoning", f"推理第{loop_count}轮",
                        "done", {"thought": decision.get("reasoning", ""), "action": "final_answer", "action_input": None},
                        elapsed, 40 + int(50 * loop_count / max_loops))
        return state

    tool_name = (decision.get("action") or decision.get("tool_name")
                 or decision.get("tool") or decision.get("tool_call") or "")
    tool_input = (decision.get("action_input") or decision.get("tool_input")
                  or decision.get("query") or {})
    if isinstance(tool_input, str):
        tool_input = {"query": tool_input}

    if tool_name in state.get("failed_tools", set()):
        state["tool_calls"] = state.get("tool_calls", []) + [{
            "tool_name": tool_name, "status": "skipped", "reason": "previously_failed",
        }]
        return state

    await push_step(state, 4, "tool_call", f"工具调用: {tool_name}",
                    "running", {"tool_name": tool_name, "input": tool_input}, 0,
                    40 + int(50 * loop_count / max_loops))

    t_tool = time.time()
    tool_result = await _execute_tool(tool_name, tool_input, state)
    tool_elapsed = int((time.time() - t_tool) * 1000)

    tool_entry = {
        "tool_name": tool_name, "status": tool_result["status"],
        "input": tool_input, "output": tool_result.get("data"),
        "error": tool_result.get("error"),
        "duration_ms": tool_elapsed,
    }
    state["tool_calls"] = state.get("tool_calls", []) + [tool_entry]

    if tool_result["status"] in ("timeout", "failed"):
        state["failed_tools"] = state.get("failed_tools", set()) | {tool_name}
        cb = state.get("stream_callback")
        if cb:
            await cb({
                "event": "tool_warning",
                "data": {
                    "tool": tool_name, "reason": tool_result["status"],
                    "message": f"工具 {tool_name} {tool_result['status']}: {tool_result.get('error', '')}",
                }
            })

    # Cross-validate tool results against visual clues
    if tool_result["status"] == "success":
        redirect = await _cross_validate(state, tool_name, tool_result, tool_entry)
        if redirect:
            # LLM redirected: auto re-search or geocode with corrected region
            await _apply_redirect(state, tool_name, redirect, loop_count, max_loops)

    await push_step(state, 4, "tool_call", f"工具结果: {tool_name}",
                    "done", tool_entry, tool_elapsed,
                    40 + int(50 * loop_count / max_loops))

    if loop_count % 3 == 0:
        state["last_redirect_at"] = loop_count
        reflection = _llm_chat([
            SystemMessage(content="检查当前推理方向是否正确，是否需要修正。"),
            HumanMessage(content=f"线索: {json.dumps(state.get('clues', {}), ensure_ascii=False)}\n"
                                 f"工具结果: {json.dumps(state.get('tool_calls', []), ensure_ascii=False)}\n"
                                 f"回答'方向正确'或指出需要修正的方向。"),
        ], temperature=0.1, max_tokens=300)
        state["messages"] = state.get("messages", []) + [
            AIMessage(content=f"[反思第{loop_count}轮] {reflection}")
        ]

    return state


async def _execute_tool(tool_name: str, tool_input: dict, state: AgentState) -> dict:
    """Execute a tool with timeout and error handling."""
    import asyncio

    timeout = TOOL_TIMEOUTS.get(tool_name, 10)

    try:
        if tool_name in TOOL_MAP:
            result = await asyncio.wait_for(
                asyncio.to_thread(TOOL_MAP[tool_name].invoke, tool_input),
                timeout=timeout,
            )
            return {"status": "success", "data": result}

        elif tool_name in MAP_TOOLS:
            map_service = create_map_service()
            if tool_name == "geocode":
                result = await asyncio.wait_for(
                    map_service.geocode(tool_input.get("address", "")),
                    timeout=timeout,
                )
                return {"status": "success", "data": [vars(r) for r in result]}
            elif tool_name == "reverse_geocode":
                result = await asyncio.wait_for(
                    map_service.reverse_geocode(
                        tool_input.get("lat", 0), tool_input.get("lng", 0)
                    ), timeout=timeout,
                )
                return {"status": "success", "data": vars(result) if result else None}
            elif tool_name == "search_nearby":
                result = await asyncio.wait_for(
                    map_service.search_nearby(
                        tool_input.get("lat", 0), tool_input.get("lng", 0),
                        tool_input.get("keyword", ""), tool_input.get("radius", 5000),
                    ), timeout=timeout,
                )
                return {"status": "success", "data": [vars(r) for r in result]}
            elif tool_name == "get_streetview":
                result = await asyncio.wait_for(
                    map_service.get_streetview(
                        tool_input.get("lat", 0), tool_input.get("lng", 0),
                    ), timeout=timeout,
                )
                return {"status": "success", "data": "streetview_bytes" if result else None}

        return {"status": "failed", "error": f"Unknown tool: {tool_name}"}

    except asyncio.TimeoutError:
        return {"status": "timeout", "error": f"超时 ({timeout}s)"}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


async def result_synthesize_node(state: AgentState) -> AgentState:
    """Node 5: Result synthesis."""
    t0 = time.time()

    raw_result = state.get("result") or {}
    fmt = {
        "clues": json.dumps(state.get("clues", {}), ensure_ascii=False),
        "exif_data": json.dumps(state.get("exif_data", {}), ensure_ascii=False),
        "reasoning_history": json.dumps(state.get("tool_calls", []), ensure_ascii=False),
    }
    if raw_result.get("_action") == "final_answer" and raw_result.get("_reason") == "max_loops_reached":
        prompt_text = RESULT_SYNTHESIS_PROMPT.format(
            **fmt,
            final_output="已达到最大推理轮数，基于已有线索给出最佳推测。",
        )
        response = _llm_chat([HumanMessage(content=prompt_text)], temperature=0.1, max_tokens=1000)
        result = _parse_decision(response) or raw_result
    elif "address" in raw_result:
        result = raw_result
    else:
        prompt_text = RESULT_SYNTHESIS_PROMPT.format(
            **fmt,
            final_output=json.dumps(raw_result, ensure_ascii=False),
        )
        response = _llm_chat([HumanMessage(content=prompt_text)], temperature=0.1, max_tokens=1000)
        result = _parse_decision(response) or raw_result

    tool_calls = state.get("tool_calls", [])
    total = len(tool_calls)
    success = sum(1 for t in tool_calls if t["status"] == "success")
    timeout = sum(1 for t in tool_calls if t["status"] == "timeout")
    failed = sum(1 for t in tool_calls if t["status"] == "failed")

    result["tool_stats"] = {
        "total_calls": total, "success": success,
        "timeout": timeout, "failed": failed,
    }
    result["tokens_used"] = 0
    result["total_elapsed_ms"] = 0

    state["result"] = result

    elapsed = int((time.time() - t0) * 1000)
    await push_step(state, 5, "final", "结果整合", "done", {}, elapsed, 100)

    return state
