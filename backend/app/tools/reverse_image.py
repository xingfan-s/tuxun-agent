"""
Reverse image search tool.

Multi-engine reverse image search via PicImageSearch library:
  Baidu + GoogleLens + Bing + Yandex (parallel via asyncio, 12s timeout per engine)
"""

import asyncio
import base64
from app.utils.logging import structlog

from langchain.tools import tool
from app.config import get_settings

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# PicImageSearch engines (lazy imports)
# ---------------------------------------------------------------------------

_engines = None


def _get_engines():
    """Lazy-init PicImageSearch engine clients."""
    global _engines
    if _engines is None:
        from PicImageSearch.sync import BaiDu, GoogleLens, Yandex, Bing

        _engines = {
            "baidu": BaiDu(),
            "google_lens": GoogleLens(hl="zh-CN", country="CN",
                                       search_type="all"),
            "bing": Bing(),
            "yandex": Yandex(),
        }
    return _engines


# ---------------------------------------------------------------------------
# Main tool (sync wrapper for LangChain compatibility)
# ---------------------------------------------------------------------------

@tool
def reverse_image_search(image_base64: str, context: str = "") -> list[dict]:
    """对图片进行反向搜索，查找相似图片在互联网上的来源。

    使用多引擎并行搜索（百度识图 + Google Lens + Bing + Yandex）。
    搜索结果中的域名、标题、Bing实体识别等信息可能包含地理位置线索，
    需要配合 search_place 或 geocode 进一步确认具体地点。

    Args:
        image_base64: 图片的 base64 编码
        context: 补充搜索关键词

    Returns:
        [{"title": str, "url": str, "snippet": str, "source": str}, ...]
    """
    try:
        # Run the async multi-engine search from sync context
        img_bytes = base64.b64decode(image_base64)
        return _multi_engine_search_sync(img_bytes, context)
    except Exception as e:
        logger.warning("multi_engine_search_failed", error=str(e))

    return [{"title": "以图搜图无结果", "url": "", "snippet": "请尝试使用 extract_china_clues 提取文字后通过 search_place 搜索", "source": "error"}]


# ======================================================================
# Async multi-engine search (called from _execute_tool with proper timeout)
# ======================================================================

_ENGINE_TIMEOUT = 12


async def reverse_image_search_async(image_base64: str, context: str = "") -> list[dict]:
    """Async version: search all engines in parallel via asyncio.to_thread.

    Called directly from _execute_tool in nodes.py to avoid nested
    ThreadPoolExecutor inside asyncio.to_thread.
    """
    if get_settings().reverse_image_service.lower() in {"none", "disabled", "off"}:
        logger.info("reverse_image_search_skipped", reason="service_disabled")
        return [{
            "title": "以图搜图已禁用",
            "url": "",
            "snippet": "REVERSE_IMAGE_SERVICE=none",
            "source": "disabled",
        }]
    try:
        engines = _get_engines()
        img_bytes = base64.b64decode(image_base64)
        results = await _search_all_engines(engines, img_bytes, context)
        if results:
            return results
    except Exception as e:
        logger.warning("async_multi_engine_search_failed", error=str(e))

    return [{"title": "以图搜图无结果", "url": "",
             "snippet": "请尝试使用 extract_china_clues 提取文字后通过 search_place 搜索", "source": "error"}]


async def _search_all_engines(engines: dict, img_bytes: bytes, context: str) -> list[dict]:
    """Run reverse image search on all engines in parallel via asyncio."""
    all_results: list[dict] = []
    seen_urls: set[str] = set()

    async def _search_one(name: str, engine) -> list[dict]:
        try:
            resp = await asyncio.wait_for(
                asyncio.to_thread(engine.search, file=img_bytes),
                timeout=_ENGINE_TIMEOUT,
            )
            return _parse_engine_response(name, resp, context)
        except asyncio.TimeoutError:
            logger.warning("engine_timeout", engine=name)
            return []
        except Exception as exc:
            logger.warning("engine_search_error", engine=name, error=str(exc))
            return []

    engine_results = await asyncio.gather(*[
        _search_one(name, engine) for name, engine in engines.items()
    ])

    for results in engine_results:
        for r in results:
            key = r.get("url", "") or r.get("title", "")
            if key and key not in seen_urls:
                seen_urls.add(key)
                all_results.append(r)

    return all_results[:10]


def _multi_engine_search_sync(img_bytes: bytes, context: str) -> list[dict]:
    """Sync fallback: used when called outside an async context."""
    import concurrent.futures
    engines = _get_engines()

    all_results: list[dict] = []
    seen_urls: set[str] = set()

    def _search_one(name: str, engine) -> list[dict]:
        try:
            resp = engine.search(file=img_bytes)
        except Exception as exc:
            logger.warning("engine_search_error", engine=name, error=str(exc))
            return []
        return _parse_engine_response(name, resp, context)

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    try:
        futures = {
            pool.submit(_search_one, name, engine): name
            for name, engine in engines.items()
        }
        for future in concurrent.futures.as_completed(futures, timeout=_ENGINE_TIMEOUT):
            name = futures[future]
            try:
                results = future.result(timeout=0)
            except Exception as exc:
                logger.warning("engine_future_error", engine=name, error=str(exc))
                continue

            for r in results:
                key = r.get("url", "") or r.get("title", "")
                if key and key not in seen_urls:
                    seen_urls.add(key)
                    all_results.append(r)
    except TimeoutError:
        logger.warning("multi_engine_timeout", finished=len(all_results))
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    return all_results[:10]


# ======================================================================
# Engine-specific response parsers
# ======================================================================

def _parse_engine_response(engine_name: str, resp, context: str) -> list[dict]:
    """Parse a PicImageSearch response into structured results."""

    if engine_name == "baidu":
        return _parse_baidu_response(resp)
    elif engine_name == "google_lens":
        return _parse_google_lens_response(resp)
    elif engine_name == "bing":
        return _parse_bing_response(resp)
    elif engine_name == "yandex":
        return _parse_yandex_response(resp)

    return []


def _parse_baidu_response(resp) -> list[dict]:
    """Parse BaiDuResponse from PicImageSearch."""
    results = []

    for item in resp.exact_matches[:5]:
        if item.url:
            domain = _extract_domain(item.url)
            results.append({
                "title": item.title or f"百度相同图片: {domain}",
                "snippet": f"[百度·相同图片] {domain}",
                "url": item.url,
                "source": "baidu_exact",
            })

    for item in resp.raw[:8]:
        if item.url:
            domain = _extract_domain(item.url)
            results.append({
                "title": item.title or f"百度相似图片: {domain}",
                "snippet": f"[百度·相似图片] {domain}",
                "url": item.url,
                "source": "baidu_similar",
            })

    return results[:8]


def _parse_google_lens_response(resp) -> list[dict]:
    """Parse GoogleLensResponse from PicImageSearch."""
    results = []

    for item in resp.raw[:10]:
        if item.url:
            site = getattr(item, "site_name", "") or _extract_domain(item.url)
            title = getattr(item, "title", "") or ""
            results.append({
                "title": title or f"Google Lens 匹配: {site}",
                "snippet": f"[Google Lens] {site}" + (f": {title}" if title else ""),
                "url": item.url,
                "source": "google_lens",
            })

    for item in getattr(resp, "related_searches", [])[:5]:
        if getattr(item, "title", ""):
            results.append({
                "title": item.title,
                "snippet": f"[Google Lens·相关搜索] {item.title}",
                "url": getattr(item, "url", ""),
                "source": "google_lens_related",
            })

    return results[:10]


def _parse_bing_response(resp) -> list[dict]:
    """Parse BingResponse from PicImageSearch.

    Bing returns rich structured data: entities, travel info, best guess,
    pages including the image, and visually similar images.
    """
    results = []

    if resp.best_guess:
        results.append({
            "title": f"Bing 推测: {resp.best_guess}",
            "snippet": f"[Bing·最佳推测] {resp.best_guess}",
            "url": "",
            "source": "bing_best_guess",
        })

    for entity in resp.entities[:5]:
        name = entity.name or ""
        desc = entity.description or entity.short_description or ""
        snippet = f"[Bing·实体识别] {name}"
        if desc:
            snippet += f" — {desc}"
        if entity.profiles:
            snippet += " | " + ", ".join(
                p.get("social_network", "") for p in entity.profiles[:2]
            )
        results.append({
            "title": name or "Bing 实体",
            "snippet": snippet,
            "url": "",
            "source": "bing_entity",
        })

    if resp.travel:
        travel = resp.travel
        if travel.destination_name:
            results.append({
                "title": f"目的地: {travel.destination_name}",
                "snippet": f"[Bing·旅行信息] 目的地: {travel.destination_name}",
                "url": travel.travel_guide_url or "",
                "source": "bing_travel",
            })
        for attr in travel.attractions[:3]:
            if attr.title:
                results.append({
                    "title": f"景点: {attr.title}",
                    "snippet": f"[Bing·景点] {attr.title} | 类型: {', '.join(attr.interest_types[:3]) if attr.interest_types else '未知'}",
                    "url": attr.url or "",
                    "source": "bing_attraction",
                })

    for page in resp.pages_including[:5]:
        if page.url:
            domain = _extract_domain(page.url)
            results.append({
                "title": page.name or f"Bing 包含页面: {domain}",
                "snippet": f"[Bing·包含此图片的页面] {domain}",
                "url": page.url,
                "source": "bing_pages",
            })

    for item in resp.visual_search[:5]:
        if item.url:
            domain = _extract_domain(item.url)
            results.append({
                "title": item.name or f"Bing 相似图片: {domain}",
                "snippet": f"[Bing·相似图片] {domain}",
                "url": item.url,
                "source": "bing_visual",
            })

    return results[:12]


def _parse_yandex_response(resp) -> list[dict]:
    """Parse YandexResponse from PicImageSearch."""
    results = []

    for item in resp.raw[:10]:
        if item.url:
            domain = getattr(item, "source", "") or _extract_domain(item.url)
            title = getattr(item, "title", "") or ""
            content = getattr(item, "content", "") or ""
            snippet = f"[Yandex] {domain}"
            if content:
                snippet += f" — {content[:120]}"
            results.append({
                "title": title or f"Yandex 匹配: {domain}",
                "snippet": snippet,
                "url": item.url,
                "source": "yandex",
            })

    return results[:8]


def _extract_domain(url: str) -> str:
    import re
    m = re.search(r"https?://([^/]+)", url)
    return m.group(1) if m else url
