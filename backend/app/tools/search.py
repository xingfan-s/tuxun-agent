import httpx
from app.utils.logging import structlog
from langchain.tools import tool
from app.config import get_settings

logger = structlog.get_logger()

_FAKE_KEYS = {"xxxxxxxx", ""}


@tool
def search_place(query: str, region: str = "") -> list[dict]:
    """搜索地点、店名、路牌上的文字。适用于查找商家、地标、道路名称。

    Args:
        query: 搜索关键词，如店名、路名、地标名
        region: 可选，限定搜索区域，如 "俄罗斯" "日本"

    Returns:
        [{"title": str, "snippet": str, "url": str}, ...]
    """
    settings = get_settings()
    if region:
        query = f"{query} {region}"

    # 中国地理定位优先高德POI搜索（精度最高，数据最全）
    amap_results = []
    try:
        if settings.amap_server_api_key not in _FAKE_KEYS:
            amap_results = _search_amap_poi(query, region, settings.amap_server_api_key)
    except Exception as e:
        logger.warning("search_amap_failed", error=str(e))

    # 补充通用搜索作为辅助
    web_results = []
    try:
        if settings.search_service == "serpapi" and settings.serpapi_api_key not in _FAKE_KEYS:
            web_results = _search_serpapi(query, settings.serpapi_api_key)
    except Exception as e:
        logger.warning("search_serpapi_failed", error=str(e))

    if not web_results:
        try:
            if settings.bing_search_api_key not in _FAKE_KEYS:
                web_results = _search_bing(query, settings.bing_search_api_key)
        except Exception as e:
            logger.warning("search_bing_failed", error=str(e))

    # 高德结果在前（精度高），通用搜索在后
    results = amap_results + web_results
    if results:
        return results

    logger.warning("search_all_failed")
    return [{"title": "搜索不可用", "snippet": "所有搜索服务均失败或未配置", "url": ""}]


def _search_serpapi(query: str, api_key: str) -> list[dict]:
    import requests
    resp = requests.get("https://serpapi.com/search", params={
        "q": query, "api_key": api_key, "engine": "google", "num": 5,
    }, timeout=8)
    resp.raise_for_status()
    data = resp.json()
    results = []
    for r in data.get("organic_results", [])[:5]:
        results.append({"title": r.get("title", ""), "snippet": r.get("snippet", ""),
                        "url": r.get("link", "")})
    return results


def _search_bing(query: str, api_key: str) -> list[dict]:
    import requests
    resp = requests.get("https://api.bing.microsoft.com/v7.0/search", params={
        "q": query, "count": 5, "mkt": "zh-CN",
    }, headers={"Ocp-Apim-Subscription-Key": api_key}, timeout=8)
    resp.raise_for_status()
    data = resp.json()
    results = []
    for r in data.get("webPages", {}).get("value", [])[:5]:
        results.append({"title": r.get("name", ""), "snippet": r.get("snippet", ""),
                        "url": r.get("url", "")})
    return results


def _search_amap_poi(query: str, region: str, api_key: str) -> list[dict]:
    """Use Amap POI text search as search_place backend."""
    try:
        params = {"key": api_key, "keywords": query, "offset": 5}
        if region:
            params["city"] = region
        with httpx.Client(timeout=10) as client:
            resp = client.get(
                "https://restapi.amap.com/v3/place/text", params=params
            )
        data = resp.json()
        if data.get("status") != "1":
            return []
        results = []
        for p in data.get("pois", [])[:5]:
            results.append({
                "title": p.get("name", ""),
                "snippet": f"{p.get('address', '')} | {p.get('type', '')}",
                "url": "",
            })
        return results
    except Exception:
        return []
