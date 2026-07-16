import structlog
from langchain.tools import tool
from app.config import get_settings

logger = structlog.get_logger()


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

    try:
        if settings.search_service == "serpapi" and settings.serpapi_api_key:
            return _search_serpapi(query, settings.serpapi_api_key)
        elif settings.bing_search_api_key:
            return _search_bing(query, settings.bing_search_api_key)
        else:
            logger.warning("search_no_api_key")
            return [{"title": "搜索不可用", "snippet": "未配置搜索 API Key", "url": ""}]
    except Exception as e:
        logger.error("search_error", error=str(e))
        raise


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
