from langchain.tools import tool
from app.utils.logging import structlog

logger = structlog.get_logger()


@tool
def search_landmark(description: str) -> list[dict]:
    """根据建筑/自然景观描述搜索著名地标。

    Args:
        description: 地标描述，如 "洋葱头穹顶教堂 金色"

    Returns:
        [{"title": str, "snippet": str, "url": str}, ...]
    """
    # Try Wikipedia first, fall back to Amap
    results = _search_wikipedia(description)
    if results:
        return results

    # Fallback: Amap POI search
    try:
        from app.config import get_settings
        settings = get_settings()
        if settings.amap_server_api_key and settings.amap_server_api_key not in {"xxxxxxxx", ""}:
            return _search_amap_poi(description, settings.amap_server_api_key)
    except Exception:
        pass

    return results


def _search_wikipedia(description: str) -> list[dict]:
    import requests
    try:
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query", "list": "search",
                "srsearch": description, "format": "json", "srlimit": 5,
            },
            headers={"User-Agent": "TuXun-Agent/0.1"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for r in data.get("query", {}).get("search", [])[:5]:
            page_url = f"https://en.wikipedia.org/?curid={r['pageid']}"
            results.append({
                "title": r.get("title", ""),
                "snippet": _clean_html(r.get("snippet", "")),
                "url": page_url,
            })
        return results
    except Exception as e:
        logger.warning("landmark_wikipedia_error", error=str(e))
        return []


def _search_amap_poi(description: str, api_key: str) -> list[dict]:
    """Use Amap POI text search for landmark queries."""
    try:
        import httpx
        params = {"key": api_key, "keywords": description, "offset": 5}
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


def _clean_html(text: str) -> str:
    import re
    return re.sub(r'<[^>]+>', '', text)
