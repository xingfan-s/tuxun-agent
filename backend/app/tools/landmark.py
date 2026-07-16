from langchain.tools import tool
import structlog

logger = structlog.get_logger()


@tool
def search_landmark(description: str) -> list[dict]:
    """根据建筑/自然景观描述搜索著名地标。

    Args:
        description: 地标描述，如 "洋葱头穹顶教堂 金色"

    Returns:
        [{"title": str, "snippet": str, "url": str}, ...]
    """
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
        logger.error("landmark_search_error", error=str(e))
        raise


def _clean_html(text: str) -> str:
    import re
    return re.sub(r'<[^>]+>', '', text)
