from langchain.tools import tool
import structlog
from app.config import get_settings

logger = structlog.get_logger()


@tool
def reverse_image_search(image_base64: str, context: str = "") -> list[dict]:
    """对图片进行反向搜索，查找相似图片在互联网上的位置。
    适用于地标建筑、旅游景点、知名街道等标志性场景。

    Args:
        image_base64: 压缩后的图片 base64
        context: 补充搜索关键词（如 "landmark" "tourism"）

    Returns:
        [{"title": str, "url": str, "snippet": str, "source": str}, ...]
    """
    settings = get_settings()
    if settings.reverse_image_service == "none":
        return [{"title": "以图搜图未启用", "url": "", "snippet": "请在 .env 中配置 REVERSE_IMAGE_SERVICE", "source": "none"}]

    try:
        if settings.reverse_image_service == "bing_visual" and settings.bing_visual_api_key:
            return _bing_visual_search(image_base64, context, settings.bing_visual_api_key)
        else:
            return [{"title": "以图搜图不可用", "url": "", "snippet": "API Key 未配置", "source": "none"}]
    except Exception as e:
        logger.error("reverse_image_error", error=str(e))
        raise


def _bing_visual_search(image_base64: str, context: str, api_key: str) -> list[dict]:
    import requests
    import io
    import base64

    img_bytes = base64.b64decode(image_base64)
    files = {"image": ("image.jpg", io.BytesIO(img_bytes), "image/jpeg")}
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    params = {"safeSearch": "Strict"}
    if context:
        params["q"] = context

    resp = requests.post(
        "https://api.bing.microsoft.com/v7.0/images/visualsearch",
        files=files, headers=headers, params=params, timeout=12,
    )
    resp.raise_for_status()
    data = resp.json()

    results = []
    for tag in data.get("tags", [])[:5]:
        for action in tag.get("actions", [])[:1]:
            results.append({
                "title": action.get("displayName", tag.get("displayName", "")),
                "url": action.get("actionUrl", ""),
                "snippet": tag.get("displayName", ""),
                "source": "bing_visual",
            })
    return results[:5]
