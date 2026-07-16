import structlog
from openai import OpenAI

from app.config import get_settings

logger = structlog.get_logger()

PRIVATE_SCENE_PROMPT = """你是一个图片安全审查助手。请判断这张图片是否属于以下任一隐私场景：

拒绝场景（返回 REJECT）：
- 室内私人空间（客厅、卧室、酒店房间、办公室内部）
- 住宅楼外观且能看到清晰的门牌号
- 包含人物面部特写的照片
- 包含身份证、快递单、银行卡等隐私信息的截图

允许场景（返回 PASS）：
- 公共街道、道路、高速公路
- 旅游景点、公园、广场
- 野外自然风景
- 商业区、购物街、市场
- 公共交通（地铁、公交、火车）
- 建筑外观（无清晰门牌号）

只回答一个词：PASS 或 REJECT，然后简要说明原因。"""


def check_scene(image_base64: str) -> tuple[bool, str]:
    """
    Returns (passed: bool, reason: str).
    passed=True means the scene is safe to analyze.
    """
    settings = get_settings()
    if not settings.qwen_api_key:
        return True, "no_api_key_configured"

    client = OpenAI(
        api_key=settings.qwen_api_key,
        base_url=settings.qwen_base_url,
    )

    try:
        response = client.chat.completions.create(
            model=settings.qwen_vl_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": PRIVATE_SCENE_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                ]
            }],
            max_tokens=50,
            temperature=0,
        )
        text = response.choices[0].message.content.strip().upper()
        if text.startswith("REJECT"):
            reason = text.replace("REJECT", "").strip(":：. ").strip()
            return False, reason or "private_scene_detected"
        return True, "scene_ok"
    except Exception as e:
        logger.warning("scene_check_error", error=str(e))
        return True, f"scene_check_skipped: {str(e)}"
