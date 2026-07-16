from app.config import get_settings
from app.safety.face_detect import has_too_many_faces, count_faces
from app.safety.scene_check import check_scene
from app.safety.text_check import check_image_text, decode_base64_image


def run_safety_check(image_base64: str) -> dict:
    """
    Run all three safety checks. Returns:
    {
        "passed": bool,
        "reason": str | None,
        "face_count": int,
        "scene": str,
        "sensitive_text": bool,
    }
    """
    settings = get_settings()

    # Layer 1: Face detection (local, fast)
    img = decode_base64_image(image_base64)
    face_count = count_faces(img) if img is not None else 0
    if has_too_many_faces(img, settings.safety_face_max_count):
        return {
            "passed": False,
            "reason": f"检测到{face_count}个人脸，可能包含人物面部特写",
            "face_count": face_count,
            "scene": "unknown",
            "sensitive_text": False,
        }

    # Layer 2: Scene classification (Qwen-VL)
    scene_passed, scene_reason = check_scene(image_base64)
    if not scene_passed:
        return {
            "passed": False,
            "reason": scene_reason,
            "face_count": face_count,
            "scene": scene_reason,
            "sensitive_text": False,
        }

    # Layer 3: Sensitive text (OCR)
    has_sensitive, sensitive_types = check_image_text(image_base64)
    if has_sensitive:
        return {
            "passed": False,
            "reason": f"检测到敏感信息: {', '.join(sensitive_types)}",
            "face_count": face_count,
            "scene": "public" if scene_passed else scene_reason,
            "sensitive_text": True,
        }

    return {
        "passed": True,
        "reason": None,
        "face_count": face_count,
        "scene": "public",
        "sensitive_text": False,
    }
