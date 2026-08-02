from app.config import get_settings
from app.safety.face_detect import has_too_many_faces, count_faces


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
    Layers are ordered by cost: face detection first (fast, local), then
    scene classification (API call), then OCR (expensive model load) only
    if both earlier layers pass.
    """
    settings = get_settings()
    warnings: list[str] = []
    # Import heavyweight/API-backed checks only when the full safety pipeline
    # is requested. Local face-detection tests and health checks stay usable
    # without optional logging/OCR packages installed.
    from app.safety.scene_check import check_scene
    from app.safety.text_check import check_image_text, decode_base64_image

    # Layer 1: Face detection (local, fast, no API call)
    img = decode_base64_image(image_base64)
    face_count = count_faces(img) if img is not None else 0
    face_warning = has_too_many_faces(img, settings.safety_face_max_count)
    if face_warning:
        warnings.append("face_signal")
    if face_warning and settings.safety_face_policy == "reject":
        return {
            "passed": False,
            "reason": f"检测到{face_count}个人脸，可能包含人物面部特写",
            "face_count": face_count,
            "scene": "unknown",
            "sensitive_text": False,
            "warnings": warnings,
        }

    # Layer 2: Scene classification (Qwen-VL API call)
    scene_passed, scene_reason = check_scene(image_base64)
    if not scene_passed:
        return {
            "passed": False,
            "reason": scene_reason,
            "face_count": face_count,
            "scene": scene_reason,
            "sensitive_text": False,
            "warnings": warnings,
        }

    # Layer 3: Sensitive text (OCR — expensive, only run if earlier layers pass)
    has_sensitive, sensitive_types = check_image_text(image_base64)
    if has_sensitive:
        return {
            "passed": False,
            "reason": f"检测到敏感信息: {', '.join(sensitive_types)}",
            "face_count": face_count,
            "scene": "public",
            "sensitive_text": True,
            "warnings": warnings,
        }

    warnings.extend(kind for kind in sensitive_types if kind in {"ocr_unavailable", "ocr_check_failed"})

    return {
        "passed": True,
        "reason": None,
        "face_count": face_count,
        "scene": "public",
        "sensitive_text": False,
        "warnings": warnings,
    }
