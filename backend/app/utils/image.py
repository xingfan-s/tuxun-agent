import base64
import io
import uuid
from pathlib import Path
from PIL import Image
from app.config import get_settings

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


def validate_image(content_type: str, file_size: int) -> tuple[bool, str]:
    """Validate uploaded image. Returns (valid, error_message)."""
    settings = get_settings()
    if content_type not in ALLOWED_TYPES:
        return False, f"不支持的文件格式：{content_type}，仅支持 JPG/PNG/WebP"
    if file_size > settings.max_file_size_mb * 1024 * 1024:
        return False, f"文件大小超过限制（{settings.max_file_size_mb}MB）"
    return True, ""


def save_upload(file_bytes: bytes, filename: str) -> tuple[str, str]:
    """Save uploaded file with UUID name. Returns (file_path, uuid_filename)."""
    settings = get_settings()
    ext = Path(filename).suffix.lower() or ".jpg"
    uuid_name = f"{uuid.uuid4().hex}{ext}"
    upload_path = Path(settings.upload_dir) / uuid_name
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    upload_path.write_bytes(file_bytes)
    return str(upload_path), uuid_name


def compress_for_vision(image_path: str, max_size: int = 2048) -> str:
    """Compress image and return base64 for Qwen-VL."""
    img = Image.open(image_path)
    img = img.convert("RGB")

    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def delete_image(image_path: str):
    """Delete uploaded image file."""
    try:
        Path(image_path).unlink(missing_ok=True)
    except Exception as e:
        import structlog
        structlog.get_logger().warning("delete_image_error", path=image_path, error=str(e))
