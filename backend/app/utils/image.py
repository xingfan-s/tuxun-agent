import base64
import io
from app.utils.logging import structlog
import uuid
import time
from pathlib import Path
from PIL import Image
from app.config import get_settings

logger = structlog.get_logger()

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

# Magic byte signatures for allowed image formats
_MAGIC_SIGNATURES = {
    b'\xff\xd8\xff': "image/jpeg",       # JPEG
    b'\x89PNG\r\n\x1a\n': "image/png",   # PNG
    b'RIFF': "image/webp",               # WebP (RIFF....WEBP)
}

_WEBP_FULL_MAGIC = b'WEBP'


def validate_image(content_type: str, file_size: int, file_bytes: bytes = b"") -> tuple[bool, str]:
    """Validate uploaded image. Returns (valid, error_message).

    Checks both the client-supplied content type AND the actual file magic bytes
    to prevent content-type spoofing.
    """
    settings = get_settings()
    if content_type not in ALLOWED_TYPES:
        return False, f"不支持的文件格式：{content_type}，仅支持 JPG/PNG/WebP"
    if file_size > settings.max_file_size_mb * 1024 * 1024:
        return False, f"文件大小超过限制（{settings.max_file_size_mb}MB）"
    if file_bytes:
        actual_type = detect_image_type(file_bytes)
        if actual_type is None:
            return False, "文件内容不是有效的图片格式（JPG/PNG/WebP）"
        if actual_type != content_type:
            return False, "文件类型与实际内容不一致"
        try:
            Image.MAX_IMAGE_PIXELS = settings.max_image_pixels
            with Image.open(io.BytesIO(file_bytes)) as image:
                image.verify()
                width, height = image.size
                if width * height > settings.max_image_pixels:
                    return False, "图片分辨率超过上限"
        except Exception:
            return False, "图片解码失败"
    return True, ""


def detect_image_type(file_bytes: bytes) -> str | None:
    """Detect image type from magic bytes. Returns MIME type or None."""
    if len(file_bytes) < 12:
        return None
    for magic, mime in _MAGIC_SIGNATURES.items():
        if file_bytes.startswith(magic):
            if mime == "image/webp":
                # WebP: check for WEBP at offset 8
                if file_bytes[8:12] == _WEBP_FULL_MAGIC:
                    return mime
                return None
            return mime
    return None


def save_upload(file_bytes: bytes, filename: str, content_type: str | None = None) -> tuple[str, str]:
    """Save uploaded file with UUID name. Returns (file_path, uuid_filename)."""
    settings = get_settings()
    ext = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}.get(
        content_type or "", ".jpg"
    )
    uuid_name = f"{uuid.uuid4().hex}{ext}"
    upload_path = Path(settings.upload_dir) / uuid_name
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    upload_path.write_bytes(file_bytes)
    return str(upload_path), uuid_name


def compress_for_vision(image_path: str, max_size: int = 1024) -> str:
    """Compress image and return base64 for Qwen-VL."""
    Image.MAX_IMAGE_PIXELS = get_settings().max_image_pixels
    img = Image.open(image_path)
    img = img.convert("RGB")

    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def encode_image_for_ocr(image_path: str, max_size: int = 4096) -> str:
    """Encode image to base64 at higher resolution for OCR (PaddleOCR needs more detail than VL)."""
    Image.MAX_IMAGE_PIXELS = get_settings().max_image_pixels
    img = Image.open(image_path)
    img = img.convert("RGB")

    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def delete_image(image_path: str):
    """Delete uploaded image file."""
    try:
        Path(image_path).unlink(missing_ok=True)
    except Exception as e:
        import structlog
        structlog.get_logger().warning("delete_image_error", path=image_path, error=str(e))


def cleanup_expired_uploads(directory: str, ttl_seconds: int, active_paths: set[str] | None = None) -> int:
    """Remove stale upload files while preserving files owned by active tasks."""
    root = Path(directory)
    if not root.exists():
        return 0
    active = {str(Path(path).resolve()) for path in (active_paths or set())}
    now = time.time()
    removed = 0
    for path in root.iterdir():
        if not path.is_file() or str(path.resolve()) in active:
            continue
        try:
            if now - path.stat().st_mtime > max(1, ttl_seconds):
                path.unlink()
                removed += 1
        except OSError as exc:
            logger.warning("upload_cleanup_failed", error_type=type(exc).__name__)
    return removed
