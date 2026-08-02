import re
import base64
import threading
from typing import TYPE_CHECKING
from app.config import get_settings

if TYPE_CHECKING:
    import numpy as np

from app.utils.logging import structlog

logger = structlog.get_logger(__name__)

SENSITIVE_PATTERNS = [
    (re.compile(r'1[3-9]\d{9}'), 'phone_number'),
    (re.compile(r'\d{6}(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[\dXx]'), 'id_card'),
    (re.compile(r'\d{16,19}'), 'bank_card'),
    (re.compile(r'(顺丰|中通|圆通|韵达|申通|EMS|京东)\S{8,}'), 'express'),
]

_ocr_instance = None
_ocr_status = "lazy_load_pending"
_ocr_lock = threading.Lock()


class OcrCapabilityError(RuntimeError):
    """The OCR safety capability is required but cannot run."""


def _get_ocr():
    """Lazy singleton for PaddleOCR (lightweight, no angle classification)."""
    global _ocr_instance, _ocr_status
    if _ocr_instance is not None:
        return _ocr_instance

    with _ocr_lock:
        if _ocr_instance is not None:
            return _ocr_instance
        try:
            import os
            os.environ.setdefault("MKL_DEBUG_CPU_TYPE", "5")

            # On Windows, PaddlePaddle and Torch ship overlapping native
            # runtimes. Loading Torch first avoids PaddleOCR's indirect Torch
            # import resolving incompatible DLL symbols.
            import torch  # noqa: F401
            from paddleocr import PaddleOCR

            _ocr_instance = PaddleOCR(lang="ch", use_angle_cls=False, show_log=False)
            _ocr_status = "loaded"
            logger.info("paddleocr_safety_loaded")
        except Exception as exc:
            _ocr_status = f"unavailable:{type(exc).__name__}"
            raise
    return _ocr_instance


def get_ocr_capability_status() -> str:
    return _ocr_status


def probe_ocr_capability() -> str:
    """Run one cached, minimal inference for strict readiness checks."""
    global _ocr_status
    try:
        import numpy as np

        ocr = _get_ocr()
        ocr.ocr(np.zeros((32, 32, 3), dtype=np.uint8), cls=False)
        _ocr_status = "ok"
        return _ocr_status
    except Exception as exc:
        _ocr_status = f"unavailable:{type(exc).__name__}"
        raise OcrCapabilityError("OCR safety service is unavailable") from exc


def check_text(text: str) -> tuple[bool, list[str]]:
    """
    Returns (has_sensitive: bool, found_types: list[str]).
    """
    found = []
    for pattern, ptype in SENSITIVE_PATTERNS:
        if pattern.search(text):
            found.append(ptype)
    return len(found) > 0, found


def decode_base64_image(b64_str: str):
    """Decode base64 image string to OpenCV image array."""
    import cv2
    import numpy as np

    img_bytes = base64.b64decode(b64_str)
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def check_image_text(image_base64: str) -> tuple[bool, list[str]]:
    """
    OCR the image and check for sensitive text.
    OCR failures fail closed only when OCR is explicitly required. Otherwise
    the safety pipeline continues with an operator-visible capability warning.
    """
    global _ocr_status
    strict = get_settings().safety_require_ocr
    try:
        ocr = _get_ocr()
        img = decode_base64_image(image_base64)
        if img is None:
            return False, []
        results = ocr.ocr(img, cls=False)
        if not results or not results[0]:
            return False, []
        texts = [line[1][0] for line in results[0] if line]
        combined = ' '.join(texts)
        _ocr_status = "ok"
        return check_text(combined)
    except Exception as exc:
        reason = "ocr_unavailable" if isinstance(exc, ImportError) else "ocr_check_failed"
        _ocr_status = f"unavailable:{type(exc).__name__}"
        if strict:
            logger.error("ocr_safety_check_failed", error=str(exc))
            raise OcrCapabilityError("OCR safety service is unavailable") from exc
        logger.warning("ocr_safety_check_skipped", error=str(exc))
        return False, [reason]
