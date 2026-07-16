import re
import base64
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

SENSITIVE_PATTERNS = [
    (re.compile(r'1[3-9]\d{9}'), 'phone_number'),
    (re.compile(r'\d{6}(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[\dXx]'), 'id_card'),
    (re.compile(r'\d{16,19}'), 'bank_card'),
    (re.compile(r'(顺丰|中通|圆通|韵达|申通|EMS|京东)\S{8,}'), 'express'),
]


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
    Falls back to returning no sensitive text if OCR fails.
    """
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(lang='ch', use_angle_cls=False, show_log=False)
        img = decode_base64_image(image_base64)
        if img is None:
            return False, []
        results = ocr.ocr(img, cls=False)
        if not results or not results[0]:
            return False, []
        texts = [line[1][0] for line in results[0] if line]
        combined = ' '.join(texts)
        return check_text(combined)
    except ImportError:
        return False, []
    except Exception:
        return False, []
