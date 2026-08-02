import numpy as np
import cv2
from types import SimpleNamespace

import pytest


def test_face_detect_no_faces():
    from app.safety.face_detect import count_faces
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    result = count_faces(img)
    assert result == 0


def test_face_detect_cascade_exists():
    from app.safety.face_detect import _get_cascade

    face_cascade = _get_cascade()
    assert not face_cascade.empty()


def test_optional_ocr_failure_returns_capability_warning(monkeypatch):
    import app.safety.text_check as text_check

    def fail_to_load_ocr():
        raise RuntimeError("OCR runtime unavailable")

    monkeypatch.setattr(text_check, "_get_ocr", fail_to_load_ocr)
    monkeypatch.setattr(
        text_check,
        "get_settings",
        lambda: SimpleNamespace(safety_require_ocr=False),
    )
    assert text_check.check_image_text("aGVsbG8=") == (False, ["ocr_check_failed"])


def test_required_ocr_failure_is_technical_error(monkeypatch):
    import app.safety.text_check as text_check

    monkeypatch.setattr(
        text_check,
        "_get_ocr",
        lambda: (_ for _ in ()).throw(RuntimeError("OCR runtime unavailable")),
    )
    monkeypatch.setattr(
        text_check,
        "get_settings",
        lambda: SimpleNamespace(safety_require_ocr=True),
    )

    with pytest.raises(text_check.OcrCapabilityError):
        text_check.check_image_text("aGVsbG8=")
