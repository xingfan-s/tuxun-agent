"""
OCR-based Chinese geolocation clue extraction.

Uses PaddleOCR (already installed) to extract text from image,
then runs regex parsers for license plates, phone area codes,
and highway numbers.

Lazy singleton for PaddleOCR to avoid reloading the model.
"""

import os
os.environ.setdefault("MKL_DEBUG_CPU_TYPE", "5")

import re
import base64
from app.utils.logging import structlog
from typing import Any

from langchain.tools import tool

from app.tools.china_knowledge import (
    PLATE_TO_CITY,
    PLATE_PROVINCE_CHARS,
    AREA_CODE_TO_CITY,
    NATIONAL_HIGHWAYS,
    PROVINCE_HIGHWAY_PREFIX,
)

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Lazy PaddleOCR singleton
# ---------------------------------------------------------------------------

_ocr_instance = None


def _get_ocr():
    global _ocr_instance
    if _ocr_instance is None:
        from paddleocr import PaddleOCR
        _ocr_instance = PaddleOCR(lang="ch", use_angle_cls=True, show_log=False)
    return _ocr_instance


# ---------------------------------------------------------------------------
# Image decode
# ---------------------------------------------------------------------------

def _decode_image(image_base64: str):
    """Decode base64 image to numpy array for OpenCV."""
    import cv2
    import numpy as np

    img_bytes = base64.b64decode(image_base64)
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------
# OCR extraction
# ---------------------------------------------------------------------------

def extract_all_text(image_base64: str) -> list[dict]:
    """Run PaddleOCR on image and return recognized text lines with confidence.

    Returns:
        [{"text": str, "confidence": float, "bbox": [[x1,y1],...]}, ...]
        confidence is PaddleOCR's per-line score (0.0-1.0).
    """
    try:
        ocr = _get_ocr()
        img = _decode_image(image_base64)
        if img is None:
            return []
        results = ocr.ocr(img, cls=False)
        if not results or not results[0]:
            return []
        rich = []
        for line in results[0]:
            if not line:
                continue
            bbox, (text, conf) = line[0], line[1]
            rich.append({"text": text, "confidence": float(conf), "bbox": bbox})
        return rich
    except ImportError:
        logger.warning("ocr_import_error", detail="PaddleOCR not installed")
        return []
    except Exception as e:
        logger.warning("ocr_extract_error", error=str(e))
        return []


# ---------------------------------------------------------------------------
# License plate parser
# ---------------------------------------------------------------------------

# Chinese license plate pattern: province_abbr + letter + optional separator + 5-6 alphanumeric
_PLATE_PATTERN = re.compile(
    r"([京津沪渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新])"
    r"([A-HJ-NP-Z])"
    r"[·\s\-]?"
    r"[\dA-HJ-NP-Z]{5,6}"
)

# Also match partial plates (just province + letter, common with occluded plates)
_PARTIAL_PLATE_PATTERN = re.compile(
    r"([京津沪渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新])"
    r"([A-HJ-NP-Z])"
)

# Min OCR confidence thresholds for license plate text.
# Full plates (province + letter + 5-6 chars = 8+ chars): hard to hallucinate.
# Partial plates (just province + letter = 2 chars): VERY easy to hallucinate
# in blurred/masked areas — require significantly higher confidence.
_MIN_PLATE_CONF_FULL = 0.65
_MIN_PLATE_CONF_PARTIAL = 0.85


def _get_match_confidence(matched_text: str, conf_map: dict[str, float]) -> float:
    """Find the OCR confidence for a regex match.

    The regex match may span multiple OCR-detected text lines,
    so we check which known texts the match contains and take
    the minimum — if any part is low-confidence, the whole detection is suspect.
    """
    confs = []
    for text, conf in conf_map.items():
        if text in matched_text:
            confs.append(conf)
    if not confs:
        return 0.0
    return min(confs)


def parse_license_plates(ocr_results: list[dict]) -> list[dict[str, Any]]:
    """Find Chinese license plates in OCR text and look up city.

    Filters low-confidence OCR results to avoid hallucinated plates
    in blurred/masked regions.

    Args:
        ocr_results: List of {"text": str, "confidence": float, "bbox": ...} dicts.

    Returns:
        List of dicts with province, city, plate_code, confidence level,
        and ocr_confidence (the raw PaddleOCR score).
    """
    combined = " ".join(r["text"] for r in ocr_results)
    # Build text→confidence lookup (keep highest if duplicate)
    conf_map: dict[str, float] = {}
    for r in ocr_results:
        t = r["text"]
        if t not in conf_map or r["confidence"] > conf_map[t]:
            conf_map[t] = r["confidence"]

    results = []
    seen = set()

    # Try full plates first
    for m in _PLATE_PATTERN.finditer(combined):
        province_char = m.group(1)
        letter = m.group(2)
        code = province_char + letter
        if code in seen:
            continue
        matched_text = m.group(0)
        ocr_conf = _get_match_confidence(matched_text, conf_map)
        if ocr_conf < _MIN_PLATE_CONF_FULL:
            continue
        seen.add(code)

        info = PLATE_TO_CITY.get(code)
        if info:
            results.append({
                "type": "license_plate",
                "plate_code": code,
                "province": info[0],
                "city": info[1],
                "matched_text": matched_text,
                "confidence": "full_plate",
                "ocr_confidence": round(ocr_conf, 3),
            })
        else:
            province_name = PLATE_PROVINCE_CHARS.get(province_char, province_char)
            results.append({
                "type": "license_plate",
                "plate_code": code,
                "province": province_name,
                "city": "未知",
                "matched_text": matched_text,
                "confidence": "province_only",
                "ocr_confidence": round(ocr_conf, 3),
            })

    # Then partial plates (only if not already found by full match)
    for m in _PARTIAL_PLATE_PATTERN.finditer(combined):
        province_char = m.group(1)
        letter = m.group(2)
        code = province_char + letter
        if code in seen:
            continue
        matched_text = m.group(0)
        ocr_conf = _get_match_confidence(matched_text, conf_map)
        # Partial plates (2 chars) are easily hallucinated — require high confidence
        if ocr_conf < _MIN_PLATE_CONF_PARTIAL:
            continue
        seen.add(code)

        info = PLATE_TO_CITY.get(code)
        if info:
            results.append({
                "type": "license_plate",
                "plate_code": code,
                "province": info[0],
                "city": info[1],
                "matched_text": matched_text,
                "confidence": "partial_plate",
                "ocr_confidence": round(ocr_conf, 3),
            })

    return results


# ---------------------------------------------------------------------------
# Phone area code parser
# ---------------------------------------------------------------------------

# Chinese landline: area_code(3-4 digits)-number(7-8 digits)
_PHONE_PATTERN = re.compile(
    r"(?<!\d)(0\d{2,3})[-\s]?\d{7,8}(?!\d)"
)

# Mobile numbers (not geographic but still useful to extract)
_MOBILE_PATTERN = re.compile(r"1[3-9]\d{9}")


def parse_phone_area_codes(texts: list[str]) -> list[dict[str, Any]]:
    """Find Chinese landline phone numbers and map area codes to cities."""
    combined = " ".join(texts)
    results = []
    seen = set()

    for m in _PHONE_PATTERN.finditer(combined):
        area_code = m.group(1)
        if area_code in seen:
            continue
        seen.add(area_code)

        info = AREA_CODE_TO_CITY.get(area_code)
        if info:
            results.append({
                "type": "phone_area_code",
                "area_code": area_code,
                "province": info[0],
                "city": info[1],
                "matched_text": m.group(0),
            })
        else:
            results.append({
                "type": "phone_area_code",
                "area_code": area_code,
                "province": "未知",
                "city": f"区号{area_code}",
                "matched_text": m.group(0),
            })

    return results


# ---------------------------------------------------------------------------
# Highway number parser
# ---------------------------------------------------------------------------

# National highway: G followed by 1-3 digits (standalone, not part of longer word)
_HIGHWAY_NATIONAL = re.compile(r"(?<![a-zA-Z0-9])(G\d{1,3})(?![a-zA-Z0-9])")

# Province highway: S followed by 1-3 digits
_HIGHWAY_PROVINCE = re.compile(r"(?<![a-zA-Z0-9])(S\d{1,3})(?![a-zA-Z0-9])")


def parse_highway_numbers(texts: list[str]) -> list[dict[str, Any]]:
    """Find Chinese highway numbers and look up route info."""
    combined = " ".join(texts)
    results = []
    seen = set()

    for m in _HIGHWAY_NATIONAL.finditer(combined):
        code = m.group(1)
        if code in seen:
            continue
        seen.add(code)

        info = NATIONAL_HIGHWAYS.get(code)
        if info:
            results.append({
                "type": "highway",
                "highway_code": code,
                "name": info["name"],
                "route": info["route"],
                "provinces": info["provinces"],
                "matched_text": m.group(0),
            })
        else:
            # Try to classify by number
            num = int(code[1:])
            if num < 200:
                category = "首都放射线（北京出发）"
            elif num < 300:
                category = "南北纵线"
            elif num < 400:
                category = "东西横线"
            else:
                category = "国道"
            results.append({
                "type": "highway",
                "highway_code": code,
                "name": "未知",
                "route": category,
                "provinces": [],
                "matched_text": m.group(0),
            })

    # Province highways
    for m in _HIGHWAY_PROVINCE.finditer(combined):
        code = m.group(1)
        if code in seen:
            continue
        seen.add(code)

        prefix = code[:3] if len(code) >= 3 else code[:2]
        province = PROVINCE_HIGHWAY_PREFIX.get(prefix)
        results.append({
            "type": "highway",
            "highway_code": code,
            "name": f"{province}省道" if province else "省道",
            "route": f"{province}境内" if province else "省级道路",
            "provinces": [province] if province else [],
            "matched_text": m.group(0),
        })

    return results


# ---------------------------------------------------------------------------
# Main tool
# ---------------------------------------------------------------------------

@tool
def extract_china_clues(image_base64: str) -> dict:
    """从图片中提取中国境内的地理定位线索。

    使用OCR识别图片中的中文文字，自动解析：
    - 车牌号（如粤B→深圳市）
    - 电话区号（如0571→杭州市）
    - 公路编号（如G318→沪聂线）
    - 所有可见文字

    适用于中国境内的地理定位场景。

    Args:
        image_base64: 图片的base64编码字符串

    Returns:
        {
            "all_text": ["所有识别到的文字行"],
            "text_count": 文字行数,
            "license_plates": [{"plate_code": "粤B", "province": "广东省", "city": "深圳市", ...}],
            "phone_area_codes": [{"area_code": "0571", "province": "浙江省", "city": "杭州市", ...}],
            "highways": [{"highway_code": "G318", "name": "沪聂线", "route": "上海→聂拉木", ...}],
            "geolocation_summary": "一句话摘要，如'车牌粤B提示深圳市，电话区号0571提示杭州市'"
        }
    """
    ocr_results = extract_all_text(image_base64)
    texts = [r["text"] for r in ocr_results]

    if not texts:
        return {
            "all_text": [],
            "text_count": 0,
            "license_plates": [],
            "phone_area_codes": [],
            "highways": [],
            "geolocation_summary": "OCR未识别到文字",
        }

    plates = parse_license_plates(ocr_results)
    phones = parse_phone_area_codes(texts)
    highways = parse_highway_numbers(texts)

    # Build summary
    summary_parts = []
    for p in plates:
        ocr_note = f"(OCR={p.get('ocr_confidence', '?')})" if p.get("ocr_confidence") else ""
        summary_parts.append(f"车牌{p['plate_code']}→{p['province']}{p['city']}{ocr_note}")
    for ph in phones:
        summary_parts.append(f"电话区号{ph['area_code']}→{ph['province']}{ph['city']}")
    for h in highways:
        if h.get("name") != "未知":
            summary_parts.append(f"公路{h['highway_code']}({h['name']})途经{h.get('provinces', [])}")

    summary = "；".join(summary_parts) if summary_parts else f"共识别{len(texts)}行文字，未发现车牌/区号/公路编号"

    return {
        "all_text": texts,
        "text_count": len(texts),
        "license_plates": plates,
        "phone_area_codes": phones,
        "highways": highways,
        "geolocation_summary": summary,
    }
