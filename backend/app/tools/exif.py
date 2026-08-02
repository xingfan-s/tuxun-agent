from app.utils.logging import structlog
from pathlib import Path
from langchain.tools import tool
import exifread

logger = structlog.get_logger()


@tool
def extract_exif(image_path: str) -> dict:
    """提取图片 EXIF 元数据，包括 GPS、拍摄时间、设备型号。

    Args:
        image_path: 图片文件路径

    Returns:
        {
            "gps": {"lat": float, "lng": float} | None,
            "datetime": str | None,
            "device": str | None,
            "has_gps": bool
        }
    """
    path = Path(image_path)
    if not path.exists():
        return {"gps": None, "datetime": None, "device": None, "has_gps": False}

    try:
        with open(path, "rb") as f:
            tags = exifread.process_file(f, details=False)

        gps = _parse_gps(tags)
        dt = _parse_datetime(tags)
        device = str(tags.get("Image Model", "")) if tags.get("Image Model") else None

        return {
            "gps": gps,
            "datetime": dt,
            "device": device,
            "has_gps": gps is not None,
        }
    except Exception as e:
        logger.warning("exif_read_error", error=str(e))
        return {"gps": None, "datetime": None, "device": None, "has_gps": False}


def _parse_gps(tags: dict) -> dict | None:
    try:
        lat_tag = tags.get("GPS GPSLatitude")
        lat_ref = tags.get("GPS GPSLatitudeRef")
        lng_tag = tags.get("GPS GPSLongitude")
        lng_ref = tags.get("GPS GPSLongitudeRef")
        if not all([lat_tag, lat_ref, lng_tag, lng_ref]):
            return None

        lat = _dms_to_decimal(lat_tag.values, str(lat_ref))
        lng = _dms_to_decimal(lng_tag.values, str(lng_ref))
        return {"lat": round(lat, 6), "lng": round(lng, 6)}
    except Exception:
        return None


def _dms_to_decimal(values, ref: str) -> float:
    degrees = float(values[0])
    minutes = float(values[1])
    seconds = float(values[2])
    decimal = degrees + minutes / 60 + seconds / 3600
    if ref in ("S", "W"):
        decimal = -decimal
    return decimal


def _parse_datetime(tags: dict) -> str | None:
    dt = tags.get("Image DateTime")
    return str(dt) if dt else None
