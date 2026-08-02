"""Coordinate helpers used at map-provider boundaries.

The application stores and exposes WGS84.  Amap uses GCJ-02, so conversion is
performed explicitly in its adapter instead of silently mixing coordinates.
"""

from __future__ import annotations

import math


WGS84 = "WGS84"
GCJ02 = "GCJ-02"
BD09 = "BD-09"


def validate_coordinate(lat: float | None, lng: float | None) -> bool:
    if lat is None or lng is None:
        return False
    return math.isfinite(lat) and math.isfinite(lng) and -90 <= lat <= 90 and -180 <= lng <= 180


def out_of_china(lat: float, lng: float) -> bool:
    return not (73.0 <= lng <= 135.0 and 3.0 <= lat <= 54.0)


def _require_coordinate(lat: float, lng: float) -> None:
    if not validate_coordinate(lat, lng):
        raise ValueError("coordinate is outside the valid WGS84 range")


def _transform_lat(x: float, y: float) -> float:
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(y * math.pi) + 40.0 * math.sin(y / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(y / 12.0 * math.pi) + 320 * math.sin(y * math.pi / 30.0)) * 2.0 / 3.0
    return ret


def _transform_lng(x: float, y: float) -> float:
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(x * math.pi) + 40.0 * math.sin(x / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(x / 12.0 * math.pi) + 300.0 * math.sin(x / 30.0 * math.pi)) * 2.0 / 3.0
    return ret


def wgs84_to_gcj02(lat: float, lng: float) -> tuple[float, float]:
    _require_coordinate(lat, lng)
    if out_of_china(lat, lng):
        return lat, lng
    a = 6378245.0
    ee = 0.00669342162296594323
    d_lat = _transform_lat(lng - 105.0, lat - 35.0)
    d_lng = _transform_lng(lng - 105.0, lat - 35.0)
    rad_lat = lat / 180.0 * math.pi
    magic = 1 - ee * math.sin(rad_lat) ** 2
    sqrt_magic = math.sqrt(magic)
    d_lat = d_lat * 180.0 / ((a * (1 - ee)) / (magic * sqrt_magic) * math.pi)
    d_lng = d_lng * 180.0 / (a / sqrt_magic * math.cos(rad_lat) * math.pi)
    return lat + d_lat, lng + d_lng


def gcj02_to_wgs84(lat: float, lng: float) -> tuple[float, float]:
    _require_coordinate(lat, lng)
    if out_of_china(lat, lng):
        return lat, lng
    guess_lat, guess_lng = lat, lng
    for _ in range(8):
        transformed_lat, transformed_lng = wgs84_to_gcj02(guess_lat, guess_lng)
        guess_lat -= transformed_lat - lat
        guess_lng -= transformed_lng - lng
    return guess_lat, guess_lng


def bd09_to_gcj02(lat: float, lng: float) -> tuple[float, float]:
    """Convert Baidu BD-09 coordinates to GCJ-02."""
    _require_coordinate(lat, lng)
    z = math.sqrt(lng * lng + lat * lat) - 0.00002 * math.sin(lat * math.pi * 3000.0 / 180.0)
    theta = math.atan2(lat, lng) - 0.000003 * math.cos(lng * math.pi * 3000.0 / 180.0)
    return (
        z * math.sin(theta) - 0.006,
        z * math.cos(theta) - 0.0065,
    )


def bd09_to_wgs84(lat: float, lng: float) -> tuple[float, float]:
    _require_coordinate(lat, lng)
    gcj_lat, gcj_lng = bd09_to_gcj02(lat, lng)
    return gcj02_to_wgs84(gcj_lat, gcj_lng)
