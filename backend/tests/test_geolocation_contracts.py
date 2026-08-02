import pytest

from app.geolocation.coordinates import (
    gcj02_to_wgs84,
    wgs84_to_gcj02,
)
from app.geolocation.result_validation import normalize_result
from app.tools.map import GeoResult


@pytest.mark.parametrize("lat,lng", [(39.9042, 116.4074), (31.2304, 121.4737), (23.1291, 113.2644)])
def test_gcj02_round_trip_for_major_cities(lat, lng):
    gcj_lat, gcj_lng = wgs84_to_gcj02(lat, lng)
    restored_lat, restored_lng = gcj02_to_wgs84(gcj_lat, gcj_lng)
    assert abs(restored_lat - lat) < 1e-5
    assert abs(restored_lng - lng) < 1e-5


def test_coordinate_conversion_rejects_invalid_input():
    with pytest.raises(ValueError):
        wgs84_to_gcj02(91, 116)


def test_unverified_road_precision_is_downgraded():
    result = normalize_result({
        "city": "杭州市",
        "lat": 30.2741,
        "lng": 120.1551,
        "precision_level": "road",
    })
    assert result["precision_level"] == "city"
    assert result["confidence_kind"] == "ranking_score"


def test_map_result_rejects_invalid_coordinate():
    with pytest.raises(ValueError):
        GeoResult(lat=91, lng=120, display_name="invalid")
