from app.geolocation.coordinates import gcj02_to_wgs84, wgs84_to_gcj02
from app.geolocation.result_validation import normalize_result


def test_coordinate_round_trip_in_china():
    lat, lng = 39.9042, 116.4074
    gcj_lat, gcj_lng = wgs84_to_gcj02(lat, lng)
    restored_lat, restored_lng = gcj02_to_wgs84(gcj_lat, gcj_lng)
    assert abs(restored_lat - lat) < 1e-5
    assert abs(restored_lng - lng) < 1e-5


def test_coordinate_outside_china_is_unchanged():
    assert wgs84_to_gcj02(40.7, -74.0) == (40.7, -74.0)


def test_unknown_coordinate_is_null_with_explicit_semantics():
    result = normalize_result({
        "country": "中国",
        "province": "浙江省",
        "city": "杭州市",
        "lat": 0,
        "lng": 0,
    })
    assert result["lat"] is None
    assert result["lng"] is None
    assert result["coord_system"] == "unknown"
    assert result["precision_level"] == "city"
    assert result["uncertainty_radius_m"] == 25_000
