from abc import ABC, abstractmethod
from dataclasses import dataclass
import structlog
import httpx
from app.config import get_settings

logger = structlog.get_logger()


@dataclass
class GeoResult:
    lat: float
    lng: float
    display_name: str
    country: str = ""
    province: str = ""
    city: str = ""


@dataclass
class POI:
    name: str
    lat: float
    lng: float
    address: str = ""
    category: str = ""


# ---------- Base ----------

class BaseMapService(ABC):
    @abstractmethod
    async def geocode(self, address: str) -> list[GeoResult]: ...
    @abstractmethod
    async def reverse_geocode(self, lat: float, lng: float) -> GeoResult | None: ...
    @abstractmethod
    async def search_nearby(self, lat: float, lng: float, keyword: str, radius: int = 5000) -> list[POI]: ...
    @abstractmethod
    async def get_streetview(self, lat: float, lng: float) -> bytes | None: ...


# ---------- Amap (高德) ----------

class AmapService(BaseMapService):
    BASE = "https://restapi.amap.com/v3"

    def __init__(self, api_key: str | None = None):
        settings = get_settings()
        self.api_key = api_key or settings.amap_api_key
        self.client = httpx.AsyncClient(timeout=15)

    async def geocode(self, address: str) -> list[GeoResult]:
        resp = await self.client.get(
            f"{self.BASE}/geocode/geo",
            params={"key": self.api_key, "address": address},
        )
        data = resp.json()
        if data.get("status") != "1" or not data.get("geocodes"):
            return []
        result = []
        for g in data["geocodes"]:
            lng, lat = g["location"].split(",")
            result.append(GeoResult(
                lat=float(lat), lng=float(lng),
                display_name=g.get("formatted_address", address),
            ))
        return result

    async def reverse_geocode(self, lat: float, lng: float) -> GeoResult | None:
        resp = await self.client.get(
            f"{self.BASE}/geocode/regeo",
            params={"key": self.api_key, "location": f"{lng},{lat}", "extensions": "base"},
        )
        data = resp.json()
        if data.get("status") != "1" or not data.get("regeocode"):
            return None
        rg = data["regeocode"]
        comp = rg.get("addressComponent", {})
        return GeoResult(
            lat=lat, lng=lng,
            display_name=rg.get("formatted_address", ""),
            country=comp.get("country", ""),
            province=comp.get("province", ""),
            city=comp.get("city", "") or comp.get("province", ""),
        )

    async def search_nearby(self, lat: float, lng: float, keyword: str, radius: int = 5000) -> list[POI]:
        resp = await self.client.get(
            f"{self.BASE}/place/around",
            params={"key": self.api_key, "location": f"{lng},{lat}",
                    "keywords": keyword, "radius": radius, "offset": 10},
        )
        data = resp.json()
        if data.get("status") != "1" or not data.get("pois"):
            return []
        return [
            POI(name=p.get("name", ""), lat=float(p["location"].split(",")[1]),
                lng=float(p["location"].split(",")[0]),
                address=p.get("address", ""), category=p.get("type", ""))
            for p in data["pois"]
        ]

    async def get_streetview(self, lat: float, lng: float) -> bytes | None:
        return None  # Amap static map requires a different approach


# ---------- Nominatim ----------

class NominatimService(BaseMapService):
    BASE = "https://nominatim.openstreetmap.org"

    def __init__(self):
        self.client = httpx.AsyncClient(
            headers={"User-Agent": "TuXun-Agent/0.1"},
            timeout=10,
        )

    async def geocode(self, address: str) -> list[GeoResult]:
        resp = await self.client.get(
            f"{self.BASE}/search", params={"q": address, "format": "json", "limit": 5}
        )
        resp.raise_for_status()
        return [_parse_nominatim_geocode(r) for r in resp.json()]

    async def reverse_geocode(self, lat: float, lng: float) -> GeoResult | None:
        resp = await self.client.get(
            f"{self.BASE}/reverse", params={"lat": lat, "lon": lng, "format": "json"}
        )
        resp.raise_for_status()
        data = resp.json()
        if not data or "error" in data:
            return None
        return _parse_nominatim_reverse(data)

    async def search_nearby(self, lat: float, lng: float, keyword: str, radius: int = 5000) -> list[POI]:
        resp = await self.client.get(
            f"{self.BASE}/search",
            params={"q": keyword, "format": "json", "limit": 10, "bounded": 1,
                    "viewbox": f"{lng - 0.05},{lat - 0.05},{lng + 0.05},{lat + 0.05}"}
        )
        resp.raise_for_status()
        return [_parse_nominatim_poi(r) for r in resp.json()]

    async def get_streetview(self, lat: float, lng: float) -> bytes | None:
        return None


def _parse_nominatim_geocode(r: dict) -> GeoResult:
    return GeoResult(
        lat=float(r["lat"]), lng=float(r["lon"]),
        display_name=r.get("display_name", ""),
    )


def _parse_nominatim_reverse(r: dict) -> GeoResult:
    addr = r.get("address", {})
    return GeoResult(
        lat=float(r["lat"]), lng=float(r["lon"]),
        display_name=r.get("display_name", ""),
        country=addr.get("country", ""),
        province=addr.get("state", ""),
        city=addr.get("city", addr.get("town", addr.get("village", ""))),
    )


def _parse_nominatim_poi(r: dict) -> POI:
    return POI(
        name=r.get("display_name", ""),
        lat=float(r["lat"]), lng=float(r["lon"]),
        category=r.get("type", ""),
    )


# ---------- Map Service Manager ----------

class MapServiceManager:
    def __init__(self, primary: BaseMapService, fallback: BaseMapService | None = None):
        self.primary = primary
        self.fallback = fallback
        self._primary_failures = 0
        self._using_fallback = False

    @property
    def active(self) -> BaseMapService:
        if self._using_fallback and self.fallback:
            return self.fallback
        return self.primary

    async def _handle_failure(self):
        self._primary_failures += 1
        if self._primary_failures >= 3 and self.fallback and not self._using_fallback:
            logger.warning("map_switching_to_fallback", failures=self._primary_failures)
            self._using_fallback = True

    async def geocode(self, address: str) -> list[GeoResult]:
        try:
            return await self.active.geocode(address)
        except Exception:
            await self._handle_failure()
            if self._using_fallback:
                try:
                    return await self.fallback.geocode(address)
                except Exception:
                    pass
            return []

    async def reverse_geocode(self, lat: float, lng: float) -> GeoResult | None:
        try:
            return await self.active.reverse_geocode(lat, lng)
        except Exception:
            await self._handle_failure()
            if self._using_fallback and self.fallback:
                try:
                    return await self.fallback.reverse_geocode(lat, lng)
                except Exception:
                    pass
            return None

    async def search_nearby(self, lat: float, lng: float, keyword: str, radius: int = 5000) -> list[POI]:
        try:
            return await self.active.search_nearby(lat, lng, keyword, radius)
        except Exception:
            await self._handle_failure()
            if self._using_fallback and self.fallback:
                try:
                    return await self.fallback.search_nearby(lat, lng, keyword, radius)
                except Exception:
                    pass
            return []

    async def get_streetview(self, lat: float, lng: float) -> bytes | None:
        try:
            return await self.active.get_streetview(lat, lng)
        except Exception:
            return None


_map_service: MapServiceManager | None = None


def create_map_service() -> MapServiceManager:
    global _map_service
    if _map_service is not None:
        return _map_service

    settings = get_settings()
    service_map = {
        "nominatim": NominatimService,
        "amap": AmapService,
    }
    primary_map = service_map.get(settings.map_service, NominatimService)()

    fallback_map = None
    if settings.map_service_fallback:
        fb_cls = service_map.get(settings.map_service_fallback)
        if fb_cls:
            fallback_map = fb_cls()

    _map_service = MapServiceManager(primary_map, fallback_map)
    return _map_service
