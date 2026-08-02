#!/usr/bin/env python3
"""
Seed the CLIP+FAISS geo-image database with Chinese city reference images.

Uses Amap API (already configured) to:
  1. Search for landmark POIs in each city
  2. Download POI photos (real user-uploaded photos of landmarks)
  3. Fall back to Amap static maps if no photos available

Usage:
    cd backend
    python scripts/seed_geo_db.py              # all cities
    python scripts/seed_geo_db.py --cities 10  # first 10 cities only
    python scripts/seed_geo_db.py --limit 20   # max 20 images per city
"""

import sys
import time
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.cities import CITIES
from app.config import get_settings

import structlog
logger = structlog.get_logger()


# ============================================================
# Amap API helpers
# ============================================================

async def amap_text_search(city: str, keyword: str, offset: int = 20) -> list[dict]:
    """Search Amap POI text search API. Returns list of POI dicts."""
    import httpx
    settings = get_settings()
    if not settings.amap_api_key:
        return []
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                "https://restapi.amap.com/v3/place/text",
                params={
                    "key": settings.amap_api_key,
                    "keywords": keyword,
                    "city": city,
                    "offset": offset,
                },
            )
            data = resp.json()
            if data.get("status") != "1" or not data.get("pois"):
                return []
            return [
                {
                    "id": p.get("id", ""),
                    "name": p.get("name", ""),
                    "lat": float(p["location"].split(",")[1]),
                    "lng": float(p["location"].split(",")[0]),
                    "address": p.get("address", ""),
                    "type": p.get("type", ""),
                }
                for p in data["pois"]
            ]
    except Exception as e:
        logger.debug("amap_search_failed", city=city, keyword=keyword, error=str(e))
    return []


async def amap_poi_photos(poi_id: str) -> list[str]:
    """Get photo URLs for a POI from its detail page."""
    import httpx
    settings = get_settings()
    if not settings.amap_api_key or not poi_id:
        return []
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                "https://restapi.amap.com/v3/place/detail",
                params={"key": settings.amap_api_key, "id": poi_id},
            )
            data = resp.json()
            if data.get("status") != "1":
                return []
            pois = data.get("pois", [])
            if not pois:
                return []
            photos = pois[0].get("photos", [])
            return [p.get("url", "") for p in photos if p.get("url")]
    except Exception as e:
        logger.debug("amap_photos_failed", poi_id=poi_id, error=str(e))
    return []


async def download_image(url: str, save_path: str) -> bool:
    """Download an image from URL to local path."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, headers={"User-Agent": "TuXun-Agent/0.1"})
            if resp.status_code == 200 and len(resp.content) > 1000:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, "wb") as f:
                    f.write(resp.content)
                return True
    except Exception as e:
        logger.debug("download_failed", url=url[:80], error=str(e))
    return False


async def download_static_map(lat: float, lng: float, zoom: int = 15) -> bytes | None:
    """Download Amap static map image at given coordinates."""
    import httpx
    settings = get_settings()
    if not settings.amap_api_key:
        return None
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                "https://restapi.amap.com/v3/staticmap",
                params={
                    "key": settings.amap_api_key,
                    "location": f"{lng},{lat}",
                    "zoom": zoom,
                    "size": "600*400",
                    "scale": 1,
                },
            )
            if resp.status_code == 200 and len(resp.content) > 1000:
                return resp.content
    except Exception as e:
        logger.debug("staticmap_failed", lat=lat, lng=lng, error=str(e))
    return None


# ============================================================
# Main seeding logic
# ============================================================

async def seed_city(db, city_name: str, province: str,
                     center_lat: float, center_lng: float,
                     tags: list[str], images_dir: Path,
                     max_per_city: int = 20) -> int:
    """Seed images for one city. Returns number of images added."""
    added = 0

    # --- Strategy: search POIs by multiple keywords, get their photos ---
    keywords = ["景点", "公园", "广场", "商圈", "地标"]
    seen_poi_ids: set[str] = set()

    for keyword in keywords:
        if added >= max_per_city:
            break
        pois = await amap_text_search(city_name, keyword, offset=15)
        for poi in pois:
            if added >= max_per_city:
                break
            if poi["id"] in seen_poi_ids:
                continue
            seen_poi_ids.add(poi["id"])

            photos = await amap_poi_photos(poi["id"])
            for photo_url in photos:
                if added >= max_per_city:
                    break
                safe_name = f"{city_name}_{added:03d}"
                img_path = images_dir / f"{safe_name}.jpg"
                ok = await download_image(photo_url, str(img_path))
                if ok:
                    img_id = db.add(
                        str(img_path), poi["lat"], poi["lng"],
                        city=city_name, tags=tags + [poi["name"], keyword],
                        source="amap_poi",
                    )
                    if img_id >= 0:
                        added += 1
                await asyncio.sleep(0.05)

    # --- Fallback: static map tiles at key city coordinates ---
    if added < 3:
        offsets = [(0, 0), (0.01, 0), (-0.01, 0), (0, 0.01), (0, -0.01)]
        for dl, dn in offsets:
            if added >= max_per_city:
                break
            img_bytes = await download_static_map(center_lat + dn, center_lng + dl)
            if img_bytes:
                img_path = images_dir / f"{city_name}_map_{added:03d}.png"
                img_path.write_bytes(img_bytes)
                img_id = db.add(
                    str(img_path), center_lat + dn, center_lng + dl,
                    city=city_name, tags=tags + ["static_map"],
                    source="amap_static_map",
                )
                if img_id >= 0:
                    added += 1
                await asyncio.sleep(0.05)

    return added


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Seed CLIP geo-image database")
    parser.add_argument("--cities", type=int, default=0,
                        help="Only process first N cities (0=all)")
    parser.add_argument("--limit", type=int, default=20,
                        help="Max images per city (default: 20)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without downloading")
    args = parser.parse_args()

    settings = get_settings()
    print(f"=== CLIP Geo-Image Database Seeder ===")
    print(f"Amap API:      {'YES' if settings.amap_api_key else 'NO'}")
    print(f"Tencent SV:    {'YES' if settings.tencent_map_key else 'NO'} (not used)")
    print(f"DB path:       {settings.clip_db_path}")
    print(f"Cities:        {args.cities if args.cities > 0 else len(CITIES)}")
    print(f"Max/city:      {args.limit}")
    print()

    if args.dry_run:
        print("[DRY RUN] No images downloaded.")
        return

    # Setup
    from app.tools.clip_search import get_db
    db = get_db(settings.clip_db_path)
    images_dir = Path(settings.clip_db_path) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    start_count = db.count()
    cities_to_process = CITIES[:args.cities] if args.cities > 0 else CITIES
    total_added = 0
    t_start = time.time()

    for i, (city, province, lat, lng, tags) in enumerate(cities_to_process):
        t_city = time.time()
        added = await seed_city(db, city, province, lat, lng, tags,
                               images_dir, max_per_city=args.limit)
        total_added += added
        elapsed = int((time.time() - t_city) * 1000)

        pct = (i + 1) / len(cities_to_process) * 100
        bar = "=" * int(pct / 4) + "-" * (25 - int(pct / 4))
        print(f"[{bar}] {i+1:3d}/{len(cities_to_process)} {city:6s} +{added:2d} "
              f"({elapsed}ms)  total:{total_added}")

        if (i + 1) % 10 == 0:
            db.save()
            print(f"  -> checkpoint ({db.count()} in DB)")

    db.save()
    total_elapsed = int(time.time() - t_start)

    print(f"\n=== Done ({total_elapsed}s) ===")
    print(f"Added this run: {total_added}")
    print(f"Total in DB:    {db.count()} (was {start_count})")

    stats = db.stats()
    print(f"\nTop cities:")
    for city, count in stats.get("top_cities", [])[:15]:
        print(f"  {city}: {count}")


if __name__ == "__main__":
    asyncio.run(main())
