#!/usr/bin/env python3
"""
Batch reverse-geocode GeoComp entries to populate city/province labels.

Reads metadata.json, groups entries by GPS grid (~1km), reverse geocodes
each unique grid cell via Amap API, and writes city labels back.

Usage:
    cd backend
    python scripts/label_geocomp_cities.py
    python scripts/label_geocomp_cities.py --dry-run
"""

import sys
import json
import asyncio
import structlog
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import get_settings

logger = structlog.get_logger()


async def reverse_geocode_batch(coords: list[tuple[float, float]],
                                concurrency: int = 5) -> dict[tuple[float, float], dict]:
    """Reverse geocode a batch of coordinates via Amap API.

    Returns: {(lat, lng): {"province": str, "city": str, "district": str}}
    """
    import httpx
    settings = get_settings()
    if not settings.amap_api_key:
        logger.error("no_amap_api_key")
        return {}

    sem = asyncio.Semaphore(concurrency)
    results: dict = {}
    failed = 0

    async def _reverse_one(lat: float, lng: float):
        nonlocal failed
        async with sem:
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.get(
                        "https://restapi.amap.com/v3/geocode/regeo",
                        params={
                            "key": settings.amap_api_key,
                            "location": f"{lng},{lat}",
                            "extensions": "base",
                        },
                    )
                    data = resp.json()
                    if data.get("status") == "1" and data.get("regeocode"):
                        comp = data["regeocode"].get("addressComponent", {})
                        # Amap returns empty city for province-level cities
                        city = comp.get("city", "") or comp.get("province", "") or ""
                        results[(lat, lng)] = {
                            "province": comp.get("province", "") or "",
                            "city": city,
                            "district": comp.get("district", "") or "",
                        }
                    else:
                        failed += 1
            except Exception as e:
                failed += 1
                logger.debug("regeo_failed", lat=lat, lng=lng, error=str(e)[:80])
            await asyncio.sleep(0.15)  # Rate limit: ~7 QPS

    # Process in batches with progress
    total = len(coords)
    for i in range(0, total, 50):
        batch = coords[i:i+50]
        tasks = [_reverse_one(lat, lng) for lat, lng in batch]
        await asyncio.gather(*tasks)
        if (i + 50) % 500 == 0 or i + 50 >= total:
            print(f"  Reverse geocoding: {min(i+50, total)}/{total} "
                  f"({len(results)} ok, {failed} failed)")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Label GeoComp entries with city names")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--db-path", default="./data/geo_image_db")
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()

    db_path = Path(args.db_path)
    meta_path = db_path / "metadata.json"
    if not meta_path.exists():
        print(f"Error: {meta_path} not found")
        return

    with open(meta_path, "r") as f:
        data = json.load(f)

    entries = data.get("entries", {})
    empty_label_entries = {
        k: v for k, v in entries.items()
        if not v.get("city") and "geocomp" in v.get("source", "")
    }

    print(f"Total entries: {len(entries)}")
    print(f"GeoComp entries without city: {len(empty_label_entries)}")

    if not empty_label_entries:
        print("All entries have city labels. Nothing to do.")
        return

    # Group by GPS grid (0.03° ≈ 3km) to reduce API calls
    grids: dict[tuple[float, float], list[str]] = {}
    for uid, meta in empty_label_entries.items():
        key = (round(meta["lat"], 2), round(meta["lon"], 2))
        grids.setdefault(key, []).append(uid)

    unique_coords = [(lat, lng) for lat, lng in grids.keys()]
    print(f"Unique GPS grids: {len(unique_coords)}")
    print(f"Estimated API calls: {len(unique_coords)} (~{len(unique_coords)*0.15:.0f}s at 7 QPS)")

    if args.dry_run:
        print("[DRY RUN] No API calls made.")
        return

    # Batch reverse geocode
    results = asyncio.run(reverse_geocode_batch(
        unique_coords, concurrency=args.concurrency))

    # Update metadata
    updated = 0
    for (lat, lng), geo in results.items():
        grid_key = (round(lat, 2), round(lng, 2))
        for uid in grids.get(grid_key, []):
            if uid in entries:
                entries[uid]["city"] = geo["city"]
                entries[uid]["province"] = geo.get("province", "")
                # Add district/city as tags for better search
                if geo.get("city"):
                    tags = entries[uid].get("tags", [])
                    if geo["city"] not in tags:
                        tags.insert(0, geo["city"])
                    entries[uid]["tags"] = tags
                updated += 1

    data["entries"] = entries
    with open(meta_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nUpdated {updated} entries with city labels")
    print(f"Success rate: {len(results)}/{len(unique_coords)} grids")

    # Stats
    cities = {}
    for v in entries.values():
        c = v.get("city", "unknown")
        if c:
            cities[c] = cities.get(c, 0) + 1
    print(f"Top cities now:")
    for city, count in sorted(cities.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"  {city}: {count}")


if __name__ == "__main__":
    main()
