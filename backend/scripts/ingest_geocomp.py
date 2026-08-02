#!/usr/bin/env python3
"""
Ingest GeoComp (tuxun) China street view images into CLIP+FAISS database.

Pipeline:
  1. Download GeoComp tuxun_combined.parquet metadata (~2GB)
  2. Filter for China mainland locations (31 provinces)
  3. Deduplicate by GPS grid (~500m cells) to avoid redundant encoding
  4. Download street view images via Baidu/Gaode Map API using panoids
  5. CLIP-encode and insert into GeoImageDB
  6. Save FAISS index

Requirements:
  pip install pandas pyarrow httpx pillow

Usage:
  cd backend
  python scripts/ingest_geocomp.py --max-images 10000 --grid-size 3
  python scripts/ingest_geocomp.py --dry-run  # preview without downloading
"""

import sys
import time
import asyncio
import hashlib
import structlog
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = structlog.get_logger()

# ============================================================
# Configuration
# ============================================================

# China province names (Chinese) for filtering
CHINA_PROVINCES = {
    "北京", "天津", "上海", "重庆",
    "河北", "山西", "内蒙古", "辽宁", "吉林", "黑龙江",
    "江苏", "浙江", "安徽", "福建", "江西", "山东",
    "河南", "湖北", "湖南", "广东", "广西", "海南",
    "四川", "贵州", "云南", "西藏", "陕西", "甘肃",
    "青海", "宁夏", "新疆",
    # English names that appear in GeoComp
    "Beijing", "Tianjin", "Shanghai", "Chongqing",
    "Guangdong", "Guangxi", "Hainan", "Fujian", "Jiangxi", "Hunan",
    "Hubei", "Henan", "Shandong", "Shanxi", "Shaanxi",
    "Sichuan", "Guizhou", "Yunnan", "Xizang", "Tibet",
    "Jiangsu", "Zhejiang", "Anhui",
    "Hebei", "Liaoning", "Jilin", "Heilongjiang",
    "Gansu", "Qinghai", "Ningxia", "Xinjiang",
    "Nei Mongol", "Inner Mongolia",
}

# China approximate bounding box
CHINA_BBOX = {
    "min_lat": 18.0, "max_lat": 54.0,
    "min_lng": 73.0, "max_lng": 135.0,
}


def is_china_location(lat: float, lng: float, nation: str = "",
                      province: str = "", city: str = "",
                      china_flag: bool = False) -> bool:
    """Check if a location is in mainland China.

    Uses the `china` boolean from GeoComp metadata first (most reliable),
    falls back to bbox + nation/province name matching.
    """
    # GeoComp china flag is the gold standard
    if china_flag:
        return True
    # Bounding box check
    if not (CHINA_BBOX["min_lat"] <= lat <= CHINA_BBOX["max_lat"] and
            CHINA_BBOX["min_lng"] <= lng <= CHINA_BBOX["max_lng"]):
        return False
    # Nation check
    if nation and nation.lower() in ("china", "chinese", "cn", "中国"):
        return True
    # Province check
    if province and any(p in str(province) for p in CHINA_PROVINCES):
        return True
    if city and any(c in str(city) for c in CHINA_PROVINCES):
        return True
    # Macau/Hong Kong exclusion (~22N, 113-114E)
    if 21.5 <= lat <= 23.0 and 113.0 <= lng <= 114.5:
        return False
    # Taiwan exclusion (~21-26N, 120-122E)
    if 21.0 <= lat <= 26.0 and 120.0 <= lng <= 122.0:
        return False
    # Bbox fallback: inside China bbox but not HK/Macau/Taiwan
    return True


# ============================================================
# GeoComp Metadata Extraction
# ============================================================

def load_geocomp_china(parquet_path: str, max_locations: int = 20000,
                       grid_size: int = 3) -> list[dict]:
    """Load GeoComp parquet(s), filter China, deduplicate by GPS grid.

    Args:
        parquet_path: Path to a single .parquet file or a directory of shards
        max_locations: Maximum unique locations to return
        grid_size: Grid precision for dedup (3 = ~500m at equator)

    Returns:
        List of {"lat": float, "lng": float, "pano_id": str, "source": str,
                 "province": str, "city": str, "maps_name": str}
    """
    import pandas as pd
    import json as _json
    from pathlib import Path as _Path

    locations: list[dict] = []
    seen_cells: set[str] = set()

    pp = _Path(parquet_path)
    if pp.is_dir():
        files = sorted(pp.glob("data-*.parquet"))  # Sharded: data-00000.parquet, ...
        if not files:
            files = sorted(pp.glob("*.parquet"))
    elif pp.is_file():
        files = [pp]
    else:
        logger.error("parquet_not_found", path=parquet_path)
        return locations

    logger.info("loading_geocomp", files=len(files))

    # Strategy: process shards until we hit max_locations
    total_rows = 0
    for fp in files:
        if len(locations) >= max_locations:
            break

        try:
            with open(fp, "rb") as fh:
                # Quick check: skip files that are too small (< 10KB = empty)
                fh.seek(0, 2)
                if fh.tell() < 10000:
                    continue
            df = pd.read_parquet(fp)
        except Exception as e:
            logger.debug("parquet_read_error", file=str(fp), error=str(e)[:80])
            continue

        total_rows += len(df)

        for _, row in df.iterrows():
            if len(locations) >= max_locations:
                break

            try:
                data = _json.loads(row["data"]) if isinstance(row["data"], str) else row["data"]
            except (_json.JSONDecodeError, TypeError):
                continue

            # ---- China check: use GeoComp's built-in flag ----
            china_flag = data.get("china", False)
            if isinstance(china_flag, str):
                china_flag = china_flag.lower() == "true"
            if not china_flag:
                # Also check mapsName for China-related keywords
                maps_name = str(data.get("mapsName", "") or "")
                china_keywords = ["中国", "北京", "上海", "广州", "深圳", "杭州", "成都",
                                "重庆", "武汉", "南京", "西安", "China", "广东", "浙江",
                                "江苏", "山东", "河南", "湖北", "湖南", "四川", "福建"]
                if not any(kw in maps_name for kw in china_keywords):
                    continue

            # ---- Extract GPS ----
            lat = data.get("lat") or data.get("latitude")
            lng = data.get("lng") or data.get("lon") or data.get("longitude")
            if lat is None or lng is None:
                rounds = data.get("rounds", [])
                if rounds:
                    r0 = rounds[0]
                    lat = r0.get("lat") or r0.get("targetLat")
                    lng = r0.get("lng") or r0.get("targetLng") or r0.get("targetLon")
            if lat is None or lng is None:
                continue

            lat, lng = float(lat), float(lng)

            # ---- Extract metadata ----
            rounds = data.get("rounds", [])
            r0 = rounds[0] if rounds else {}
            nation = str(data.get("nation", "") or r0.get("nation", "") or "")
            pano_id = str(r0.get("panoId", "") or "")
            maps_name = str(data.get("mapsName", "") or "")

            # Fallback bbox check for non-china-flagged items
            if not china_flag and not is_china_location(lat, lng, nation):
                continue

            # ---- Grid dedup ----
            cell_key = f"{round(lat, grid_size)},{round(lng, grid_size)}"
            if cell_key in seen_cells:
                continue
            seen_cells.add(cell_key)

            # Detect source
            pid = pano_id.lower()
            if "baidu" in pid or pid.startswith("b"):
                source = "baidu"
            elif "gaode" in pid or "amap" in pid or pid.startswith("g"):
                source = "gaode"
            else:
                source = "google"

            locations.append({
                "lat": lat,
                "lng": lng,
                "pano_id": pano_id,
                "source": source,
                "province": "",
                "city": "",
                "maps_name": maps_name,
            })

    logger.info("geocomp_china_filtered",
                total=len(locations),
                cells=len(seen_cells),
                rows_scanned=total_rows,
                files_scanned=len(files))
    return locations


# ============================================================
# Street View Download
# ============================================================

async def download_baidu_streetview(lat: float, lng: float,
                                    pano_id: str = "") -> bytes | None:
    """Download Baidu Street View image at given coordinates."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            if pano_id:
                url = f"https://mapsv0.bdimg.com/?qt=pr3d&fovy=90&quality=80&panoid={pano_id}&width=640&height=480"
            else:
                # Use coordinate-based URL
                url = (
                    f"https://api.map.baidu.com/panorama/v2?"
                    f"ak=deeE2e1DGudGlGjGjrGjGU7jjG7GIGjY&"
                    f"width=640&height=480&location={lng},{lat}&fov=90"
                )
            resp = await client.get(url)
            if resp.status_code == 200 and len(resp.content) > 1000:
                return resp.content
    except Exception as e:
        logger.debug("baidu_sv_failed", lat=lat, lng=lng, error=str(e)[:80])
    return None


async def download_gaode_streetview(lat: float, lng: float) -> bytes | None:
    """Download Gaode (Amap) static map at given coordinates.

    Note: Amap doesn't have a public street view API.
    Using static map as fallback (satellite view with labels).
    """
    import httpx
    from app.config import get_settings
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
                    "zoom": 16,
                    "size": "640*480",
                    "scale": 1,
                },
            )
            if resp.status_code == 200 and len(resp.content) > 1000:
                return resp.content
    except Exception as e:
        logger.debug("gaode_map_failed", lat=lat, lng=lng, error=str(e)[:80])
    return None


async def download_streetview(loc: dict) -> bytes | None:
    """Try to download street view from available sources."""
    # Try Baidu first (has real street view in China)
    img = await download_baidu_streetview(loc["lat"], loc["lng"], loc.get("pano_id", ""))
    if img:
        return img
    # Fall back to Gaode static map
    img = await download_gaode_streetview(loc["lat"], loc["lng"])
    if img:
        return img
    return None


# ============================================================
# OSV-5M China Subset
# ============================================================

async def download_osv5m_china(max_images: int = 5000,
                               cache_dir: str = "./data/osv5m") -> list[dict]:
    """Download OSV-5M China subset metadata and images.

    OSV-5M is on HuggingFace: osv5m/osv5m
    Metadata has country/region/nearest_city columns for filtering.

    Returns list of {"lat": float, "lng": float, "path": str, "city": str}
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.warning("datasets_not_installed",
                       hint="pip install datasets")
        return []

    logger.info("loading_osv5m_metadata")
    try:
        # Load metadata only (full=False skips image download)
        ds = load_dataset("osv5m/osv5m", full=False, split="train",
                          streaming=True)
    except Exception as e:
        logger.warning("osv5m_load_failed", error=str(e)[:120])
        return []

    results: list[dict] = []
    seen_cells: set[str] = set()
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    import httpx

    for item in ds:
        if len(results) >= max_images:
            break

        country = str(item.get("country", "") or "")
        lat = float(item.get("latitude", 0) or item.get("lat", 0))
        lng = float(item.get("longitude", 0) or item.get("lng", 0) or item.get("lon", 0))

        if not is_china_location(lat, lng, country):
            continue

        # Grid dedup
        cell_key = f"{round(lat, 3)},{round(lng, 3)}"
        if cell_key in seen_cells:
            continue
        seen_cells.add(cell_key)

        city = str(item.get("nearest_city", "") or item.get("city", "") or "")
        image_url = item.get("image_url", "") or item.get("url", "") or ""

        # Try to download image
        img_path = None
        if image_url:
            img_hash = hashlib.md5(image_url.encode()).hexdigest()[:12]
            img_path = cache / f"osv5m_{img_hash}.jpg"
            if not img_path.exists():
                try:
                    async with httpx.AsyncClient(timeout=20) as client:
                        resp = await client.get(image_url)
                        if resp.status_code == 200 and len(resp.content) > 1000:
                            img_path.write_bytes(resp.content)
                except Exception:
                    img_path = None

        if img_path and img_path.exists():
            results.append({
                "lat": lat,
                "lng": lng,
                "path": str(img_path),
                "city": city,
                "source": "osv5m",
            })

    logger.info("osv5m_china_downloaded", count=len(results))
    return results


# ============================================================
# Batch Ingestion into GeoImageDB
# ============================================================

async def ingest_into_db(db, images_dir: Path,
                         locations: list[dict],
                         batch_size: int = 50) -> int:
    """Batch-ingest location images into GeoImageDB with CLIP encoding.

    Args:
        db: GeoImageDB instance
        images_dir: Directory to save downloaded images
        locations: List of location dicts with lat/lng
        batch_size: How many images to process before saving checkpoint

    Returns:
        Number of images successfully added
    """
    import numpy as np
    from datetime import datetime, timezone
    from app.tools.clip_search import get_embedder

    embedder = get_embedder()

    # Build set of existing GPS coords to skip duplicates
    existing_coords: set[tuple[float, float]] = set()
    for meta in db.metadata.values():
        existing_coords.add((round(meta.get("lat", 0), 3), round(meta.get("lon", 0), 3)))

    added = 0
    skipped_dups = 0
    batch_paths: list[str] = []
    batch_meta: list[dict] = []

    # Filter: scan for already-downloaded images first
    existing_images: dict[str, str] = {}  # grid_key → path
    for img_file in images_dir.glob("geocomp_*.jpg"):
        # Filename format: geocomp_LAT_LNG_HASH.jpg
        parts = img_file.stem.split("_")
        if len(parts) >= 3:
            try:
                lat = round(float(parts[1]), 3)
                lng = round(float(parts[2]), 3)
                key = f"{lat},{lng}"
                existing_images[key] = str(img_file)
            except ValueError:
                pass

    to_download = []
    to_ingest = []
    for l in locations:
        if l.get("path"):
            to_ingest.append(l)
        else:
            key = f"{round(l['lat'], 3)},{round(l['lng'], 3)}"
            if key in existing_images:
                l["path"] = existing_images[key]
                to_ingest.append(l)
            else:
                to_download.append(l)

    logger.info("ingest_plan",
                to_download=len(to_download),
                to_ingest=len(to_ingest),
                reused_existing=len(to_ingest) - len([l for l in locations if l.get("path")]))

    # Download street views
    sem = asyncio.Semaphore(8)  # Limit concurrent downloads

    async def _download_one(loc: dict) -> dict | None:
        async with sem:
            img = await download_streetview(loc)
            if img:
                safe_name = f"geocomp_{loc['lat']:.4f}_{loc['lng']:.4f}_{hashlib.md5(str(loc).encode()).hexdigest()[:6]}"
                img_path = images_dir / f"{safe_name}.jpg"
                img_path.write_bytes(img)
                loc["path"] = str(img_path)
                return loc
        return None

    if to_download:
        results = await asyncio.gather(*[_download_one(l) for l in to_download])
        for r in results:
            if r:
                to_ingest.append(r)

    logger.info("download_complete", total=len(to_ingest))

    # Batch CLIP encode and insert
    for i, loc in enumerate(to_ingest):
        img_path = Path(loc["path"])
        if not img_path.exists():
            continue

        # Skip if GPS already in DB (within grid precision)
        coord_key = (round(loc["lat"], 3), round(loc["lng"], 3))
        if coord_key in existing_coords:
            skipped_dups += 1
            continue
        existing_coords.add(coord_key)

        city = loc.get("city", "") or loc.get("province", "") or ""
        tags = [f"geocomp_{loc.get('source', 'unknown')}", f"grid_{round(loc['lat'], 3)}_{round(loc['lng'], 3)}"]

        batch_paths.append(str(img_path))
        batch_meta.append({
            "lat": loc["lat"],
            "lng": loc["lng"],
            "city": city,
            "tags": tags,
            "source": f"geocomp_{loc.get('source', 'unknown')}",
        })

        if len(batch_paths) >= batch_size:
            # Batch encode
            embeddings = embedder.encode_images(batch_paths)
            if embeddings is not None:
                for j, emb in enumerate(embeddings):
                    meta = batch_meta[j]
                    vec = emb.reshape(1, -1)
                    img_id = db._next_id
                    db._next_id += 1
                    db.index.add_with_ids(vec, np.array([img_id], dtype=np.int64))
                    db.metadata[img_id] = {
                        "lat": meta["lat"],
                        "lon": meta["lng"],
                        "city": meta["city"],
                        "tags": meta["tags"],
                        "source": meta["source"],
                        "added_at": datetime.now(timezone.utc).isoformat(),
                    }
                    added += 1
            else:
                # Fall back to single encoding
                for j, path in enumerate(batch_paths):
                    meta = batch_meta[j]
                    img_id = db.add(path, meta["lat"], meta["lng"],
                                    city=meta["city"],
                                    tags=meta["tags"],
                                    source=meta["source"])
                    if img_id >= 0:
                        added += 1

            batch_paths.clear()
            batch_meta.clear()

            if added % 500 == 0:
                db.save()
                logger.info("ingest_checkpoint", added=added, total_in_db=db.count())

    # Process remaining
    if batch_paths:
        embeddings = embedder.encode_images(batch_paths)
        if embeddings is not None:
            for j, emb in enumerate(embeddings):
                meta = batch_meta[j]
                vec = emb.reshape(1, -1)
                img_id = db._next_id
                db._next_id += 1
                db.index.add_with_ids(vec, np.array([img_id], dtype=np.int64))
                db.metadata[img_id] = {
                    "lat": meta["lat"],
                    "lon": meta["lng"],
                    "city": meta["city"],
                    "tags": meta["tags"],
                    "source": meta["source"],
                    "added_at": datetime.now(timezone.utc).isoformat(),
                }
                added += 1

    db.save()
    logger.info("ingest_complete", added=added, skipped_dups=skipped_dups,
                total_in_db=db.count())
    return added


# ============================================================
# Main
# ============================================================

async def main():
    import argparse
    import numpy as np
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(
        description="Ingest GeoComp + OSV-5M China street view into CLIP+FAISS DB")
    parser.add_argument("--max-images", type=int, default=10000,
                        help="Maximum images to ingest (default: 10000)")
    parser.add_argument("--grid-size", type=int, default=3,
                        help="GPS grid precision for dedup (3=~500m)")
    parser.add_argument("--geocomp-parquet", type=str,
                        default="./data/tuxun_shards/data",
                        help="Path to GeoComp parquet file or shard directory")
    parser.add_argument("--osv5m-count", type=int, default=5000,
                        help="Max OSV-5M images (default: 5000)")
    parser.add_argument("--skip-geocomp", action="store_true",
                        help="Skip GeoComp, only use OSV-5M")
    parser.add_argument("--skip-osv5m", action="store_true",
                        help="Skip OSV-5M, only use GeoComp")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview only, don't download")
    args = parser.parse_args()

    from app.config import get_settings
    from app.tools.clip_search import get_db

    settings = get_settings()

    print("=" * 60)
    print("GeoComp + OSV-5M → CLIP+FAISS Ingestion Pipeline")
    print("=" * 60)
    print(f"DB path:       {settings.clip_db_path}")
    print(f"Max images:    {args.max_images}")
    print(f"GPS grid:      {args.grid_size} decimals (~{111 / (10**args.grid_size):.0f}m)")
    print(f"GeoComp:       {'skip' if args.skip_geocomp else args.geocomp_parquet}")
    print(f"OSV-5M:        {'skip' if args.skip_osv5m else f'max {args.osv5m_count}'}")
    print()

    if args.dry_run:
        print("[DRY RUN] No images will be downloaded.")
        # Still show what would be processed
        if not args.skip_geocomp and Path(args.geocomp_parquet).exists():
            locs = load_geocomp_china(args.geocomp_parquet,
                                      max_locations=args.max_images,
                                      grid_size=args.grid_size)
            print(f"  GeoComp China locations: {len(locs)} (after dedup)")
        return

    # Setup
    db = get_db(settings.clip_db_path)
    images_dir = Path(settings.clip_db_path) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    start_count = db.count()
    all_locations: list[dict] = []

    # Step 1: GeoComp
    if not args.skip_geocomp:
        geocomp_path = Path(args.geocomp_parquet)
        if not geocomp_path.exists():
            print(f"\n[!] GeoComp parquet not found: {geocomp_path}")
            print("    Download from: https://huggingface.co/datasets/ShirohAO/tuxun")
            print("    File: data/tuxun_combined.parquet")
        else:
            geocomp_locs = load_geocomp_china(
                str(geocomp_path),
                max_locations=args.max_images,
                grid_size=args.grid_size,
            )
            all_locations.extend(geocomp_locs)
            print(f"GeoComp China locations: {len(geocomp_locs)}")

    # Step 2: OSV-5M
    if not args.skip_osv5m:
        osv5m_locs = await download_osv5m_china(
            max_images=args.osv5m_count,
            cache_dir="./data/osv5m",
        )
        all_locations.extend(osv5m_locs)
        print(f"OSV-5M China images:     {len(osv5m_locs)}")

    print(f"\nTotal locations:          {len(all_locations)}")
    print(f"Current DB size:          {start_count}")

    if not all_locations:
        print("\nNo locations to ingest. Check data sources.")
        return

    # Step 3: Ingest
    t_start = time.time()
    added = await ingest_into_db(db, images_dir, all_locations)
    elapsed = int(time.time() - t_start)

    print(f"\n{'=' * 60}")
    print(f"Ingestion complete ({elapsed}s)")
    print(f"  Added:     {added}")
    print(f"  DB before: {start_count}")
    print(f"  DB after:  {db.count()}")
    print(f"  Increase:  +{db.count() - start_count}")

    stats = db.stats()
    print(f"\nTop cities:")
    for city, count in stats.get("top_cities", [])[:20]:
        print(f"  {city}: {count}")


if __name__ == "__main__":
    asyncio.run(main())
