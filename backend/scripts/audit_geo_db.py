#!/usr/bin/env python3
"""Report duplicate hashes, coordinate validity and city coverage."""

from __future__ import annotations

import argparse
import json
import hashlib
from collections import Counter
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.geolocation.index_manifest import metadata_fingerprint


def audit(db_dir: Path, images_dir: Path | None = None) -> dict:
    metadata_path = db_dir / "metadata.json"
    if not metadata_path.exists():
        return {"error": f"metadata not found: {metadata_path}"}
    entries = json.loads(metadata_path.read_text(encoding="utf-8")).get("entries", {})
    coordinate_counts = Counter()
    city_counts = Counter()
    source_counts = Counter()
    invalid_coordinates = []
    for key, item in entries.items():
        lat, lng = item.get("lat"), item.get("lon", item.get("lng"))
        if not isinstance(lat, (int, float)) or not isinstance(lng, (int, float)) or not (-90 <= lat <= 90 and -180 <= lng <= 180):
            invalid_coordinates.append(key)
        else:
            coordinate_counts[(round(lat, 5), round(lng, 5))] += 1
        city_counts[item.get("city") or "unknown"] += 1
        source_counts[item.get("source") or "unknown"] += 1

    images_dir = images_dir or (db_dir / "images")
    if not images_dir.is_dir():
        # v2 deliberately contains only FAISS + metadata; reuse the source
        # image tree for duplicate-hash auditing without copying large files.
        source_images = db_dir.parent / "geo_image_db" / "images"
        if source_images.is_dir():
            images_dir = source_images
    image_hashes = Counter()
    image_count = 0
    if images_dir.is_dir():
        for image_path in images_dir.rglob("*"):
            if not image_path.is_file():
                continue
            digest = hashlib.sha256()
            with image_path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            image_hashes[digest.hexdigest()] += 1
            image_count += 1

    index_count = None
    try:
        import faiss
        index_count = int(faiss.read_index(str(db_dir / "index.faiss")).ntotal)
    except Exception:
        pass

    manifest_path = db_dir / "manifest.json"
    manifest = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    return {
        "entry_count": len(entries),
        "metadata_count": len(entries),
        "index_count": index_count,
        "index_metadata_count_match": index_count is None or index_count == len(entries),
        "image_count": image_count,
        "image_source": str(images_dir) if image_count else None,
        "duplicate_image_hash_count": sum(max(0, count - 1) for count in image_hashes.values()),
        "invalid_coordinate_count": len(invalid_coordinates),
        "duplicate_coordinate_count": sum(1 for n in coordinate_counts.values() if n > 1),
        "city_count": len(city_counts),
        "cities_below_10": sum(1 for n in city_counts.values() if n < 10),
        "top_cities": city_counts.most_common(20),
        "source_distribution": dict(source_counts),
        "manifest_present": manifest is not None,
        "manifest_entry_count_match": manifest is None or manifest.get("entry_count") == len(entries),
        "manifest_fingerprint_match": manifest is None or manifest.get("data_fingerprint") == metadata_fingerprint({int(k): v for k, v in entries.items()}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="./data/geo_image_db_v2")
    parser.add_argument("--images-dir", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(audit(Path(args.db), args.images_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
