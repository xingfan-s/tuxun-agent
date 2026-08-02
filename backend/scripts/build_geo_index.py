#!/usr/bin/env python3
"""Rebuild a normalized cosine/IP v2 index without modifying v1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.geolocation.index_manifest import build_manifest


def _load_entries(source: Path) -> dict[int, dict]:
    metadata_path = source / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing source metadata: {metadata_path}")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {int(key): value for key, value in payload.get("entries", {}).items()}


def build(source: Path, output: Path, limit: int = 0) -> dict:
    import faiss
    import numpy as np

    entries = _load_entries(source)
    source_index_path = source / "index.faiss"
    if not entries or not source_index_path.is_file():
        raise ValueError("source index must contain index.faiss and metadata entries")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")

    source_index = faiss.read_index(str(source_index_path))
    base_index = faiss.downcast_index(source_index.index) if hasattr(source_index, "index") else source_index
    source_ids = (
        faiss.vector_to_array(source_index.id_map).astype(np.int64)
        if hasattr(source_index, "id_map") else np.arange(source_index.ntotal, dtype=np.int64)
    )
    count = min(int(source_index.ntotal), len(source_ids))
    if limit:
        count = min(count, limit)
    vectors = np.vstack([base_index.reconstruct(position) for position in range(count)]).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("source index contains zero-length embeddings")
    vectors /= norms
    ids = source_ids[:count]
    selected_metadata = {int(entry_id): entries[int(entry_id)] for entry_id in ids if int(entry_id) in entries}
    if len(selected_metadata) != count:
        raise ValueError("source index IDs and metadata IDs are inconsistent")

    output.mkdir(parents=True, exist_ok=True)
    target_index = faiss.IndexIDMap2(faiss.IndexFlatIP(int(source_index.d)))
    target_index.add_with_ids(vectors, ids)
    faiss.write_index(target_index, str(output / "index.faiss"))
    metadata_payload = {
        "next_id": max(selected_metadata, default=-1) + 1,
        "entries": {str(key): value for key, value in selected_metadata.items()},
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = build_manifest(selected_metadata, dimension=int(source_index.d), index_type="flat")
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "source": str(source),
        "output": str(output),
        "source_entries": len(entries),
        "source_vectors": int(source_index.ntotal),
        "added": count,
        "normalized_min": float(np.linalg.norm(vectors, axis=1).min()),
        "normalized_max": float(np.linalg.norm(vectors, axis=1).max()),
        "manifest": str(output / "manifest.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a versioned v2 CLIP index")
    parser.add_argument("--source", type=Path, default=Path("./data/geo_image_db"))
    parser.add_argument("--output", type=Path, default=Path("./data/geo_image_db_v2"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output, args.limit), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
