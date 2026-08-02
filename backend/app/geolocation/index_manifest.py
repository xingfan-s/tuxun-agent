"""Versioned metadata contract for CLIP/FAISS indexes."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

INDEX_SCHEMA_VERSION = 2
CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
CLIP_MODEL_REVISION = "local"


def faiss_path(path: str | Path) -> str:
    """Return a FAISS-compatible path on Windows.

    The Windows FAISS wheel can reject existing files when given an absolute
    path containing non-ASCII characters. The app runs from ``backend`` and
    its data paths are below that directory, so an ASCII relative path is
    both stable and portable. Fall back to the original path when a relative
    ASCII representation is not available.
    """
    resolved = Path(path).expanduser()
    try:
        relative = os.path.relpath(str(resolved), os.getcwd())
        if not relative.startswith("..") and all(ord(char) < 128 for char in relative):
            return relative
    except (OSError, ValueError):
        pass
    return str(resolved)


def metadata_fingerprint(metadata: dict[int, dict[str, Any]]) -> str:
    payload = json.dumps(
        {str(key): value for key, value in sorted(metadata.items())},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_manifest(
    metadata: dict[int, dict[str, Any]],
    *,
    dimension: int,
    index_type: str,
    model_revision: str = CLIP_MODEL_REVISION,
) -> dict[str, Any]:
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "model_id": CLIP_MODEL_ID,
        "model_revision": model_revision,
        "dimension": int(dimension),
        "metric": "cosine",
        "normalized": True,
        "index_type": index_type,
        "entry_count": len(metadata),
        "data_fingerprint": metadata_fingerprint(metadata),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }


def validate_manifest(
    manifest: dict[str, Any],
    *,
    dimension: int,
    entry_count: int,
    metadata: dict[int, dict[str, Any]] | None = None,
    require_v2: bool = False,
) -> None:
    if require_v2 and manifest.get("schema_version") != INDEX_SCHEMA_VERSION:
        raise ValueError("unsupported CLIP index schema version")
    if manifest.get("model_id") not in (None, CLIP_MODEL_ID):
        raise ValueError("CLIP index model does not match runtime model")
    if manifest.get("metric") not in (None, "cosine"):
        raise ValueError("CLIP index metric must be cosine")
    if manifest.get("normalized") is False:
        raise ValueError("CLIP index vectors must be normalized")
    if manifest.get("dimension") not in (None, int(dimension)):
        raise ValueError("CLIP index dimension does not match runtime model")
    if manifest.get("entry_count") not in (None, int(entry_count)):
        raise ValueError("CLIP index entry count does not match metadata")
    fingerprint = manifest.get("data_fingerprint")
    if not fingerprint:
        raise ValueError("CLIP index manifest is missing data_fingerprint")
    if metadata is not None and fingerprint != metadata_fingerprint(metadata):
        raise ValueError("CLIP index metadata fingerprint mismatch")
