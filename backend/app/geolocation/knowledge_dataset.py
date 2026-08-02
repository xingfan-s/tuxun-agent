"""Versioned metadata validation for the curated geography knowledge base."""

import json
from pathlib import Path


MANIFEST_PATH = Path(__file__).with_name("knowledge_manifest.json")


def load_manifest() -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not manifest.get("dataset_version"):
        raise ValueError("invalid geography knowledge manifest")
    if not isinstance(manifest.get("sources"), list) or not manifest["sources"]:
        raise ValueError("knowledge manifest requires sources")
    return manifest
