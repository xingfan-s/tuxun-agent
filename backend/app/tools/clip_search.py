"""
CLIP + FAISS image similarity search for geo-location.

Encodes images with CLIP ViT-L/14 (reusing transformers from geoclip dep),
indexes embeddings with FAISS for fast retrieval, returns visually similar
images with their known GPS coordinates.

Two classes:
  CLIPEmbedder — singleton image encoder (768-dim embeddings)
  GeoImageDB   — FAISS index + JSON metadata store
"""

import json
import hashlib
import os
import time
from app.utils.logging import structlog
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from app.geolocation.index_manifest import build_manifest, faiss_path, validate_manifest
from app.tools.model_runtime import MODEL_LOAD_LOCK

# Use HuggingFace mirror for China access (force override)
os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT") or "https://hf-mirror.com"

logger = structlog.get_logger()


def _model_offline() -> bool:
    return any(
        os.environ.get(name, "false").lower() in {"1", "true", "yes", "on"}
        for name in ("MODEL_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE")
    )


def _normalize_embeddings(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, 1e-12)

# ============================================================
# CLIP Embedder
# ============================================================

_embedder = None


def get_embedder():
    """Lazy-load CLIP embedder singleton."""
    global _embedder
    if _embedder is None:
        with MODEL_LOAD_LOCK:
            if _embedder is None:
                _embedder = CLIPEmbedder()
    return _embedder


class CLIPEmbedder:
    """CLIP ViT-L/14 image encoder (768-dim embeddings)."""

    def __init__(self):
        from transformers import CLIPModel, CLIPImageProcessor
        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14", local_files_only=_model_offline(),
        )
        self.processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-large-patch14", local_files_only=_model_offline(),
        )
        self.model.eval()

    def encode_image(self, image_path: str) -> np.ndarray | None:
        """Encode a single image → (768,) float32 array."""
        import torch
        from PIL import Image
        try:
            img = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=img, return_tensors="pt")
            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
            # features is BaseModelOutputWithPooling; use pooler_output
            emb = features.pooler_output if hasattr(features, 'pooler_output') else features
            return _normalize_embeddings(emb[0].numpy().astype(np.float32))
        except Exception as e:
            logger.error("clip_encode_failed", path=image_path, error=str(e))
            return None

    def encode_images(self, image_paths: list[str]) -> np.ndarray | None:
        """Encode multiple images → (N, 768) float32 array."""
        import torch
        from PIL import Image
        try:
            images = [Image.open(p).convert("RGB") for p in image_paths]
            inputs = self.processor(images=images, return_tensors="pt")
            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
            emb = features.pooler_output if hasattr(features, 'pooler_output') else features
            return _normalize_embeddings(emb.numpy().astype(np.float32))
        except Exception as e:
            logger.error("clip_batch_encode_failed", count=len(image_paths), error=str(e))
            return None

    @property
    def dim(self) -> int:
        return self.model.config.projection_dim  # 768


# ============================================================
# Geo Image Database
# ============================================================

class GeoImageDB:
    """FAISS-backed geo-tagged image database.

    Stores normalized CLIP embeddings in a FAISS inner-product index (cosine)
    metadata (GPS, city, tags) in a companion JSON file.

    Usage:
        db = GeoImageDB("./data/geo_image_db")
        db.add(image_path, lat=30.274, lon=120.155, city="杭州")
        matches = db.search(query_image_path, top_k=5)
        db.save()  # persist to disk
    """

    def __init__(self, db_dir: str = "./data/geo_image_db", *, require_v2: bool = False):
        self.db_dir = Path(db_dir)
        self.require_v2 = require_v2
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.db_dir / "index.faiss"
        self.meta_path = self.db_dir / "metadata.json"
        self.manifest_path = self.db_dir / "manifest.json"

        self._embedder = None
        self._next_id = 0
        self.metadata: dict[int, dict] = {}
        self._is_ivf = False  # v2.1: track index type for auto-upgrade
        self._metric = "cosine"

        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            raise ImportError("faiss-cpu required: pip install faiss-cpu")

        self._load_or_create()

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder

    def _create_index(self):
        """Create a new empty FAISS index."""
        dim = self.embedder.dim
        # Start with flat index for < 5000 images, upgrade to IVF later
        base_index = self.faiss.IndexFlatIP(dim)
        self.index = self.faiss.IndexIDMap(base_index)
        self._is_ivf = False
        logger.info("faiss_index_created", dim=dim, type="flat")

    def _upgrade_to_ivf(self):
        """Upgrade from flat index to IVF for faster search at scale."""
        if self._is_ivf:
            return

        n_total = self.index.ntotal
        if n_total < 5000:
            return  # Don't bother with IVF for small datasets

        dim = self.embedder.dim
        # Number of centroids: sqrt(N) is a good rule of thumb
        nlist = max(16, min(int(n_total ** 0.5), 4096))

        try:
            # Extract all vectors from current IDMap index
            # IDMap indices: reconstruct() takes the flat position, not the user ID
            vectors = np.zeros((n_total, dim), dtype=np.float32)
            valid_count = 0
            for flat_pos in range(n_total):
                try:
                    vec = self.index.reconstruct(flat_pos)
                    vectors[valid_count] = vec
                    valid_count += 1
                except Exception:
                    pass
            vectors = vectors[:valid_count]
            n_total = valid_count

            if n_total < 1000:
                logger.warning("faiss_ivf_upgrade_skipped", reason="too_few_valid",
                             valid=valid_count)
                return

            # Create IVF inside IDMap (IDMap must wrap an EMPTY index)
            quantizer = self.faiss.IndexFlatIP(dim)
            ivf_base = self.faiss.IndexIVFFlat(quantizer, dim, nlist,
                                                self.faiss.METRIC_INNER_PRODUCT)
            new_index = self.faiss.IndexIDMap(ivf_base)

            # Train the IVF (on the base index inside IDMap)
            ivf_base.train(vectors)
            ivf_base.nprobe = min(nlist // 4, 32)

            # Add vectors with their original user IDs
            id_array = np.array(sorted(self.metadata.keys()), dtype=np.int64)
            # Align vectors with sorted IDs
            sorted_meta = sorted(self.metadata.items(), key=lambda x: x[0])
            sorted_vectors = np.zeros((len(sorted_meta), dim), dtype=np.float32)
            for j, (uid, _) in enumerate(sorted_meta):
                try:
                    sorted_vectors[j] = vectors[j]
                except Exception:
                    pass

            new_index.add_with_ids(sorted_vectors[:len(id_array)],
                                   id_array[:len(sorted_vectors)])

            self.index = new_index
            self._is_ivf = True
            logger.info("faiss_upgraded_to_ivf", nlist=nlist, nprobe=ivf_base.nprobe,
                       n_total=n_total)
        except Exception as e:
            logger.warning("faiss_ivf_upgrade_failed", error=str(e),
                         detail="keeping flat index")

    def _load_or_create(self):
        """Load existing index + metadata from disk, or create new."""
        if self.index_path.exists() and self.meta_path.exists():
            try:
                self.index = self.faiss.read_index(faiss_path(self.index_path))
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.metadata = {int(k): v for k, v in data.get("entries", {}).items()}
                self._next_id = data.get("next_id", 0)
                if self.manifest_path.exists():
                    with open(self.manifest_path, "r", encoding="utf-8") as f:
                        manifest = json.load(f)
                    validate_manifest(
                        manifest,
                        dimension=int(getattr(self.index, "d", 0)),
                        entry_count=int(self.index.ntotal),
                        metadata=self.metadata,
                        require_v2=self.require_v2,
                    )
                elif self.require_v2:
                    raise ValueError("v2 CLIP index requires manifest.json")
                self._normalize_legacy_index_if_needed()
                # v2.1: Detect index type
                raw_index = self.index.index if hasattr(self.index, 'index') else self.index
                self._is_ivf = hasattr(raw_index, 'nprobe') or hasattr(raw_index, 'nlist')
                if self._is_ivf and hasattr(raw_index, 'nprobe'):
                    raw_index.nprobe = min(raw_index.nlist // 4, 32) if hasattr(raw_index, 'nlist') else 8
                logger.info("faiss_index_loaded", total=self.count(),
                          path=str(self.index_path),
                          type="ivf" if self._is_ivf else "flat")
                return
            except ValueError:
                raise
            except Exception as e:
                logger.warning("faiss_load_failed", error=str(e),
                             detail="creating new index")

        self._create_index()
        self._next_id = 0
        self.metadata = {}

    def _normalize_legacy_index_if_needed(self):
        """Convert a legacy L2 index to an in-memory cosine/IP index.

        v1 persisted unnormalized CLIP features with IndexFlatL2. Converting
        only in memory keeps fallback read-only while making its ranking
        comparable to the normalized v2 query vectors.
        """
        raw_index = self.index.index if hasattr(self.index, "index") else self.index
        if getattr(raw_index, "metric_type", self.faiss.METRIC_INNER_PRODUCT) != self.faiss.METRIC_L2:
            self._metric = "cosine"
            return
        count = int(self.index.ntotal)
        if count == 0:
            self._metric = "cosine"
            return
        dim = int(raw_index.d)
        vectors = np.vstack(
            [raw_index.reconstruct(position) for position in range(count)]
        ).astype(np.float32)
        vectors = _normalize_embeddings(vectors)
        if hasattr(self.index, "id_map"):
            ids = self.faiss.vector_to_array(self.index.id_map).astype(np.int64)
        else:
            ids = np.arange(count, dtype=np.int64)
        if len(ids) != count:
            raise ValueError("legacy index IDs and vectors are inconsistent")
        normalized = self.faiss.IndexIDMap2(self.faiss.IndexFlatIP(dim))
        normalized.add_with_ids(vectors, ids)
        self.index = normalized
        self._metric = "cosine"
        logger.info("legacy_clip_index_normalized", total=count, metric="cosine")

    def add(self, image_path: str, lat: float, lon: float,
            city: str = "", tags: list[str] | None = None,
            source: str = "manual") -> int:
        """Add an image to the database. Returns the assigned ID."""
        vec = self.embedder.encode_image(image_path)
        if vec is None:
            return -1

        vec_2d = _normalize_embeddings(vec.reshape(1, -1))
        img_id = self._next_id
        self._next_id += 1

        self.index.add_with_ids(vec_2d, np.array([img_id], dtype=np.int64))
        self.metadata[img_id] = {
            "lat": lat,
            "lon": lon,
            "city": city,
            "tags": tags or [],
            "source": source,
            "added_at": datetime.now(timezone.utc).isoformat(),
        }

        # v2.1: Auto-upgrade to IVF at 5000 images
        if not self._is_ivf and self.count() >= 5000:
            logger.info("faiss_triggering_ivf_upgrade", count=self.count())
            self._upgrade_to_ivf()

        return img_id

    def search(self, image_path: str, top_k: int = 5) -> list[dict]:
        """Search for visually similar images.

        Returns:
            List of dicts sorted by similarity (closest first):
            [{"distance": float, "lat": float, "lon": float,
              "city": str, "tags": list, "source": str}, ...]
        """
        if self.count() == 0:
            return []

        vec = self.embedder.encode_image(image_path)
        if vec is None:
            return []

        vec_2d = vec.reshape(1, -1)

        # v2.1: Set nprobe dynamically for IVF indexes based on dataset size
        if self._is_ivf:
            base_index = self.index.index if hasattr(self.index, 'index') else self.index
            if hasattr(base_index, 'nprobe'):
                n_total = self.count()
                # More probes = more accurate but slower
                base_index.nprobe = min(max(n_total // 1000, 4), 64)

        distances, ids = self.index.search(vec_2d, min(top_k, self.count()))

        results = []
        for dist, img_id in zip(distances[0], ids[0]):
            if img_id == -1:
                continue
            meta = self.metadata.get(int(img_id), {})
            similarity = float(dist)
            results.append({
                "similarity": similarity,
                "distance": 1.0 - similarity,
                "lat": meta.get("lat"),
                "lon": meta.get("lon"),
                "city": meta.get("city", ""),
                "tags": meta.get("tags", []),
                "source": meta.get("source", ""),
            })

        return results

    def count(self) -> int:
        """Number of images in the database."""
        return self.index.ntotal if hasattr(self.index, 'ntotal') else 0

    def save(self):
        """Persist index and metadata to disk."""
        self.faiss.write_index(self.index, faiss_path(self.index_path))
        data = {
            "next_id": self._next_id,
            "entries": {str(k): v for k, v in self.metadata.items()},
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        manifest = build_manifest(
            self.metadata,
            dimension=self.embedder.dim,
            index_type="ivf" if self._is_ivf else "flat",
        )
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        logger.info("faiss_index_saved", total=self.count(), path=str(self.db_dir))

    def stats(self) -> dict:
        """Return summary statistics."""
        cities = {}
        for m in self.metadata.values():
            c = m.get("city", "unknown")
            cities[c] = cities.get(c, 0) + 1
        return {
            "total": self.count(),
            "top_cities": sorted(cities.items(), key=lambda x: x[1], reverse=True)[:10],
        }


# ============================================================
# Module-level convenience
# ============================================================

_db: GeoImageDB | None = None
_db_path: str | None = None


def _has_persisted_index(path: Path) -> bool:
    return (path / "index.faiss").is_file() and (path / "metadata.json").is_file()


def get_db(db_dir: str | None = None, fallback_dir: str | None = None) -> GeoImageDB:
    """Get or create the global GeoImageDB singleton."""
    global _db, _db_path
    if db_dir is None:
        try:
            from app.config import get_settings
            settings = get_settings()
            db_dir = settings.clip_db_path
            fallback_dir = fallback_dir or settings.clip_db_fallback_path
        except Exception:
            db_dir = "./data/geo_image_db"
    primary = Path(db_dir).expanduser().resolve()
    fallback = Path(fallback_dir).expanduser().resolve() if fallback_dir else None
    selected = primary
    require_v2 = primary.name == "geo_image_db_v2"
    if not _has_persisted_index(primary) and fallback and _has_persisted_index(fallback):
        selected = fallback
        require_v2 = False
        logger.warning("clip_index_fallback", primary=str(primary), fallback=str(fallback))
    resolved = str(selected)
    if _db is None or _db_path != resolved:
        try:
            _db = GeoImageDB(resolved, require_v2=require_v2)
        except (ValueError, OSError) as exc:
            if selected == primary and fallback and _has_persisted_index(fallback):
                logger.warning("clip_index_invalid_fallback", primary=str(primary), error=str(exc))
                resolved = str(fallback)
                _db = GeoImageDB(resolved, require_v2=False)
            else:
                raise
        _db_path = resolved
    return _db


def search_similar_images(image_path: str, top_k: int = 5) -> dict | None:
    """Convenience function: search CLIP+FAISS for similar images.

    Returns dict with keys: matches, db_size, or None on failure.
    """
    try:
        db = get_db()
        t0 = time.time()
        matches = db.search(image_path, top_k)
        elapsed_ms = int((time.time() - t0) * 1000)
        return {
            "matches": matches,
            "db_size": db.count(),
            "search_ms": elapsed_ms,
        }
    except Exception as e:
        logger.error("clip_search_failed", error=str(e))
        return None


# LangChain tool wrapper for ReAct loop
from langchain.tools import tool as langchain_tool

@langchain_tool
def search_similar_images_tool(image_path: str, top_k: int = 5) -> dict:
    """CLIP+FAISS 本地图片库检索。在本地已索引的地理图片库中搜索视觉最相似的图片，
    返回其GPS坐标和城市信息。用于发现视觉上与已知地点相似的位置。

    Args:
        image_path: 图片文件路径
        top_k: 返回前K个最相似结果（默认5）
    """
    result = search_similar_images(image_path, top_k=top_k)
    if result is None:
        return {"matches": [], "db_size": 0, "error": "search_failed"}
    return result
