import json

import numpy as np
import pytest

from app.tools import clip_search


faiss = pytest.importorskip("faiss")


def _write_index(path, *, metric, vectors):
    path.mkdir(parents=True)
    dim = vectors.shape[1]
    base = faiss.IndexFlatIP(dim) if metric == "ip" else faiss.IndexFlatL2(dim)
    index = faiss.IndexIDMap2(base)
    ids = np.arange(len(vectors), dtype=np.int64)
    index.add_with_ids(vectors.astype(np.float32), ids)
    faiss.write_index(index, str(path / "index.faiss"))
    entries = {
        str(int(index_id)): {
            "lat": 30 + int(index_id),
            "lon": 120 + int(index_id),
            "city": "test",
        }
        for index_id in ids
    }
    (path / "metadata.json").write_text(
        json.dumps({"next_id": len(entries), "entries": entries}), encoding="utf-8"
    )


def test_invalid_v2_manifest_falls_back_to_normalized_v1(tmp_path, monkeypatch):
    primary = tmp_path / "geo_image_db_v2"
    fallback = tmp_path / "geo_image_db"
    _write_index(primary, metric="ip", vectors=np.array([[1.0, 0.0]], dtype=np.float32))
    _write_index(fallback, metric="l2", vectors=np.array([[3.0, 4.0]], dtype=np.float32))
    (primary / "manifest.json").write_text(
        json.dumps({
            "schema_version": 2,
            "model_id": "openai/clip-vit-large-patch14",
            "dimension": 2,
            "metric": "cosine",
            "normalized": True,
            "entry_count": 1,
            "data_fingerprint": "invalid",
        }),
        encoding="utf-8",
    )

    monkeypatch.setattr(clip_search, "_db", None)
    monkeypatch.setattr(clip_search, "_db_path", None)
    db = clip_search.get_db(str(primary), str(fallback))

    raw_index = faiss.downcast_index(db.index.index)
    assert clip_search._db_path == str(fallback.resolve())
    assert raw_index.metric_type == faiss.METRIC_INNER_PRODUCT
    assert np.linalg.norm(raw_index.reconstruct(0)) == pytest.approx(1.0, abs=1e-6)
