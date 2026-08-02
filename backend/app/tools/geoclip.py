"""
GeoCLIP: Image-to-GPS geolocation model.

Wraps the geoclip PyPI package (VicenteVivan/geo-clip, NeurIPS 2023).
Predicts GPS coordinates directly from an image using a CLIP ViT-L/14
backbone with a location encoder trained on MP-16.

Model: ~314M params (277M frozen CLIP + trainable MLPs + location encoder).
Singleton loaded to avoid reloading on every inference call.

Network note: GeoCLIP's ImageEncoder calls AutoProcessor.from_pretrained()
which tries to download chat_template.jinja (does not exist for CLIP). On
machines without direct HuggingFace access, this triggers 5 retries (~30s).
We suppress this by setting TRANSFORMERS_OFFLINE=1 when the model is cached,
then restoring the original value after loading.
"""

import os
from app.utils.logging import structlog
from app.tools.model_runtime import MODEL_LOAD_LOCK
from pathlib import Path

logger = structlog.get_logger()

# HuggingFace mirror for China access
_HF_MIRROR = os.environ.get("HF_ENDPOINT") or "https://hf-mirror.com"
os.environ.setdefault("HF_ENDPOINT", _HF_MIRROR)

# Offline mode is opt-in. A clean Windows machine must be able to download
# model files on first use without editing Python source.
if os.environ.get("MODEL_OFFLINE", "false").lower() in {"1", "true", "yes", "on"}:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

_model = None
_load_attempted = False
_load_error: str | None = None

# Check if CLIP model is already cached so we can skip network requests
_CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
_CACHE_DIR = os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface"))
_MODEL_CACHE_ROOT = os.path.join(
    _CACHE_DIR, "hub", f"models--{_CLIP_MODEL_ID.replace('/', '--')}", "snapshots",
)
_MODEL_CACHED = False
if os.path.isdir(_MODEL_CACHE_ROOT):
    for snapshot in os.scandir(_MODEL_CACHE_ROOT):
        if not snapshot.is_dir():
            continue
        required = ("config.json", "model.safetensors", "preprocessor_config.json")
        if all(os.path.isfile(os.path.join(snapshot.path, name)) for name in required):
            _MODEL_CACHED = True
            break

# Transformers 5 probes for processor_config.json even though CLIP uses
# preprocessor_config.json. Enter offline mode before importing Transformers
# when a complete local model snapshot is available.
if _MODEL_CACHED:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


def _patch_geoclip_image_encoder(model):
    """Fix GeoCLIP for newer transformers that return BaseModelOutputWithPooling.

    In transformers >= 4.45, CLIPModel.get_image_features() returns
    BaseModelOutputWithPooling instead of a plain tensor. Patch the CLIP
    model's method to extract .pooler_output before GeoCLIP's mlp sees it.
    """
    try:
        image_encoder = model.image_encoder
        clip_model = image_encoder.CLIP
        original_get_image_features = clip_model.get_image_features

        def patched_get_image_features(pixel_values=None, **kwargs):
            result = original_get_image_features(pixel_values=pixel_values, **kwargs)
            if hasattr(result, 'pooler_output'):
                return result.pooler_output
            return result

        clip_model.get_image_features = patched_get_image_features
        logger.info("geoclip_patched", detail="BaseModelOutputWithPooling workaround")
    except Exception as e:
        logger.warning("geoclip_patch_failed", error=str(e))


def _get_model():
    """Lazy-load the GeoCLIP model singleton."""
    global _model, _load_attempted, _load_error

    if _model is not None:
        return _model

    with MODEL_LOAD_LOCK:
        if _model is not None:
            return _model
        if _load_attempted:
            return None

        _load_attempted = True

        try:
            import torch
            from geoclip import GeoCLIP
            _model = GeoCLIP()
            # Monkey-patch: newer transformers return BaseModelOutputWithPooling
            # from get_image_features(), but GeoCLIP expects a plain tensor.
            _patch_geoclip_image_encoder(_model)
            if torch.cuda.is_available():
                _model = _model.to("cuda")
                logger.info("geoclip_loaded", device="cuda")
            else:
                logger.info("geoclip_loaded", device="cpu")
            _model.eval()
            _load_error = None
        except ImportError as exc:
            logger.warning("geoclip_import_failed", error=str(exc))
            _load_error = f"ImportError: {exc}"
            return None
        except Exception as e:
            logger.error("geoclip_load_failed", error=str(e))
            _load_error = f"{type(e).__name__}: {e}"
            _model = None
            return None

        return _model


def predict_location(image_path: str, top_k: int = 10) -> list[dict] | None:
    """Predict GPS coordinates from an image.

    Args:
        image_path: Path to the image file.
        top_k: Number of top predictions to return.

    Returns:
        List of dicts [{"lat": float, "lon": float, "probability": float}, ...]
        sorted by probability descending, or None if model unavailable.
    """
    model = _get_model()
    if model is None:
        return None

    path = Path(image_path)
    if not path.exists():
        logger.warning("geoclip_image_not_found", path=str(path))
        return None

    try:
        import torch
        with torch.no_grad():
            gps, probs = model.predict(str(path), top_k=top_k)

        results = []
        for (lat, lon), prob in zip(gps, probs):
            results.append({
                "lat": round(float(lat), 6),
                "lon": round(float(lon), 6),
                "probability": round(float(prob), 6),
            })

        return results

    except Exception as e:
        logger.error("geoclip_inference_failed", error=str(e))
        return None


def is_available() -> bool:
    """Check if GeoCLIP model is loaded (does NOT trigger loading)."""
    return _model is not None


def get_load_status() -> str:
    if _model is not None:
        return "ok"
    if _load_error:
        return f"unavailable:{_load_error.split(':', 1)[0]}"
    return "lazy_load_pending"


def preload_model() -> bool:
    """Explicitly load the GeoCLIP model. Call during startup to avoid
    first-request latency. Returns True if loaded successfully."""
    return _get_model() is not None
