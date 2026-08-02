import os

# Apply offline mode before importing model libraries. It is opt-in so a clean
# Windows installation can download its model files on first use.
if os.environ.get("MODEL_OFFLINE", "false").lower() in {"1", "true", "yes", "on"}:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

from app.utils.logging import structlog
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
try:
    from slowapi import _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware
    _SLOWAPI_AVAILABLE = True
except ImportError:
    _SLOWAPI_AVAILABLE = False
    RateLimitExceeded = Exception

    def _rate_limit_exceeded_handler(request, exc):
        return None

    class SlowAPIMiddleware:
        def __init__(self, app, *args, **kwargs):
            self.app = app

from app.config import get_settings

logger = structlog.get_logger()

settings = get_settings()

if settings.model_offline:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

# Ensure PaddlePaddle MKL compatibility env var is set before any imports
if settings.mkl_debug_cpu_type:
    os.environ.setdefault("MKL_DEBUG_CPU_TYPE", settings.mkl_debug_cpu_type)


def create_app() -> FastAPI:
    app = FastAPI(title="图寻 Agent", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://localhost:3000",
            "http://localhost:80",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if _SLOWAPI_AVAILABLE:
        from slowapi.extension import Limiter
        from slowapi.util import get_remote_address
        from pathlib import Path
        # slowapi delegates to Starlette Config and reads a cwd-relative
        # `.env` using the Windows code page by default. The project settings
        # already parsed the UTF-8 env file, so disable that second read.
        limiter = Limiter(
            key_func=get_remote_address,
            default_limits=[settings.rate_limit],
            config_filename=str(Path(__file__).with_name("__init__.py")),
        )
        app.state.limiter = limiter
        app.add_middleware(SlowAPIMiddleware)
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    from app.routers.task import router as task_router
    app.include_router(task_router, prefix="/api")

    return app


app = create_app()


@app.get("/health/live")
async def health_live():
    return {"status": "ok"}


@app.get("/metrics", response_class=PlainTextResponse)
async def application_metrics():
    from app.metrics import metrics
    return metrics.render_prometheus()


@app.get("/health/ready")
async def health_ready():
    """Report actionable startup dependencies instead of a generic 200."""
    import json
    from pathlib import Path

    checks: dict[str, str] = {}
    warnings: dict[str, str] = {}
    upload_dir = Path(settings.upload_dir)
    if not upload_dir.exists():
        checks["task_storage"] = "missing_upload_directory"
    elif not os.access(upload_dir, os.W_OK):
        checks["task_storage"] = "upload_directory_not_writable"
    if not _SLOWAPI_AVAILABLE:
        checks["rate_limit"] = "missing:slowapi"
    try:
        import langgraph  # noqa: F401
    except ImportError:
        checks["agent_runtime"] = "missing:langgraph"
    if settings.geoclip_enabled:
        try:
            from app.tools.geoclip import get_load_status
            geoclip_status = get_load_status()
            if geoclip_status != "ok":
                if settings.preload_models:
                    checks["geoclip"] = geoclip_status
                else:
                    warnings["geoclip"] = geoclip_status
        except Exception as exc:
            checks["geoclip"] = f"unavailable:{type(exc).__name__}"
    if settings.clip_search_enabled:
        index_path = Path(settings.clip_db_path) / "index.faiss"
        metadata_path = Path(settings.clip_db_path) / "metadata.json"
        manifest_path = Path(settings.clip_db_path) / "manifest.json"
        fallback_index_path = Path(settings.clip_db_fallback_path) / "index.faiss"
        if index_path.exists():
            try:
                import faiss
                from app.geolocation.index_manifest import faiss_path, validate_manifest

                index = faiss.read_index(faiss_path(index_path))
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))["entries"]
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                validate_manifest(
                    manifest,
                    dimension=int(getattr(index, "d", 0)),
                    entry_count=int(index.ntotal),
                    metadata={int(key): value for key, value in metadata.items()},
                    require_v2=True,
                )
            except FileNotFoundError:
                checks["clip_index"] = "missing_manifest"
            except Exception as exc:
                if fallback_index_path.exists():
                    warnings["clip_index"] = "fallback_v1_active"
                else:
                    checks["clip_index"] = f"invalid:{type(exc).__name__}"
        elif not fallback_index_path.exists():
            checks["clip_index"] = "missing_index"
        else:
            warnings["clip_index"] = "fallback_v1_active"
    try:
        from app.geolocation.knowledge_dataset import load_manifest
        load_manifest()
    except Exception as exc:
        checks["knowledge_dataset"] = f"invalid:{type(exc).__name__}"
    if settings.safety_require_api and not settings.qwen_api_key:
        checks["qwen_api_key"] = "missing"
    from app.safety.text_check import get_ocr_capability_status, probe_ocr_capability
    if settings.safety_require_ocr:
        try:
            probe_ocr_capability()
        except Exception as exc:
            checks["safety_ocr"] = f"unavailable:{type(exc).__name__}"
    else:
        ocr_status = get_ocr_capability_status()
        if ocr_status != "ok":
            warnings["safety_ocr"] = ocr_status
    if checks:
        raise HTTPException(
            status_code=503,
            detail={"status": "not_ready", "checks": checks, "warnings": warnings},
        )
    return {"status": "ready", "checks": {"models": "ok", "index": "ok"}, "warnings": warnings}


@app.on_event("startup")
async def startup():
    logger.info("app_startup", upload_dir=settings.upload_dir)
    from pathlib import Path
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)

    # Keep Windows startup responsive. Models load lazily unless explicitly
    # preloaded after their cache has been prepared.
    if settings.preload_models and settings.geoclip_enabled:
        try:
            from app.tools.geoclip import preload_model
            if preload_model():
                logger.info("geoclip_preloaded")
            else:
                logger.warning("geoclip_unavailable")
        except Exception as e:
            logger.warning("geoclip_preload_failed", error=str(e))

    # Preload CLIP embedder for FAISS (avoids ~5s first-query latency)
    if settings.preload_models and settings.clip_search_enabled:
        try:
            from app.tools.clip_search import get_embedder
            embedder = get_embedder()
            logger.info("clip_embedder_preloaded", dim=embedder.dim)
        except Exception as e:
            logger.warning("clip_embedder_preload_failed", error=str(e))


@app.on_event("shutdown")
async def shutdown():
    """Clean up resources: close httpx clients in global singletons."""
    from app.tools.map import _map_service
    if _map_service is not None:
        await _map_service.close()
        logger.info("map_service_closed")
