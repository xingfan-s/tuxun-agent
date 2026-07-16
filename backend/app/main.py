import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.config import get_settings

logger = structlog.get_logger()

settings = get_settings()


def create_app() -> FastAPI:
    app = FastAPI(title="图寻 Agent", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:80"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from slowapi.extension import Limiter
    limiter = Limiter(key_func=lambda: "global", default_limits=[settings.rate_limit])
    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    from app.routers.task import router as task_router
    app.include_router(task_router, prefix="/api")

    return app


app = create_app()


@app.on_event("startup")
async def startup():
    logger.info("app_startup", upload_dir=settings.upload_dir)
    from pathlib import Path
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
