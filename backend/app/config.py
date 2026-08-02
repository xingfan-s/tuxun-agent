from pydantic import model_validator
from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    max_total_tool_calls: int = 24
    # 大模型
    qwen_api_key: str = ""
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_vl_model: str = "qwen-vl-max"
    qwen_model: str = "qwen-max"

    # 地图服务
    map_service: str = "nominatim"
    map_service_fallback: str = ""
    # Keep server and browser keys separate. `amap_api_key` is retained only
    # as a backwards-compatible migration input and is never returned by API.
    amap_server_key: str = ""
    amap_web_key: str = ""
    amap_api_key: str = ""
    google_maps_api_key: str = ""
    tencent_map_key: str = ""

    # 搜索
    search_service: str = "bing"
    serpapi_api_key: str = ""
    bing_search_api_key: str = ""

    # 以图搜图
    reverse_image_service: str = "none"

    # 应用
    upload_dir: str = str(BACKEND_DIR / "uploads")
    max_file_size_mb: int = 20
    max_react_loops: int = 4
    react_top_k: int = 5
    max_tool_elapsed_seconds: int = 120
    tool_timeout_seconds: int = 20
    worker_pool_size: int = 4
    task_timeout_seconds: int = 420

    # 安全
    safety_face_max_count: int = 3
    safety_face_policy: str = "signal"
    safety_delete_image_after_done: bool = True
    safety_require_api: bool = True  # fail closed when the scene-check API is unavailable
    safety_require_ocr: bool = False  # fail closed on OCR errors only when explicitly enabled
    max_image_pixels: int = 50_000_000
    upload_ttl_seconds: int = 30 * 60

    # GeoCLIP
    geoclip_enabled: bool = True
    model_offline: bool = False
    preload_models: bool = False

    # CLIP + FAISS image similarity search
    clip_search_enabled: bool = True
    clip_db_path: str = str(BACKEND_DIR / "data" / "geo_image_db_v2")
    clip_db_fallback_path: str = str(BACKEND_DIR / "data" / "geo_image_db")

    # Candidate fusion weights. They remain ranking weights until a validated
    # calibration artifact is configured.
    rank_weight_geoclip: float = 0.25
    rank_weight_clip: float = 0.20
    rank_weight_ocr: float = 0.28
    rank_weight_vision: float = 0.17
    rank_weight_knowledge: float = 0.10
    calibration_path: str = ""
    model_cost_per_1k_tokens: float = 0.0

    # 限流
    rate_limit: str = "10/minute"

    # PaddlePaddle MKL compatibility (AMD CPU)
    mkl_debug_cpu_type: str = "5"

    model_config = {
        "env_file": BACKEND_DIR / ".env",
        "env_file_encoding": "utf-8",
        "extra": "allow",
        "protected_namespaces": ("settings_",),
    }

    @model_validator(mode="after")
    def resolve_local_paths(self):
        for field in ("upload_dir", "clip_db_path", "clip_db_fallback_path"):
            value = Path(getattr(self, field)).expanduser()
            if not value.is_absolute():
                setattr(self, field, str((BACKEND_DIR / value).resolve()))
        return self

    @property
    def amap_server_api_key(self) -> str:
        return self.amap_server_key or self.amap_api_key


@lru_cache()
def get_settings() -> Settings:
    return Settings()
