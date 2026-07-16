from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    # 大模型
    qwen_api_key: str = ""
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_vl_model: str = "qwen-vl-max"
    qwen_model: str = "qwen-max"

    # 地图服务
    map_service: str = "nominatim"
    map_service_fallback: str = ""
    amap_api_key: str = ""
    google_maps_api_key: str = ""

    # 搜索
    search_service: str = "bing"
    serpapi_api_key: str = ""
    bing_search_api_key: str = ""

    # 以图搜图
    reverse_image_service: str = "none"
    bing_visual_api_key: str = ""

    # 应用
    upload_dir: str = "./uploads"
    max_file_size_mb: int = 20
    max_react_loops: int = 10
    worker_pool_size: int = 4
    task_timeout_seconds: int = 120

    # 安全
    safety_face_max_count: int = 3
    safety_delete_image_after_done: bool = True

    # 限流
    rate_limit: str = "10/minute"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()
