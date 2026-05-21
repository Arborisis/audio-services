import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_env: str = "production"
    analyzer_secret: str = ""
    laravel_api_url: str = ""
    laravel_api_secret: str = ""

    cf_email: str = ""
    cf_api_key: str = ""

    r2_endpoint: str = ""
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    r2_bucket: str = "arborisis"
    r2_region: str = "auto"

    max_file_size_mb: int = 500
    max_duration_seconds: int = 600

    birdnet_confidence_threshold: float = 0.25
    birdnet_publish_confidence_threshold: float = 0.45
    birdnet_repeated_confidence_threshold: float = 0.32
    birdnet_max_detections: int = 80
    birdnet_model_path: str = "/opt/birdnet/checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite"

    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
