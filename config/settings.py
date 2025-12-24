import os
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    #simplifying path names
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    MODELS_DIR: Path = BASE_DIR / "models"
    
    #camera settings for tapo (ONVIF)
    CAMERA_IP: str
    CAMERA_USER: str
    CAMERA_PASS: str
    RTSP_PORT: int = 554
    ONVIF_PORT: int = 2020
    
    #sentry mode config
    SENTRY_FPS: int = 5
    # List of ONVIF preset indices to scan (saved on camera)
    ZONES: list[int] = [1, 2, 3]  
    
    #model Parameters
    AE_IMG_SIZE: int = 224
    AE_THRESHOLD: float = 0.004
    
    #Azure integration
    IOTHUB_CONN_STRING: str = ""

    #"ssim" ("Novel") or "mse" (Baseline)
    TRAINING_MODE: str = "ssim"
    
    class Config:
        env_file = ".env"
        extra = "ignore"

#singleton architecture
settings = Settings()

# Ensure critical directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)