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
    SENTRY_FPS: int = 1
    #list of ONVIF preset indices to scan (saved on camera)
    ZONES: list[int] = [1]
    #rolling window size for temporal consistency check
    TEMPORAL_WINDOW: int = 5
    #minimum anomaly hits within window required to fire an alert
    TEMPORAL_MIN_HITS: int = 3
    #classifier confidence above which Gemini is skipped (cost saving)
    CLASSIFIER_CONFIDENCE_GATE: float = 0.85
    
    #model Parameters
    AE_IMG_SIZE: int = 224
    AE_THRESHOLD_MSE: float = 0.004
    AE_THRESHOLD_SSIM: float = 0.05   #has (1 - SSIM) score. edit as needed
    AE_THRESHOLD_FUSED: float = 0.15  #normalized composite score, tune after calibration
    
    #Google Gemini & Local VLM integration
    LLM_PROVIDER: str = "gemini" # options: "gemini", "local"
    LLM_PROMPT_TEMPLATE: str = "concise" # options: "concise", "detailed"
    USE_GEMINI: bool = True # legacy toggle, consider LLM_PROVIDER instead
    GEMINI_API_KEY: str = "" #from .env
    
    # Local VLM Config (for heavier local models via Ollama/REST)
    LOCAL_VLM_URL: str = "http://localhost:11434/api/generate"
    LOCAL_VLM_MODEL: str = "llava:34b" # computationally expensive default
    
    # PTZ Stabilization
    ALIGNMENT_ENABLED: bool = True
    CROP_MARGIN_PERCENT: float = 0.05

    #Azure integration (empty string disables IoT Hub telemetry)
    IOTHUB_CONN_STRING: str = ""
    DEVICE_ID: str = "test-site"
    TELEMETRY_IMAGE_FORMAT: str = "jpg"
    TELEMETRY_IMAGE_QUALITY: int = 70
    TELEMETRY_IMAGE_MAX_WIDTH: int = 640

    #"fused" (SSIM+Gradient+FFT+Latent), "ssim" (SSIM+L1 denoising), or "mse" (baseline)
    TRAINING_MODE: str = "fused"
    
    class Config:
        env_file = ".env"
        extra = "ignore"

#singleton architecture
settings = Settings()

# Ensure critical directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)