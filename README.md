# HRSD's wwtp-anomaly-detection-system (WADS)

## Modules

### Core
- **Configuration:** Managed via `config/settings.py` using Pydantic. Loads secrets from `.env`.
- **Camera:** `core/camera.py` handles ONVIF PTZ control and non-blocking RTSP streaming.