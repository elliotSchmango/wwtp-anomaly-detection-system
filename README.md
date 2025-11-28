# HRSD's wwtp-anomaly-detection-system (WADS)

## Modules

### Core
- **Configuration:** Managed via `config/settings.py` using Pydantic. Loads secrets from `.env`.
- **Camera:** `core/camera.py` handles ONVIF PTZ control and non-blocking RTSP streaming.

### Core Logic
- **State Management:** `core/state.py` provides a thread-safe Singleton (`StateManager`) to share data between the Sentry loop and the UI.
- **Inference:** `core/inference.py` abstracts ONNX Runtime. It handles:
  - Autoencoder (MSE calculation for Anomaly Detection).
  - Classifier (ImageNet-normalized classification for Anomaly Type).