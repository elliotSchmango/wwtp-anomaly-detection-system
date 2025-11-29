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

### Services Logic
- **Manager:** `services/manager.py` is a thread-safe orchestrator. It prevents running Sentry and Calibration simultaneously.
- **Sentry:** `services/sentry.py` runs the main security loop:
  1. Moves Camera to Zone X.
  2. Runs Autoencoder (Anomaly Detection).
  3. Runs Classifier (If Anomaly Found).
  4. Uploads Alert to Azure.
- **Calibration:** `services/calibration.py` automates data collection. It moves to each zone and saves 50 images to `data/zone_X` for training.