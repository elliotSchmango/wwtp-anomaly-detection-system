# HRSD's wwtp-anomaly-detection-system (WADS)
## By Elliot Hong

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

### Frontend
- **UI:** `ui/app.py` is a Streamlit dashboard. It polls `StateManager` every second to update the video feed, status indicators, and logs. It communicates with `ServiceManager` to toggle Sentry and Calibration modes.

------

## How to Run

### Option A: Deployment (Docker) - Recommended

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/elliotschmango/wwtp-anomaly-detection-system.git](https://github.com/elliotschmango/wwtp-anomaly-detection-system.git)
    cd wwtp-anomaly-detection-system
    ```

2.  **Configure the System:**
    Copy the example environment file and edit it with your camera credentials.
    ```bash
    cp .env.example .env
    nano .env
    ```
    *Ensure `CAMERA_IP`, `CAMERA_USER`, and `CAMERA_PASS` are correct.*

3.  **Run with Docker Compose:**
    ```bash
    docker-compose up --build
    ```

4.  **Access the Dashboard:**
    Open your browser and navigate to:
    `http://localhost:8501`

---

### Option B: Local Development (Python venv)
Use this mode if you are modifying the code or don't have Docker installed.

1.  **Set up Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: .\venv\Scripts\Activate
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment:**
    Make sure your `.env` file is set up (see above).

4.  **Run the App:**
    ```bash
    streamlit run ui/app.py
    ```

---

### Usage Guide

* **🛡️ Sentry Mode:**
    * Toggles the main security loop.
    * The system will cycle through the defined **Zones** (PTZ presets).
    * It grabs frames, runs the **Autoencoder**, and checks for anomalies.
    * If an anomaly is detected, it runs