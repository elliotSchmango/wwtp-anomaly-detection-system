# HRSD's wwtp-anomaly-detection-system (WADS)
### By Elliot Hong

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

---

## How to Run:

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

## Hardware Setup (TP-Link Tapo C260)

**Note:** This project is specifically configured for the **TP-Link Tapo C260** Pan/Tilt Security Wi-Fi Camera.

### 1. Create a Camera Account
You **cannot** use your TP-Link Cloud email/password for this. You must create a specific local access account.
1.  Open the **Tapo App** on your phone.
2.  Tap on your C260 device card to enter the live view.
3.  Tap the **Gear Icon** (Settings) in the top right.
4.  Go to **Advanced Settings** -> **Camera Account**.
5.  Create a **Username** and **Password**.
    * *These will be the `CAMERA_USER` and `CAMERA_PASS` in your `.env` file.*

### 2. Get the IP Address
1.  While still in **Device Settings**, tap on **Device Info**.
2.  Find the **IP Address** (e.g., `192.168.1.50`).
    * *This will be the `CAMERA_IP` in your `.env` file.*
    * *Tip: It is highly recommended to set a **Static IP** for the camera in your Router settings so the IP doesn't change if the power goes out.*

### 3. Configure Zones (Presets)
The system uses the camera's onboard **Presets** to define the scanning zones. You must set these up manually in the Tapo App.
1.  Open the **Tapo App** and go to the Live View.
2.  Use the Pan/Tilt controls to move the camera to the first position you want to monitor.
3.  Tap **Pan & Tilt** -> **Presets** (or Marks).
4.  **Add Preset** and name it **"1"**.
5.  Move the camera to the next position and save it as **"2"**, then **"3"**.
    * *Note: The preset names must match the integers in `config/settings.py` (default: `ZONES = [1, 2, 3]`). If you want more zones, add presets "4", "5", etc., and update the list in `settings.py`.*

---

### Usage Guide

* **Sentry Mode:**
    * Toggles the main security loop.
    * The system will cycle through the defined **Zones** (PTZ presets).
    * It grabs frames, runs the **Autoencoder**, and checks for anomalies.
    * If an anomaly is detected, it runs the **Classifier** and logs the alert to Azure.

* **Calibrate:**
    * *Disable Sentry Mode first.*
    * Moves the camera to each zone and captures **50 frames** of "normal" data.
    * Images are saved to `data/zone_X/`.

* **Train:**
    * *Disable Sentry Mode first.*
    * Uses the data captured in **Calibration** to train a new Autoencoder model locally.
    * Automatically exports the result to `models/autoencoder.onnx`.
    * **Note:** Training may take a few minutes depending on your hardware.

---

## Training Guide

This system uses two different AI models. It is important to understand when to train each one.

### 1. Autoencoder (Anomaly Detector)
* **Role:** Learns what "Normal" looks like for your specific camera angle and lighting.
* **Data Source:** Captures images automatically via the **"Calibrate"** button.
* **When to Train:**
    * **Initially:** When you first install the camera.
    * **Environment Change:** If you move the camera, change the zoom, or if the lighting conditions change drastically (e.g., day vs. night).
    * **False Positives:** If the system keeps alerting on normal things, run Calibration and Train again to teach it those new normal conditions.

### 2. Classifier (Anomaly Type Identifier)
* **Role:** Identifies *what* the anomaly is (Fire, Leak, Corrosion, Human).
* **Data Source:** Requires a manually curated dataset in `data/classifier_data/`.
* **When to Train:**
    * **Once:** You typically only need to train this once. "Fire" always looks like fire.
    * **Do NOT Train Daily:** Unlike the Autoencoder, this model does not need to learn your specific room.
    * **New Capabilities:** Train again only if you add a new class (e.g., "Smoke") or if you find specific objects that confuse the model (add them to the training folders and retrain).

#### Classifier Data Structure
To train the classifier, organize your images like this:
```text
data/classifier_data/
├── Corrosion/
│   ├── img1.jpg
│   └── ...
├── Fire/
├── Human/
└── Leaky Pipes/
```

---