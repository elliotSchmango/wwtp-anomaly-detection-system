# HRSD WADS: Operator Startup Checklist

**System:** WWTP Anomaly Detection System (WADS)  
**Hardware Target:** NVIDIA Jetson Orin Nano & TP-Link Tapo C260  
**Role:** Facility Operator / Technician

---

# 1. Equipment Verification

Ensure all items are present before beginning.

- [ ] NVIDIA Jetson Orin Nano (Developer Kit)
- [ ] MicroSD Card (128GB+, Class 10/UHS-1)
- [ ] MicroSD → SD Adapter (for connecting card to laptop)
- [ ] Power Supply (included with Jetson)
- [ ] Monitor (**DisplayPort required** — HDMI adapters often fail on Jetson)
- [ ] DisplayPort Cable
- [ ] Keyboard & Mouse (USB)
- [ ] TP-Link Tapo C260 Camera
- [ ] USB-C Cable (for Tapo camera power)
- [ ] Laptop/PC with Internet (for flashing SD card)
- [ ] WiFi Credentials (SSID & Password)

---

# 2. Camera Configuration (Mobile App)

**Goal:** Connect camera to WiFi and enable local control.

## Power On

Plug in **Tapo C260** and wait for the LED to blink **Red/Green**.

## App Setup

1. Download the **Tapo App** on your phone.
2. Follow instructions to connect the camera to the facility WiFi.

---

## CRITICAL STEP: Create Local Camera Account

Do **not** confuse this with the TP-Link Cloud login.

Navigate to:

```
Settings (Gear Icon) → Advanced Settings → Camera Account
```

Create a generic username/password.

Example:

```
User: admin
Pass: hrsd1234
```

Record credentials:

```
User: ___________________

Pass: ___________________
```

---

## Get Camera IP Address

Navigate to:

```
Device Settings → Device Info
```

Record IP address:

```
Camera IP: ___________________
```

*(Tech Note: If possible, configure a Static IP for the camera in the router.)*

---

# Set PTZ Zones

The system works best with **one zone**.

Multiple zones are supported but require editing code.

1. Go to **Live View**
2. Move camera to desired position
3. Tap **Pan & Tilt → Presets → Add Preset**

Name the preset:

```
1
```

Optional additional zones:

```
2
3
```

⚠ **IMPORTANT**

If additional zones are created you must update:

```
config/settings.py
```

Update the `ZONES` list to match the preset numbers.

After editing the file rebuild the system:

```bash
docker-compose down
docker-compose up --build
```

---

# 3. Jetson Hardware Initialization

**Goal:** Install Operating System.

## Flash SD Card (Laptop)

NVIDIA Setup Guide:

https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit

Scroll to:

```
Write Image to the microSD Card
```

### Version Warning

Do **NOT** download:

```
JetPack 7
JetPack 7.1
```

These are for **Thor chips only**.

Download:

```
JetPack 6.x SD Card Image for Orin Nano
```

File size should be approximately **6–9 GB**.

Install:

```
BalenaEtcher
```

Flash process:

1. Insert microSD into laptop
2. Open Etcher
3. Select Image
4. Select Drive
5. Flash

Wait **~30 minutes** for flash and validation.

---

# Initial Boot (Jetson)

1. Insert the microSD card into the slot under the Jetson module.
2. Connect:

- DisplayPort Monitor
- Keyboard
- Mouse

3. Plug in power.

Green LED should illuminate.

Follow on-screen setup wizard.

Recommended configuration:

```
WiFi: same network as camera
User: orin-desktop
Login: automatic login
```

---

# 4. Software Environment Setup

Open Terminal:

```
Ctrl + Alt + T
```

Run commands sequentially.

---

## Update System

```bash
sudo apt update
sudo apt install -y curl nano git
```

---

## Verify Docker Installation

```bash
docker --version
```

If Docker is missing:

```bash
sudo apt install -y docker.io
```

Install Docker Compose:

```bash
sudo apt install -y docker-compose
```

---

## Configure Docker Permissions

```bash
sudo usermod -aG docker $USER
```

⚠ **Restart the Jetson now**

After reboot open terminal again.

---

## Download WADS Software

```bash
git clone https://github.com/elliotSchmango/wwtp-anomaly-detection-system.git
cd wwtp-anomaly-detection-system
```

---

## Patch Dependencies (Critical)

Fixes Python 3.11 + OpenCV dependency conflict.

```bash
sed -i 's/numpy==2.4.0/numpy==2.2.3/g' requirements.txt
```

---

# 5. Application Configuration

Create settings file:

```bash
cp .env.example .env
nano .env
```

Edit variables:

```
# Local Camera Settings
CAMERA_IP="192.168.1.XX"
CAMERA_USER="admin"
CAMERA_PASS="hrsd1234"

# Gemini AI
USE_GEMINI=True
GEMINI_API_KEY="AIzaSy..."

# Azure Storage
AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net"
```

Save:

```
Ctrl + O
Enter
Ctrl + X
```

---

# 6. Initial Launch (One Time)

Navigate to project folder:

```bash
cd wwtp-anomaly-detection-system
```

Build and start:

```bash
docker-compose up --build
```

Wait **5–10 minutes** for the build.

---

## Open Dashboard

Open browser on Jetson.

```
http://localhost:8501
```

---

## Stop System

Press:

```
Ctrl + C
```

in the terminal.

---

# 7. Daily Operation / Restarting

After shutdown or power outage:

```bash
cd wwtp-anomaly-detection-system
docker-compose up
```

Open dashboard:

```
http://localhost:8501
```

---

# 8. Maintenance

## Reset Configuration

If camera IP, password, `.env`, or `settings.py` changes:

```bash
docker-compose down
docker-compose up --build
```

---

## Clear Calibration Data

If camera moves or retraining is required:

```bash
sudo rm -rf data/zone_*
```

Restart calibration from the dashboard.

---

# 9. Full System Re-Installation

Backup settings:

```bash
cp .env ~/env_backup
```

Clear Docker cache:

```bash
sudo docker rm -f wwtp-anomaly-guard eagle
sudo docker system prune -f
```

Reinstall:

```bash
cd ..
sudo rm -rf wwtp-anomaly-detection-system
git clone https://github.com/elliotSchmango/wwtp-anomaly-detection-system.git
cd wwtp-anomaly-detection-system
```

Restore settings:

```bash
cp ~/env_backup .env
```

Reapply patch and build:

```bash
sed -i 's/numpy==2.4.0/numpy==2.2.3/g' requirements.txt
sudo docker-compose up --build
```

---

# Troubleshooting

## "KeyError: ContainerConfig"

```bash
sudo docker rm -f wwtp-anomaly-guard eagle
sudo docker-compose up --build
```

---

## ".env not a directory"

```bash
sudo docker-compose down
sudo rm -rf .env
nano .env
sudo docker-compose up
```

---

## "ValidationError / CAMERA_IP missing"

Check settings file:

```bash
cat .env
```

If empty recreate configuration.

---

## Network timeout during build

Check internet:

```bash
ping google.com
```

Restart Docker:

```bash
sudo systemctl restart docker
```

---

## Docker permission errors

Run command with sudo:

```bash
sudo docker-compose up --build
```

---

## Dependency errors (pip / numpy)

Re-run patch:

```bash
sed -i 's/numpy==2.4.0/numpy==2.2.3/g' requirements.txt
```

---

# 10. FUSED Anomaly Metric

FUSED is the default anomaly scoring method. It is a weighted composite of four visual signals that each detect a different class of anomaly.

| Component | Weight | What It Detects |
|---|---|---|
| **S** — SSIM Residual | 30% | Structural and brightness changes |
| **G** — Gradient Divergence | 25% | Hard edges, silhouettes, fallen objects |
| **F** — FFT Energy Shift | 25% | Surface texture changes, corrosion, fouling |
| **L** — Latent Cosine Distance | 20% | Semantic scene deviation (requires retrain) |

## Activating FUSED

FUSED is enabled by default (`TRAINING_MODE=fused` in `.env`).

No additional configuration is required. Components S, G, and F are active immediately on any existing model.

## Threshold Tuning

The default threshold is `0.15`. Adjust in `config/settings.py`:

```python
AE_THRESHOLD_FUSED: float = 0.12
```

Run Sentry Mode for several minutes on normal conditions and observe the FUSED scores in the dashboard status bar. Set your threshold slightly above the highest normal score you observe.

⚠ **Rebuild required after editing `settings.py`.** Restart with:

```bash
docker-compose up --build
```

## Enabling the L Component (Latent Cosine Distance)

The L component requires the model to be retrained with `TRAINING_MODE=fused`. Existing models trained on `mse` or `ssim` will silently skip L (its weight drops to zero automatically).

To enable it:

1. Run **Calibrate** from the dashboard to collect normal frames.
2. Set `TRAINING_MODE=fused` in `.env`.
3. Run **Train** from the dashboard.
4. Restart Sentry Mode.

After retraining, the system will automatically compute a reference latent vector from your calibration images at startup and enable the L component.

---

# 11. Inference Pipeline Improvements

Three improvements were made to the detection pipeline to reduce false positives, improve Gemini accuracy, and reduce API costs.

## Temporal Consistency (Anti-Flicker)

A single anomalous frame no longer triggers an alert. The system maintains a rolling window of recent frames per zone and requires a minimum number of consecutive anomaly hits before firing.

**Default behavior:** 3 of the last 5 frames must score above threshold.

Tune in `config/settings.py`:

```python
TEMPORAL_WINDOW: int = 5
TEMPORAL_MIN_HITS: int = 3
```

Increase `TEMPORAL_MIN_HITS` to reduce false positives. Decrease it if real threats are being missed.

---

## Smart Reference Selection

Previously, Gemini was given the first `.jpg` found in the calibration folder as its "Normal" reference — an arbitrary choice that could itself be noisy.

The system now computes a **pixel-wise median** across all calibration images at startup. This produces a single representative frame that averages out transient noise (steam, water ripple, lighting flicker) and provides Gemini with the most stable possible baseline.

No configuration required. This runs automatically whenever Sentry Mode starts.

---

## Confidence-Gated Gemini Routing

When `USE_GEMINI=True`, the system no longer always calls the Gemini API. Instead, the local classifier is run first:

- If classifier confidence is **≥ 0.85**: the classifier label is used and Gemini is skipped (saves API cost).
- If classifier confidence is **< 0.85**: Gemini is called with the median reference frame for accurate classification.

Tune the gate in `config/settings.py`:

```python
CLASSIFIER_CONFIDENCE_GATE: float = 0.85
```

Lower the value to escalate more detections to Gemini. Raise it to rely more on the local classifier.

---