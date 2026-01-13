import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloud.telemetry import TelemetrySender
from config.settings import settings

def main():
    sender = TelemetrySender()
    payload = {
        "timestamp": datetime.now().isoformat(),
        "device_id": settings.DEVICE_ID,
        "zone_id": 0,
        "anomaly_metric": settings.TRAINING_MODE,
        "anomaly_score": 0.0,
        "anomaly_flag": False,
        "classifier_label": "Test",
        "frame": None,
        "llm_base_frame": None,
        "llm_current_frame": None,
        "llm_text_input": "TEST: no camera input",
        "llm_output": "TEST"
    }
    sender.send_payload(payload)
    sender.disconnect()
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
