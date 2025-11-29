import threading
import time
from dataclasses import dataclass, asdict
from typing import Optional, Any

@dataclass
class AppState:
    mode: str = "IDLE" #options are: IDLE, SENTRY, CALIBRATING, TRAINING
    status: str = "System Ready"
    current_zone: int = 0
    
    #anomaly data
    is_anomaly: bool = False
    last_anomaly_score: float = 0.0
    last_anomaly_label: str = "None"
    last_detection_time: Optional[str] = None

    #for visualization
    latest_frame: Any = None
    
    progress: int = 0
    logs: list[str] = None

    def __post_init__(self):
        if self.logs is None:
            self.logs = []

class StateManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(StateManager, cls).__new__(cls)
            cls._instance.state = AppState()
        return cls._instance

    #thread safe updates
    def update(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self.state, k):
                    setattr(self.state, k, v)
                elif k == "log":
                    #handler to append logs
                    timestamp = time.strftime("%H:%M:%S")
                    self.state.logs.append(f"[{timestamp}] {v}")
                    if len(self.state.logs) > 50:  #keeping last 50 logs
                        self.state.logs.pop(0)

    #Returns copy of state, then updating UI
    def get_snapshot(self) -> dict:
        with self._lock:
            #avoiding deepcopy problems with numpy arrays
            d = asdict(self.state)
            d['latest_frame'] = self.state.latest_frame
            return d
        

    def set_mode(self, mode: str):
        with self._lock:
            self.state.mode = mode
            self.state.is_anomaly = False  #resets anomaly flag on mode switch