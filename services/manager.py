import threading
import logging
from core.state import StateManager
from services.sentry import SentryLoop
from services.calibration import CalibrationTask

logger = logging.getLogger("State.Manager") #tag and output log messages

class ServiceManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ServiceManager, cls).__new__(cls)
            cls._instance.active_thread = None
            cls._instance.stop_event = threading.Event()
        return cls._instance

    def _start_thread(self, target_function, mode_name, status_msg):
        """Helper to safely swap background threads"""
        self.stop_active()
        
        #resetting stop event for new thread
        self.stop_event.clear()
        StateManager().set_mode(mode_name)
        StateManager().update(status=status_msg, progress=0)
        
        self.active_thread = threading.Thread(target=target_function, daemon=True)
        self.active_thread.start()
        logger.info(f"Started service: {mode_name}")

    def start_sentry(self):
        task = SentryLoop(self.stop_event)
        self._start_thread(task.run, "SENTRY", "Starting Sentry...")

    def start_calibration(self):
        task = CalibrationTask(self.stop_event)
        self._start_thread(task.run, "CALIBRATING", "Starting Calibration...")

    def start_training(self):
        from services.training import TrainingTask #avoid looping imports
        task = TrainingTask(self.stop_event)
        self._start_thread(task.run, "TRAINING", "Starting Training...")

    #tells active thread to stop and waits for it to finish
    def stop_active(self):
        if self.active_thread and self.active_thread.is_alive():
            logger.info("Stopping active service...")
            StateManager().update(status="Stopping background tasks...")
            self.stop_event.set()
            self.active_thread.join(timeout=10) #wait 10 seconds
            self.active_thread = None
            logger.info("Service stopped.")
        
        StateManager().set_mode("IDLE")
        StateManager().update(status="System Ready", current_zone=0)