import time
import cv2
import logging
from datetime import datetime
from pathlib import Path
from core.camera import CameraClient
from core.inference import InferenceEngine
from core.state import StateManager
from core.llm import GeminiAgent
from cloud.telemetry import TelemetrySender
from config.settings import settings

logger = logging.getLogger("Eagle.Sentry")

class SentryLoop:
    def __init__(self, stop_event):
        self.stop_event = stop_event
        self.state = StateManager()
        self.telemetry = TelemetrySender()
        self.llm = GeminiAgent() #initialize agent
        
    def run(self):
        camera = CameraClient()
        if not camera.connect():
            self.state.update(status="Camera Connection Failed", mode="ERROR")
            return

        ai = InferenceEngine()
        
        #determine mode & threshold
        if settings.TRAINING_MODE == "ssim":
            metric_label = "SSIM Loss"
            current_threshold = settings.AE_THRESHOLD_SSIM
        else:
            metric_label = "MSE"
            current_threshold = settings.AE_THRESHOLD_MSE

        try:
            logger.info(f"Sentry Mode Started ({metric_label} Mode, Threshold: {current_threshold})")
            
            while not self.stop_event.is_set():
                
                for zone in settings.ZONES:
                    if self.stop_event.is_set(): break
                    
                    #1: move camera
                    self.state.update(current_zone=zone, status=f"Scanning Zone {zone}...")
                    camera.move_to_preset(zone)
                    
                    #2: get frame
                    # Wait for movement to settle
                    time.sleep(0.5) 
                    frame = camera.get_frame()
                    
                    if frame is None:
                        logger.warning("Empty frame received")
                        continue

                    #update UI view
                    self.state.update(latest_frame=frame)

                    #3: inference
                    anomaly_score = ai.detect_anomaly(frame) #returns anomaly score
                    
                    #check threshold dynamically
                    is_anomaly = anomaly_score > current_threshold
                    
                    label = "Normal"
                    
                    if is_anomaly:
                        #4: classification
                        if settings.USE_GEMINI:
                            zone_dir = settings.DATA_DIR / f"zone_{zone}"
                            #retrieve first jpg we find in the calibration folder to use as baseline
                            ref_path = next(zone_dir.glob("*.jpg"), None)
                            
                            if ref_path:
                                ref_frame = cv2.imread(str(ref_path))
                                label = self.llm.analyze(ref_frame, frame)
                            else:
                                label = "No Reference Data"
                        else: #otherwise, use local classifier
                            label, confidence = ai.classify_anomaly(frame)

                        timestamp = datetime.now().isoformat()
                        log_msg = f"ALERT Zone {zone}: {label} ({metric_label}: {anomaly_score:.5f})"
                        logger.warning(log_msg)
                        
                        #upload to Azure
                        self.telemetry.send_alert(zone, anomaly_score, label, timestamp)
                        
                        #and update UI State
                        self.state.update(
                            is_anomaly=True, 
                            last_anomaly_label=label,
                            last_anomaly_score=anomaly_score,
                            last_detection_time=timestamp,
                            log=log_msg
                        )
                        
                        #save locally
                        snap_name = f"ALERT_{timestamp}_Z{zone}_{label}.jpg".replace(":", "-")
                        cv2.imwrite(str(settings.DATA_DIR / snap_name), frame)
                    
                    else:
                        self.state.update(
                            is_anomaly=False,
                            status=f"Zone {zone} Clear ({metric_label}: {anomaly_score:.5f})"
                        )
                    time.sleep(1.0 / settings.SENTRY_FPS)

        except Exception as e:
            logger.error(f"Sentry Crash: {e}")
            self.state.update(status=f"Sentry Error: {str(e)}", mode="ERROR")
        finally:
            camera.release()
            self.telemetry.disconnect()
            logger.info("Sentry Mode Stopped")
            
            # Reset to IDLE only if we weren't interrupted by another task start
            if self.stop_event.is_set():
                self.state.set_mode("IDLE")