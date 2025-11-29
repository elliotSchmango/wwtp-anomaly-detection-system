import time
import cv2
import logging
from datetime import datetime
from core.camera import CameraClient
from core.inference import InferenceEngine
from core.state import StateManager
from azure.telemetry import TelemetrySender
from config.settings import settings

logger = logging.getLogger("Eagle.Sentry")

class SentryLoop:
    def __init__(self, stop_event):
        self.stop_event = stop_event
        self.state = StateManager()
        self.telemetry = TelemetrySender()
        
    def run(self):
        #initialize resources
        camera = CameraClient()
        if not camera.connect():
            self.state.update(status="Camera Connection Failed", mode="ERROR")
            return

        ai = InferenceEngine()
        
        #sentry loop:
        try:
            logger.info("Sentry Mode Started")
            while not self.stop_event.is_set():
                
                for zone in settings.ZONES:
                    if self.stop_event.is_set(): break
                    
                    #1) move camera
                    self.state.update(current_zone=zone, status=f"Scanning Zone {zone}...")
                    camera.move_to_preset(zone)
                    
                    #2) capture frame
                    #wait to settle & grab fresh frame
                    time.sleep(0.5) 
                    frame = camera.get_frame()
                    
                    if frame is None:
                        logger.warning("Empty frame received")
                        continue
                    
                    self.state.update(latest_frame=frame) #update UI

                    #run through AE model
                    mse = ai.detect_anomaly(frame)
                    is_anomaly = mse > settings.AE_THRESHOLD
                    
                    label = "Normal"
                    
                    if is_anomaly:
                        #now classification if anomaly logic passes
                        label, confidence = ai.classify_anomaly(frame)
                        timestamp = datetime.now().isoformat()
                        
                        #alert
                        log_msg = f"ALERT Zone {zone}: {label} ({confidence:.2f}) MSE: {mse:.5f}"
                        logger.warning(log_msg)
                        
                        #sending to azure
                        self.telemetry.send_alert(zone, mse, label, timestamp)
                        
                        #display UI updates
                        self.state.update(
                            is_anomaly=True, 
                            last_anomaly_label=label,
                            last_anomaly_score=mse,
                            last_detection_time=timestamp,
                            log=log_msg
                        )
                        
                        #save image locally to data/
                        snap_name = f"ALERT_{timestamp}_Z{zone}_{label}.jpg".replace(":", "-")
                        cv2.imwrite(str(settings.DATA_DIR / snap_name), frame)
                    
                    else:
                        self.state.update(
                            is_anomaly=False,
                            status=f"Zone {zone} Clear (MSE {mse:.5f})"
                        )

                    #loop delay
                    time.sleep(1.0 / settings.SENTRY_FPS)

        except Exception as e:
            logger.error(f"Sentry Crash: {e}")
            self.state.update(status=f"Sentry Error: {str(e)}", mode="ERROR")
        finally:
            camera.release()
            self.telemetry.disconnect()
            logger.info("Sentry Mode Stopped")
            
            #reset to "IDLE" ONLY if we weren't interrupted by another start command
            if self.stop_event.is_set():
                self.state.set_mode("IDLE")