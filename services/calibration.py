import time
import cv2
import random
from pathlib import Path
from core.camera import CameraClient
from core.state import StateManager
from config.settings import settings

class CalibrationTask:
    def __init__(self, stop_event):
        self.stop_event = stop_event
        self.state = StateManager()
        
        #calibration config
        self.samples_per_zone = 50
        self.warmup_frames = 10
        self.save_every_n_frames = 5

    def run(self):
        camera = CameraClient()
        if not camera.connect():
            self.state.update(status="Camera Connection Failed", mode="ERROR")
            return

        try:
            #create base data directory
            settings.DATA_DIR.mkdir(parents=True, exist_ok=True)

            for zone in settings.ZONES:
                if self.stop_event.is_set(): break

                self.state.update(current_zone=zone, status=f"Moving to Zone {zone}...")
                camera.move_to_preset(zone)
                
                #init zone specific folders
                zone_dir = settings.DATA_DIR / f"zone_{zone}"
                zone_dir.mkdir(exist_ok=True)

                self.state.update(status=f"Capturing Zone {zone}...", progress=0)
                
                saved_count = 0
                frame_count = 0
                
                #allow a second to to let auto-focus/exposure to calibrate
                for _ in range(self.warmup_frames):
                    time.sleep(0.5) #0.5 second
                    camera.get_frame()

                while saved_count < self.samples_per_zone:
                    if self.stop_event.is_set(): break
                    
                    frame = camera.get_frame()
                    if frame is None:
                        time.sleep(0.1)
                        continue
                        
                    self.state.update(latest_frame=frame) #update UI

                    #save every n-th frame
                    frame_count += 1
                    if frame_count % self.save_every_n_frames == 0:
                        filename = f"{int(time.time())}_{saved_count}.jpg"
                        cv2.imwrite(str(zone_dir / filename), frame)
                        saved_count += 1
                        
                        #progress bar!
                        progress = int((saved_count / self.samples_per_zone) * 100)
                        self.state.update(progress=progress)
                    
                    time.sleep(0.1)

            if not self.stop_event.is_set():
                self.state.update(status="Calibration Complete", progress=100)
                time.sleep(2) #display completion message

        except Exception as e:
            self.state.update(status=f"Calibration Error: {e}", mode="ERROR")
        finally:
            camera.release()
            if self.state.get_snapshot()['mode'] != "ERROR":
                self.state.set_mode("IDLE")