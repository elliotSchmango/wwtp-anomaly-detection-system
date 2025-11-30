import cv2
import time
import threading
import logging
import os
from onvif import ONVIFCamera
from config.settings import settings

#logger config
logger = logging.getLogger("State.Camera")
logging.basicConfig(level=logging.INFO)

class CameraClient:
    def __init__(self):
        self.ptz = None
        self.media = None
        self.cap = None
        self.profile = None
        
        self.latest_frame = None
        self.last_frame_time = 0
        self.running = False
        self.lock = threading.Lock()
        
        # Build RTSP URL
        self.rtsp_url = (
            f"rtsp://{settings.CAMERA_USER}:{settings.CAMERA_PASS}@"
            f"{settings.CAMERA_IP}:{settings.RTSP_PORT}/stream1"
        )

        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

    #Connects to ONVIF control and starts RTSP stream thread
    def connect(self):
        try:
            #connect to ONVIF (PTZ Control)
            self.mycam = ONVIFCamera(
                settings.CAMERA_IP, 
                settings.ONVIF_PORT, 
                settings.CAMERA_USER, 
                settings.CAMERA_PASS
            )
            self.media = self.mycam.create_media_service()
            self.ptz = self.mycam.create_ptz_service()
            self.profile = self.media.GetProfiles()[0]
            
            #connect to RTSP (Video)
            self.cap = cv2.VideoCapture(self.rtsp_url)
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open RTSP stream at {self.rtsp_url}")
            
            self.running = True
            threading.Thread(target=self._update_loop, daemon=True).start()
            
            logger.info(f"Connected to Camera at {settings.CAMERA_IP}")
            return True
            
        except Exception as e:
            logger.error(f"Camera Connection Failed: {e}")
            return False

    def _update_loop(self):
        """Continuously grabs frames to keep the buffer empty (Low Latency)"""
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.latest_frame = frame
                    self.last_frame_time = time.time()
            else:
                time.sleep(0.1)

    #return most recent frame (thread-safe copy)
    def get_frame(self):
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None

    #moves camera to a specific ONVIF preset index
    def move_to_preset(self, preset_index: int):
        try:
            if not self.ptz:
                logger.warning("PTZ service not initialized")
                return

            request = self.ptz.create_type('GotoPreset')
            request.ProfileToken = self.profile.token
            request.PresetToken = str(preset_index)
            
            self.ptz.GotoPreset(request)
            logger.info(f"Moving to Preset {preset_index}...")
            
            #wait time for camera to move per zone
            time.sleep(3.0) 
            
        except Exception as e:
            logger.error(f"PTZ Move Failed: {e}")

    def release(self):
        self.running = False
        if self.cap:
            self.cap.release()
        logger.info("Camera released")