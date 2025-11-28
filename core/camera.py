import cv2
import time
import threading
import logging
from onvif import ONVIFCamera
from config.settings import settings

#logger init
logger = logging.getLogger("Eagle.Camera")
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
        
        #building RTSP URL (Standard format for Tapo TP-Link Cam)
        # rtsp://user:pass@IP:554/stream1
        self.rtsp_url = (
            f"rtsp://{settings.CAMERA_USER}:{settings.CAMERA_PASS}@"
            f"{settings.CAMERA_IP}:{settings.RTSP_PORT}/stream1"
        )

    #connects to ONVIF and starts RTSP stream thread
    def connect(self):
        try:
            #connect ONVIF
            #adjust wsdl_dir if needed for Docker vs Local (check https://github.com/yingchengpa/python-onvif2-zeep)
            self.mycam = ONVIFCamera(
                settings.CAMERA_IP, 
                settings.ONVIF_PORT, 
                settings.CAMERA_USER, 
                settings.CAMERA_PASS
            )
            self.media = self.mycam.create_media_service()
            self.ptz = self.mycam.create_ptz_service()
            self.profile = self.media.GetProfiles()[0] #use first profile
            
            #connect RTSP
            self.cap = cv2.VideoCapture(self.rtsp_url)
            if not self.cap.isOpened():
                raise RuntimeError(f"RTSP stream not available: {self.rtsp_url}")
            
            #start BG Frame Grabber
            self.running = True
            threading.Thread(target=self._update_loop, daemon=True).start()
            
            logger.info(f"Connected to Camera Successfully at {settings.CAMERA_IP}")
            return True
            
        except Exception as e: #if it doesn't work:
            logger.error(f"Camera Connection Failed: {e}")
            return False

    #Continuously grabs frames to keep the buffer empty
    def _update_loop(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.latest_frame = frame
                    self.last_frame_time = time.time()
            else:
                # If stream drops, wait briefly before retrying
                time.sleep(0.1)

    #return most recent frame
    def get_frame(self):
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None

    #moves camera to specific onvif preset index
    def move_to_preset(self, preset_index: int):
        try:
            if not self.ptz:
                logger.warning("PTZ service not initialized")
                return

            self.ptz.GotoPreset({
                'ProfileToken': self.profile.token,
                'PresetToken': str(preset_index), 
                'Speed': {'x': 1, 'y': 1, 'z': 1} #since presets are usually strings like '1', '2' on Tapo
            })
            logger.info(f"Moving to Preset {preset_index}...")
            
            # Mechanical Wait time - adjust based on your camera speed
            time.sleep(3.0) 
            
        except Exception as e:
            logger.error(f"PTZ Move Failed: {e}") #log if exception occurs

    def release(self):
        """Cleanup resources"""
        self.running = False
        if self.cap:
            self.cap.release()
        logger.info("Camera released")