import collections
import time
import base64
import cv2
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from core.camera import CameraClient
from core.inference import InferenceEngine
from core.state import StateManager
from core.llm import VisionAgent
from core.alignment import FrameStabilizer
from cloud.telemetry import TelemetrySender
from config.settings import settings

logger = logging.getLogger("Eagle.Sentry")

class SentryLoop:
    def __init__(self, stop_event):
        self.stop_event = stop_event
        self.state = StateManager()
        self.telemetry = TelemetrySender()
        self.llm = VisionAgent()
        #median calibration reference frames per zone: {zone_id: ndarray}
        self._zone_ref_frames = {}
        #rolling anomaly hit buffers per zone: {zone_id: deque[bool]}
        self._temporal_buffers = {}

    def _precompute_median_reference(self, zone_id, zone_dir):
        #pixel-wise median across all calibration frames for a stable, noise-free reference
        frames = []
        for img_path in sorted(Path(zone_dir).glob("*.jpg")):
            frame = cv2.imread(str(img_path))
            if frame is not None:
                frames.append(frame.astype(np.float32))
        if frames:
            median_frame = np.median(np.stack(frames), axis=0).astype(np.uint8)
            self._zone_ref_frames[zone_id] = median_frame
            logger.info(f"Median reference computed for zone {zone_id} from {len(frames)} frames")
        else:
            logger.warning(f"No calibration images for zone {zone_id} — Gemini reference unavailable")

    def _get_temporal_buffer(self, zone_id):
        #return or create the rolling anomaly-hit deque for a zone
        if zone_id not in self._temporal_buffers:
            self._temporal_buffers[zone_id] = collections.deque(maxlen=settings.TEMPORAL_WINDOW)
        return self._temporal_buffers[zone_id]

    def _encode_frame(self, frame):
        fmt = settings.TELEMETRY_IMAGE_FORMAT.lower()
        if fmt not in ("jpg", "jpeg", "png"):
            fmt = "jpg"

        resized = frame
        max_width = settings.TELEMETRY_IMAGE_MAX_WIDTH
        if max_width and frame.shape[1] > max_width:
            scale = max_width / frame.shape[1]
            new_size = (max_width, int(frame.shape[0] * scale))
            resized = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

        ext = ".jpg" if fmt in ("jpg", "jpeg") else ".png"
        params = []
        if ext == ".jpg":
            params = [cv2.IMWRITE_JPEG_QUALITY, settings.TELEMETRY_IMAGE_QUALITY]

        ok, buffer = cv2.imencode(ext, resized, params)
        if not ok:
            return None

        return {
            "format": fmt,
            "data_b64": base64.b64encode(buffer).decode("ascii")
        }

    def run(self):
        camera = CameraClient()
        if not camera.connect():
            self.state.update(status="Camera Connection Failed", mode="ERROR")
            return

        ai = InferenceEngine()

        #precompute per-zone assets at startup before entering the main loop
        for z in settings.ZONES:
            zone_dir = settings.DATA_DIR / f"zone_{z}"
            #1: median reference frame for confident Gemini analysis
            self._precompute_median_reference(z, zone_dir)
            #2: latent centroid for FUSED L component (no-op if model lacks latent output)
            if settings.TRAINING_MODE == "fused":
                ai.precompute_reference_latents(z, zone_dir)

        #determine mode & threshold
        if settings.TRAINING_MODE == "fused":
            metric_label = "FUSED"
            current_threshold = settings.AE_THRESHOLD_FUSED
        elif settings.TRAINING_MODE == "ssim":
            metric_label = "SSIM Loss"
            current_threshold = settings.AE_THRESHOLD_SSIM
        else:
            metric_label = "MSE"
            current_threshold = settings.AE_THRESHOLD_MSE

        try:
            logger.info(f"Sentry Mode Started ({metric_label} | threshold:{current_threshold} | window:{settings.TEMPORAL_WINDOW} hits:{settings.TEMPORAL_MIN_HITS})")
            while not self.stop_event.is_set():

                for zone in settings.ZONES:
                    if self.stop_event.is_set(): break

                    #move camera to zone
                    self.state.update(current_zone=zone, status=f"Scanning Zone {zone}...")
                    camera.move_to_preset(zone)

                    #start tracking total pipeline latency
                    t_pipeline_start = time.perf_counter()

                    #wait for movement to settle, then grab frame
                    time.sleep(0.5)
                    t_cam_start = time.perf_counter()
                    frame = camera.get_frame()
                    t_cam = time.perf_counter() - t_cam_start

                    if frame is None:
                        logger.warning("Empty frame received")
                        continue

                    #update UI live feed
                    self.state.update(latest_frame=frame)

                    #get reference frame for alignment
                    ref_frame = self._zone_ref_frames.get(zone)

                    #stabilize frame (align to reference and crop borders)
                    t_align_start = time.perf_counter()
                    if ref_frame is not None:
                        frame, ref_frame = FrameStabilizer.stabilize(frame, ref_frame)
                    else:
                        frame, _ = FrameStabilizer.stabilize(frame, None)
                    t_align = time.perf_counter() - t_align_start

                    #compute anomaly score via FUSED or legacy metric
                    t_scorer_start = time.perf_counter()
                    if settings.TRAINING_MODE == "fused":
                        anomaly_score, fused_components = ai.detect_anomaly_fused(frame, zone_id=zone)
                    else:
                        anomaly_score = ai.detect_anomaly(frame, zone_id=zone)
                        fused_components = {}
                    t_scorer = time.perf_counter() - t_scorer_start

                    #4: temporal consistency — require TEMPORAL_MIN_HITS of last TEMPORAL_WINDOW frames
                    buf = self._get_temporal_buffer(zone)
                    buf.append(anomaly_score > current_threshold)
                    is_anomaly = sum(buf) >= settings.TEMPORAL_MIN_HITS

                    label = "Normal"
                    llm_output = "LLM Skipped"
                    llm_text_input = self.llm.get_prompt()
                    t_clf = 0.0
                    t_vlm = 0.0

                    if is_anomaly:
                        #route anomaly to classifier or VLM based on confidence
                        if settings.USE_GEMINI:
                            if ref_frame is not None:
                                t_clf_start = time.perf_counter()
                                clf_label, clf_conf = ai.classify_anomaly(frame)
                                t_clf = time.perf_counter() - t_clf_start
                                
                                if clf_conf >= settings.CLASSIFIER_CONFIDENCE_GATE:
                                    label = clf_label
                                    llm_output = f"Classifier (conf:{clf_conf:.2f}, VLM skipped)"
                                    logger.info(f"High-confidence classifier: {label} ({clf_conf:.2f})")
                                else:
                                    #escalate to VLM API with median reference
                                    t_vlm_start = time.perf_counter()
                                    llm_output = self.llm.analyze(ref_frame, frame)
                                    t_vlm = time.perf_counter() - t_vlm_start
                                    label = llm_output
                            else:
                                #handle missing calibration data
                                llm_output = "No Reference Data"
                                label = llm_output
                        else:
                            #use local classifier only since VLM disabled
                            t_clf_start = time.perf_counter()
                            label, _ = ai.classify_anomaly(frame)
                            t_clf = time.perf_counter() - t_clf_start
                            llm_output = "LLM Disabled"

                        timestamp = datetime.now().isoformat()
                        if fused_components:
                            log_msg = (
                                f"ALERT Zone {zone}: {label} "
                                f"(FUSED:{anomaly_score:.4f} "
                                f"S:{fused_components.get('S',0):.4f} "
                                f"G:{fused_components.get('G',0):.4f} "
                                f"F:{fused_components.get('F',0):.4f} "
                                f"L:{fused_components.get('L',0):.4f})"
                            )
                        else:
                            log_msg = f"ALERT Zone {zone}: {label} ({metric_label}: {anomaly_score:.5f})"
                        logger.warning(log_msg)

                        #upload telemetry to Azure
                        t_io_start = time.perf_counter()
                        encoded_current = self._encode_frame(frame)
                        encoded_ref = self._encode_frame(ref_frame) if ref_frame is not None else None
                        payload = {
                            "timestamp": timestamp,
                            "device_id": settings.DEVICE_ID,
                            "zone_id": zone,
                            "anomaly_metric": settings.TRAINING_MODE,
                            "anomaly_score": round(anomaly_score, 6),
                            "anomaly_flag": True,
                            "classifier_label": label,
                            "frame": encoded_current,
                            "llm_base_frame": encoded_ref,
                            "llm_current_frame": encoded_current,
                            "llm_text_input": llm_text_input,
                            "llm_output": llm_output
                        }
                        self.telemetry.send_payload(payload)
                        t_io = time.perf_counter() - t_io_start

                        #update UI state
                        self.state.update(
                            is_anomaly=True,
                            last_anomaly_label=label,
                            last_anomaly_score=anomaly_score,
                            last_detection_time=timestamp,
                            log=log_msg
                        )

                        #save snapshot locally
                        snap_name = f"ALERT_{timestamp}_Z{zone}_{label}.jpg".replace(":", "-")
                        cv2.imwrite(str(settings.DATA_DIR / snap_name), frame)

                    else:
                        if fused_components:
                            clear_msg = (
                                f"Zone {zone} Clear "
                                f"(FUSED:{anomaly_score:.4f} "
                                f"S:{fused_components.get('S',0):.4f} "
                                f"G:{fused_components.get('G',0):.4f} "
                                f"F:{fused_components.get('F',0):.4f} "
                                f"L:{fused_components.get('L',0):.4f})"
                            )
                        else:
                            clear_msg = f"Zone {zone} Clear ({metric_label}: {anomaly_score:.5f})"
                        self.state.update(
                            is_anomaly=False,
                            status=clear_msg
                        )
                        t_io = 0.0

                    t_pipeline = time.perf_counter() - t_pipeline_start
                    
                    #log latency metrics for streamline testing
                    logger.debug(f"Metrics [Z{zone}] | Cam:{t_cam:.3f}s | Align:{t_align:.3f}s | Scorer:{t_scorer:.3f}s | Clf:{t_clf:.3f}s | VLM:{t_vlm:.3f}s | IO:{t_io:.3f}s | Total:{t_pipeline:.3f}s")

                    time.sleep(1.0 / settings.SENTRY_FPS)

        except Exception as e:
            logger.error(f"Sentry Crash: {e}")
            self.state.update(status=f"Sentry Error: {str(e)}", mode="ERROR")
        finally:
            camera.release()
            self.telemetry.disconnect()
            logger.info("Sentry Mode Stopped")

            #reset to IDLE only if we weren't interrupted by another task start
            if self.stop_event.is_set():
                self.state.set_mode("IDLE")
