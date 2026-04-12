import cv2
import numpy as np
import onnxruntime as ort
import logging
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from config.settings import settings

logger = logging.getLogger("Eagle.Inference")

class EMANormalizer:
    #soft-max peak tracker: normalizes raw component scores to [0,1]
    #peak decays slowly via EMA but immediately snaps up on new highs
    def __init__(self, decay=0.995, init_max=1.0):
        self.decay = decay
        self.peak = float(init_max)

    def normalize(self, value):
        self.peak = max(self.decay * self.peak, float(value))
        self.peak = max(self.peak, 1e-8) #prevent division by zero
        return float(np.clip(value / self.peak, 0.0, 1.0))


class InferenceEngine:
    def __init__(self):
        self.ae_sess = None
        self.clf_sess = None

        #class labels
        self.class_names = ["Corrosion", "Fire", "Human", "Leaky Pipes"]

        #force CPU to avoid Mac CoreML crash (CoreMLExecutionProvider is unstable on some ONNX versions)
        self.providers = ['CPUExecutionProvider']

        #per-zone EMA normalizers for FUSED components: {zone_id: {component: EMANormalizer}}
        self._normalizers = {}

        #per-zone mean latent vectors for L component: {zone_id: ndarray}
        self._zone_ref_latents = {}

        self._load_models()

    def _load_models(self):
        logger.info(f"Forcing Inference Providers: {self.providers}")

        #load autoencoder
        ae_path = settings.MODELS_DIR / "autoencoder.onnx"
        if ae_path.exists():
            try:
                self.ae_sess = ort.InferenceSession(str(ae_path), providers=self.providers)
                outputs = [o.name for o in self.ae_sess.get_outputs()]
                logger.info(f"Autoencoder loaded from {ae_path} (outputs: {outputs})")
            except Exception as e:
                logger.error(f"Failed to load Autoencoder: {e}")
        else:
            logger.warning(f"Autoencoder not found at {ae_path}. Anomaly detection will not work.")

        #load classifier
        clf_path = settings.MODELS_DIR / "classifier.onnx"
        if clf_path.exists():
            try:
                self.clf_sess = ort.InferenceSession(str(clf_path), providers=self.providers)
                logger.info(f"Classifier loaded from {clf_path}")
            except Exception as e:
                logger.error(f"Failed to load Classifier: {e}")
        else:
            logger.warning(f"Classifier not found at {clf_path}.")

    def _preprocess_ae(self, frame_bgr):
        #resize, convert BGR->RGB, normalize [0,1], transpose to NCHW
        img = cv2.resize(frame_bgr, (settings.AE_IMG_SIZE, settings.AE_IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0)

    def _preprocess_clf(self, frame_bgr):
        #resize, apply ImageNet mean/std normalization, transpose to NCHW
        img = cv2.resize(frame_bgr, (320, 320))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0)

    def _gradient_map(self, img_hwc):
        #Sobel edge magnitude map, normalized to [0,1]
        gray = (np.mean(img_hwc, axis=2) * 255).astype(np.uint8)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        #max theoretical Sobel magnitude for 8-bit input is ~1024
        return np.sqrt(gx ** 2 + gy ** 2) / 1024.0

    def _fft_energy(self, img_hwc):
        #log-magnitude 2D FFT of grayscale image, masked to mid+high spatial frequency band
        gray = np.mean(img_hwc, axis=2).astype(np.float32)
        f = np.fft.fft2(gray)
        magnitude = np.log1p(np.abs(np.fft.fftshift(f)))
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
        #exclude inner 10% (DC lobe + very low freqs like broad illumination gradients)
        r_min = int(min(cy, cx) * 0.10)
        mask = dist >= r_min
        return magnitude, mask

    def _get_normalizers(self, zone_id):
        #retrieve or create per-zone EMA normalizers for each FUSED component
        if zone_id not in self._normalizers:
            self._normalizers[zone_id] = {
                's': EMANormalizer(decay=0.995, init_max=0.50), #SSIM residual
                'g': EMANormalizer(decay=0.995, init_max=0.30), #gradient divergence
                'f': EMANormalizer(decay=0.995, init_max=2.00), #FFT energy shift
                'l': EMANormalizer(decay=0.995, init_max=0.50), #latent cosine dist
            }
        return self._normalizers[zone_id]

    def precompute_reference_latents(self, zone_id, zone_dir):
        #build mean latent vector from calibration images; enables L component of FUSED
        if self.ae_sess is None:
            return
        output_names = [o.name for o in self.ae_sess.get_outputs()]
        if "latent" not in output_names:
            logger.info("Model lacks 'latent' output — L component disabled. Retrain with TRAINING_MODE=fused to enable.")
            return

        input_name = self.ae_sess.get_inputs()[0].name
        latents = []
        for img_path in Path(zone_dir).glob("*.jpg"):
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            x = self._preprocess_ae(frame)
            result = self.ae_sess.run(["latent"], {input_name: x})
            latents.append(result[0].flatten())

        if latents:
            self._zone_ref_latents[zone_id] = np.mean(np.stack(latents), axis=0)
            logger.info(f"Reference latent computed for zone {zone_id} from {len(latents)} calibration frames")
        else:
            logger.warning(f"No calibration images found for zone {zone_id} — L component disabled for this zone")

    def detect_anomaly(self, frame, zone_id=None) -> float:
        #dispatch to FUSED or legacy scorer depending on TRAINING_MODE
        if settings.TRAINING_MODE == "fused":
            score, _ = self.detect_anomaly_fused(frame, zone_id=zone_id)
            return score

        if self.ae_sess is None:
            return 0.0

        try:
            input_name = self.ae_sess.get_inputs()[0].name
            x = self._preprocess_ae(frame)
            reconstruction = self.ae_sess.run(None, {input_name: x})[0]

            if settings.TRAINING_MODE == "ssim":
                x_img = np.transpose(x[0], (1, 2, 0))
                recon_img = np.transpose(reconstruction[0], (1, 2, 0))
                score = 1.0 - ssim(x_img, recon_img, channel_axis=2, data_range=1.0)
                return float(score)

            #default: MSE
            mse = np.mean((x - reconstruction) ** 2)
            return float(mse)
        except Exception as e:
            logger.error(f"Error during anomaly detection: {e}")
            return 0.0

    def detect_anomaly_fused(self, frame, zone_id=None) -> tuple[float, dict]:
        #FUSED: weighted composite of 4 visual anomaly signals
        #  S (0.30) — SSIM residual: structural + luminance deviation
        #  G (0.25) — Sobel gradient divergence: edge/silhouette changes
        #  F (0.25) — FFT energy shift: surface texture / corrosion changes
        #  L (0.20) — latent cosine distance: semantic scene deviation
        if self.ae_sess is None:
            return 0.0, {}

        z_key = zone_id if zone_id is not None else 0
        norms = self._get_normalizers(z_key)

        try:
            input_name = self.ae_sess.get_inputs()[0].name
            x = self._preprocess_ae(frame) #shape: (1, 3, H, W)

            #single forward pass — get reconstruction and latent (if exported)
            outputs = self.ae_sess.run(None, {input_name: x})
            reconstruction = outputs[0]
            latent_vec = outputs[1].flatten() if len(outputs) > 1 else None

            #convert NCHW tensors to HWC for spatial metric computation
            x_img = np.transpose(x[0], (1, 2, 0))
            recon_img = np.transpose(reconstruction[0], (1, 2, 0))

            #--- S: SSIM residual ---
            s_raw = float(1.0 - ssim(x_img, recon_img, channel_axis=2, data_range=1.0))

            #--- G: Sobel gradient magnitude divergence ---
            grad_x = self._gradient_map(x_img)
            grad_r = self._gradient_map(recon_img)
            g_raw = float(np.mean(np.abs(grad_x - grad_r)))

            #--- F: 2D FFT energy shift in mid+high spatial frequencies ---
            mag_x, freq_mask = self._fft_energy(x_img)
            mag_r, _ = self._fft_energy(recon_img)
            f_raw = float(np.mean(np.abs(mag_x[freq_mask] - mag_r[freq_mask])))

            #--- L: cosine distance from calibration latent centroid ---
            l_raw = 0.0
            if latent_vec is not None and z_key in self._zone_ref_latents:
                ref_z = self._zone_ref_latents[z_key]
                cos_sim = np.dot(latent_vec, ref_z) / (
                    np.linalg.norm(latent_vec) * np.linalg.norm(ref_z) + 1e-8
                )
                l_raw = float(np.clip(1.0 - cos_sim, 0.0, 1.0))

            #normalize each component to [0,1] via per-zone EMA peak tracker
            s = norms['s'].normalize(s_raw)
            g = norms['g'].normalize(g_raw)
            f = norms['f'].normalize(f_raw)
            l = norms['l'].normalize(l_raw)

            #weighted composite score
            score = (0.30 * s) + (0.25 * g) + (0.25 * f) + (0.20 * l)

            components = {
                "S": round(s_raw, 5),
                "G": round(g_raw, 5),
                "F": round(f_raw, 5),
                "L": round(l_raw, 5),
                "fused": round(score, 5),
            }
            return float(score), components

        except Exception as e:
            logger.error(f"Error during FUSED anomaly detection: {e}")
            return 0.0, {}

    def classify_anomaly(self, frame) -> tuple[str, float]:
        #runs classifier, returns (Label, Confidence)
        if self.clf_sess is None:
            return "Unknown", 0.0

        try:
            input_name = self.clf_sess.get_inputs()[0].name
            x = self._preprocess_clf(frame)
            logits = self.clf_sess.run(None, {input_name: x})[0]

            #softmax to get probabilities
            probs = np.exp(logits) / np.sum(np.exp(logits), axis=1)
            idx = np.argmax(probs)
            confidence = float(probs[0][idx])

            if 0 <= idx < len(self.class_names):
                return self.class_names[idx], confidence

            return "Unknown", confidence
        except Exception as e:
            logger.error(f"Error during classification: {e}")
            return "Error", 0.0
