import cv2
import numpy as np
import onnxruntime as ort
import logging
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from config.settings import settings

#setup logger
logger = logging.getLogger("Eagle.Inference")

class InferenceEngine:
    def __init__(self):
        self.ae_sess = None
        self.clf_sess = None
        
        #class labels
        self.class_names = ["Corrosion", "Fire", "Human", "Leaky Pipes"]
        
        #Force CPU to avoid Mac CoreML crash (CoreMLExecutionProvider is unstable on some ONNX versions)
        self.providers = ['CPUExecutionProvider']
        
        self._load_models()

    def _load_models(self):
        logger.info(f"Forcing Inference Providers: {self.providers}")

        #load AEs
        ae_path = settings.MODELS_DIR / "autoencoder.onnx"
        if ae_path.exists():
            try:
                self.ae_sess = ort.InferenceSession(str(ae_path), providers=self.providers)
                logger.info(f"Autoencoder loaded from {ae_path}")
            except Exception as e:
                logger.error(f"Failed to load Autoencoder: {e}")
        else:
            logger.warning(f"Autoencoder not found at {ae_path}. Anomaly detection will not work.")

        #then load classifier
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
        """
        Preprocessing for Autoencoder (0-1 scaling).
        """
        # Resize to AE_IMG_SIZE (e.g., 224)
        img = cv2.resize(frame_bgr, (settings.AE_IMG_SIZE, settings.AE_IMG_SIZE))
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        # Transpose to (Channels, Height, Width) -> (3, H, W)
        img = np.transpose(img, (2, 0, 1))
        # Add Batch Dimension -> (1, 3, H, W)
        return np.expand_dims(img, axis=0)

    #classifier preprocessing
    def _preprocess_clf(self, frame_bgr):
        # Resize
        img = cv2.resize(frame_bgr, (320, 320)) # Classifier usually expects 320 or 224
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalize [0, 1]
        img = img.astype(np.float32) / 255.0
        # ImageNet Mean/Std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        
        # CHW + Batch
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0)

    def detect_anomaly(self, frame) -> float:
        """
        Runs Autoencoder and returns anomaly score.
        - MSE when TRAINING_MODE != "ssim"
        - (1 - SSIM) when TRAINING_MODE == "ssim"
        Returns 0.0 if model is not loaded.
        """
        if self.ae_sess is None:
            return 0.0

        try:
            input_name = self.ae_sess.get_inputs()[0].name
            x = self._preprocess_ae(frame)
            
            #inference
            reconstruction = self.ae_sess.run(None, {input_name: x})[0]
            
            #compute anomaly score
            if settings.TRAINING_MODE == "ssim":
                # SSIM expects HWC; convert from NCHW and compute 1 - SSIM
                x_img = np.transpose(x[0], (1, 2, 0))
                recon_img = np.transpose(reconstruction[0], (1, 2, 0))
                score = 1.0 - ssim(
                    x_img,
                    recon_img,
                    channel_axis=2,
                    data_range=1.0
                )
                return float(score)

            #default: mean squared error
            mse = np.mean((x - reconstruction) ** 2)
            return float(mse)
        except Exception as e:
            logger.error(f"Error during anomaly detection: {e}")
            return 0.0
        
    #Runs classifier and returns (Label, Confidence)
    def classify_anomaly(self, frame) -> tuple[str, float]:
        if self.clf_sess is None:
            return "Unknown", 0.0

        try:
            input_name = self.clf_sess.get_inputs()[0].name
            x = self._preprocess_clf(frame)
            
            #inference
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
