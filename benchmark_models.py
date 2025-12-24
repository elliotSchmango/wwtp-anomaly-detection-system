import os
import glob
import cv2
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from skimage.metrics import structural_similarity as ssim
from pathlib import Path
from config import settings

BASELINE_MODEL_PATH = "models/autoencoder_mse.onnx"
NOVEL_MODEL_PATH = "models/autoencoder_ssim.onnx"

DATA_DIR = Path("data")
IMG_SIZE = settings.AE_IMG_SIZE

def load_images_from_folder(pattern, label, max_images=100):
    """
    Loads images from a glob pattern, resizes, normalizes, and assigns labels.
    Label 0 = Normal
    Label 1 = Anomaly
    """
    images = []
    labels = []
    
    # Recursively find images
    files = glob.glob(str(pattern), recursive=True)
    np.random.shuffle(files)
    
    print(f"Loading {min(len(files), max_images)} images from {pattern}...")
    
    for f in files[:max_images]:
        try:
            img = cv2.imread(f)
            if img is None: continue
            
            # Preprocessing to match PyTorch training (Resize -> RGB -> Normalize)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            
            # Convert to NCHW format (Batch, Channel, Height, Width) for ONNX
            img = np.transpose(img, (2, 0, 1)) 
            img = np.expand_dims(img, axis=0)
            
            images.append(img)
            labels.append(label)
        except Exception as e:
            print(f"Error loading {f}: {e}")
            
    return images, labels

def get_reconstruction(session, img_batch):
    """Runs inference on a single image batch"""
    input_name = session.get_inputs()[0].name
    # ONNX Runtime returns a list, we want the first output (the image)
    return session.run(None, {input_name: img_batch})[0]

def calculate_scores(session, images, metric="mse"):
    """
    Calculates Anomaly Scores for a list of images.
    MSE Mode: Higher Mean Squared Error = Anomaly
    SSIM Mode: Higher Dissimilarity (1 - SSIM) = Anomaly
    """
    scores = []
    for img in images:
        try:
            recon = get_reconstruction(session, img)
            
            if metric == "mse":
                # Standard MSE calculation
                # Flatten arrays to 1D to compare pixel-by-pixel
                flat_in = img.flatten()
                flat_out = recon.flatten()
                score = np.mean((flat_in - flat_out) ** 2)
                
            elif metric == "ssim":
                # SSIM requires HWC format (Height, Width, Channel)
                # We transpose (1, 3, 224, 224) -> (224, 224, 3)
                img_hwc = np.transpose(img[0], (1, 2, 0))
                recon_hwc = np.transpose(recon[0], (1, 2, 0))
                
                # Calculate SSIM
                # channel_axis=2 tells skimage it's RGB
                # data_range=1.0 tells skimage values are 0.0-1.0
                score_val = ssim(img_hwc, recon_hwc, channel_axis=2, data_range=1.0)
                
                # We want an Anomaly Score (Higher is worse)
                # SSIM is 1.0 for perfect match, so we invert it
                score = 1.0 - score_val 
                
            scores.append(score)
        except Exception as e:
            print(f"Error calculating score: {e}")
            scores.append(0.0)
            
    return scores

def main():
    print("--- 1. Loading Data ---")
    
    # A. Load Normals (from Calibration)
    # These represent the "Safe" baseline
    normal_imgs, normal_labels = load_images_from_folder(DATA_DIR / "zone_*" / "*.jpg", 0, max_images=100)
    
    # B. Load Anomalies
    # Priority: Physical Staged Anomalies -> Fallback: Stock Training Data
    anomaly_path = DATA_DIR / "test_anomalies" / "**" / "*.jpg"
    
    # Check if user created the staged folder
    if not list(glob.glob(str(anomaly_path))):
        print("⚠️ 'data/test_anomalies' is empty/missing.")
        print("   Falling back to 'data/classifier_data' (Note: This is less scientifically rigorous).")
        anomaly_path = DATA_DIR / "classifier_data" / "**" / "*.jpg"

    anomaly_imgs, anomaly_labels = load_images_from_folder(anomaly_path, 1, max_images=100)
    
    if not normal_imgs or not anomaly_imgs:
        print("❌ Error: Not enough data. Need at least 1 Normal and 1 Anomaly image.")
        return

    # Combine datasets
    all_images = normal_imgs + anomaly_imgs
    y_true = normal_labels + anomaly_labels
    
    print(f"\n📊 Dataset Prepared: {len(normal_imgs)} Normals vs {len(anomaly_imgs)} Anomalies")

    # Setup Plot
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess (AUC=0.5)')

    # --- 2. Evaluate Baseline (MSE) ---
    print("\n--- 2. Evaluating Baseline (MSE Model) ---")
    if os.path.exists(BASELINE_MODEL_PATH):
        sess_base = ort.InferenceSession(BASELINE_MODEL_PATH)
        y_scores_base = calculate_scores(sess_base, all_images, metric="mse")
        
        # Calculate ROC Metrics
        fpr_base, tpr_base, _ = roc_curve(y_true, y_scores_base)
        auc_base = auc(fpr_base, tpr_base)
        
        print(f"✅ Baseline Result: AUROC = {auc_base:.4f}")
        plt.plot(fpr_base, tpr_base, color='darkorange', lw=2, label=f'Baseline (MSE) AUC = {auc_base:.3f}')
    else:
        print(f"⚠️ {BASELINE_MODEL_PATH} not found. Skipping.")

    # --- 3. Evaluate Novel (SSIM) ---
    print("\n--- 3. Evaluating Novel (SSIM Model) ---")
    if os.path.exists(NOVEL_MODEL_PATH):
        sess_novel = ort.InferenceSession(NOVEL_MODEL_PATH)
        y_scores_novel = calculate_scores(sess_novel, all_images, metric="ssim")
        
        # Calculate ROC Metrics
        fpr_novel, tpr_novel, _ = roc_curve(y_true, y_scores_novel)
        auc_novel = auc(fpr_novel, tpr_novel)
        
        print(f"✅ Novel Result:    AUROC = {auc_novel:.4f}")
        plt.plot(fpr_novel, tpr_novel, color='green', lw=2, label=f'Novel (SSIM) AUC = {auc_novel:.3f}')
    else:
        print(f"⚠️ {NOVEL_MODEL_PATH} not found. Skipping.")

    # --- 4. Finalize Plot ---
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (False Alarms)')
    plt.ylabel('True Positive Rate (Detection Success)')
    plt.title('Performance Comparison: MSE vs SSIM Anomaly Detection')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    output_file = "evaluation_roc_curve.png"
    plt.savefig(output_file)
    print(f"\n📈 Chart saved successfully to {output_file}")

if __name__ == "__main__":
    main()