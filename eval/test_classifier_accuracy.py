import cv2
import time
import logging
import sys
from pathlib import Path
import numpy as np

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

from core.inference import InferenceEngine
from core.llm import VisionAgent
from config.settings import settings

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("TestClassifier")

def normalize_label(label: str) -> str:
    label = label.lower().strip()
    if label in ["leaky pipes", "water leak", "leak"]:
        return "water leak"
    if label == "fire": return "fire"
    if label == "smoke": return "smoke"
    if label == "human": return "human"
    if label == "corrosion": return "corrosion"
    if label == "foreign object": return "foreign object"
    if label == "chemical spill": return "chemical spill"
    return "unknown"

def run_offline_accuracy_test(use_vlm=True):
    print("\n" + "="*140)
    print("   OFFLINE CLASSIFICATION ACCURACY SIMULATOR")
    print("="*140)
    
    test_dir = project_root / "data" / "classifier_data" / "test_realworld"
    
    if not test_dir.exists():
        logger.error(f"Test directory not found: {test_dir}")
        return

    ai = InferenceEngine()
    vlm = VisionAgent() if use_vlm else None
    
    classes = [d for d in test_dir.iterdir() if d.is_dir()]
    
    total_images = 0
    clf_correct = 0
    vlm_correct = 0
    
    clf_latencies = []
    vlm_latencies = []
    
    print(f"{'File':<30} | {'Ground Truth':<15} | {'Clf Pred':<15} | {'Clf (s)':<10} | {'VLM Pred':<25} | {'VLM (s)'}")
    print("-" * 140)

    for cls_dir in classes:
        ground_truth_raw = cls_dir.name
        gt = normalize_label(ground_truth_raw)
        
        for img_path in cls_dir.glob("*.*"):
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
                
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
                
            total_images += 1
            file_name = img_path.name[:27] + "..." if len(img_path.name) > 30 else img_path.name
            
            #test local classifier efficiency & accuracy
            t0 = time.perf_counter()
            clf_label, clf_conf = ai.classify_anomaly(frame)
            t_clf = time.perf_counter() - t0
            clf_latencies.append(t_clf)
            
            clf_str = normalize_label(clf_label)
            if clf_str == gt or (gt == "water leak" and clf_str == "leaky pipes"):
                clf_correct += 1
                clf_disp = f"{clf_label} (PASS)"
            else:
                clf_disp = f"{clf_label} (FAIL)"

            #test vlm logic via a blank frame to force reference logic
            vlm_disp = "N/A"
            t_vlm = 0.0
            if use_vlm:
                blank_ref = np.zeros_like(frame)
                t0 = time.perf_counter()
                vlm_raw = vlm.analyze(blank_ref, frame)
                t_vlm = time.perf_counter() - t0
                vlm_latencies.append(t_vlm)
                
                vlm_str = normalize_label(vlm_raw)
                # handle "detailed" mode output formatting implicitly
                if "detailed:" in vlm_raw.lower():
                    cat_part = vlm_raw.split("(")[0].strip()
                    vlm_str = normalize_label(cat_part)
                
                if vlm_str == gt:
                    vlm_correct += 1
                    vlm_disp = f"{vlm_str} (PASS)"
                else:
                    vlm_disp = f"{vlm_str} (FAIL)"
                    
            print(f"{file_name:<30} | {gt:<15} | {clf_disp:<15} | {t_clf:.3f}      | {vlm_disp:<25} | {t_vlm:.3f}")

    print("-" * 140)
    print("SUMMARY")
    print("-" * 140)
    if total_images == 0:
        print("No images found to process.")
        return
        
    print(f"Total Samples Tested: {total_images}")
    print(f"Local Classifier Acc: {clf_correct / total_images * 100:.1f}% ({clf_correct}/{total_images})")
    print(f"Local Clf Latency:    {np.mean(clf_latencies):.4f}s avg")
    
    if use_vlm:
        print(f"Vision Agent Acc:     {vlm_correct / total_images * 100:.1f}% ({vlm_correct}/{total_images})")
        print(f"Vision Agent Latency: {np.mean(vlm_latencies):.4f}s avg")

if __name__ == "__main__":
    #To skip VLM testing, change use_vlm to False 
    run_offline_accuracy_test(use_vlm=True)
