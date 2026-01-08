import cv2
import logging
import sys
import os
from pathlib import Path
from datetime import datetime

current_file = Path(__file__).resolve()
project_root = current_file.parent

if (project_root / "core").exists():
    pass 
elif (project_root.parent / "core").exists():
    project_root = project_root.parent
    
sys.path.append(str(project_root))

from core.inference import InferenceEngine
from config.settings import settings

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("TestPipeline")

#CONFIG
DATA_DIR = project_root / "data"
TEST_ANOMALY_DIR = DATA_DIR / "test_anomaly"
TEST_NOT_ANOMALY_DIR = DATA_DIR / "test_not_anomaly"

def find_test_file(target_name):
    """
    Looks for the matching anomaly file in the test folders.
    """
    #check test_anomaly
    candidate = TEST_ANOMALY_DIR / target_name
    if candidate.exists(): return candidate
    #then test_not_anomaly
    candidate = TEST_NOT_ANOMALY_DIR / target_name
    if candidate.exists(): return candidate
    
    return None

def run_zone_based_test():
    print("\n" + "="*140)
    print(f"   ZONE-BASED ANOMALY TEST  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*140)
    
    #look for pngs
    zone_files = sorted(list(DATA_DIR.glob("zone_*/*.png")))
    
    if not zone_files:
        logger.error(f"No training images found in {DATA_DIR}/zone_*")
        return

    #start inference (test version)
    try:
        ai = InferenceEngine()
        if ai.ae_sess is None:
            logger.error("CRITICAL: Autoencoder model did not load. Scores will be 0.0.")
            return
    except Exception as e:
        logger.error(f"Failed to load Inference Engine: {e}")
        return

    threshold = settings.AE_THRESHOLD_SSIM if settings.TRAINING_MODE == "ssim" else settings.AE_THRESHOLD_MSE
    logger.info(f"Model: {settings.TRAINING_MODE.upper()} | Threshold: {threshold:.5f} | Zone Masters: {len(zone_files)}")
    
    print("-" * 140)
    print(f"{'Zone Master (Train)':<25} | {'Test Candidate':<25} | {'Expectation':<12} | {'Master Err':<10} | {'Test Err':<10} | {'Gap':<9} | {'Trigger?':<8} | {'Result'}")
    print("-" * 140)

    passed = 0
    total = 0
    failed_cases = []

    for master_path in zone_files:
        #if filename has '*', we expect NO FLAG, else, we expect FLAG.
        has_marker = "*" in master_path.name
        
        if has_marker:
            expected_behavior = "NO FLAG"
            file_id = master_path.name.replace("_normal*.png", "")
            target_test_name = f"{file_id}_anomaly*.png"
        else:
            expected_behavior = "FLAG"
            file_id = master_path.name.replace("_normal.png", "")
            target_test_name = f"{file_id}_anomaly.png"

        # Find the matching test file
        test_path = find_test_file(target_test_name)
        
        # Fallback: Try alternative name if specific asterisk file not found
        if not test_path:
            alt_name = f"{file_id}_anomaly.png"
            test_path = find_test_file(alt_name)
            
        if not test_path:
            # We don't fail, we just skip because maybe we don't have a test case for this zone yet
            continue

        # Load & Test
        img_master = cv2.imread(str(master_path))
        img_test = cv2.imread(str(test_path))
        
        if img_master is None or img_test is None: continue

        score_master = ai.detect_anomaly(img_master)
        score_test = ai.detect_anomaly(img_test)
        
        #eval logic
        triggered = "YES" if score_test > threshold else "NO"
        if expected_behavior == "FLAG":
            is_success = (triggered == "YES")
        else:
            is_success = (triggered == "NO")

        status = "PASS" if is_success else "FAIL"
        gap = score_test - score_master
        
        #formatting for clarity
        m_name = f"{master_path.parent.name}/{master_path.name}"
        m_name = (m_name[:22] + '..') if len(m_name) > 22 else m_name
        
        t_name = (test_path.name[:22] + '..') if len(test_path.name) > 22 else test_path.name

        print(f"{m_name:<25} | {t_name:<25} | {expected_behavior:<12} | {score_master:.5f}    | {score_test:.5f}    | {gap:+.5f}  | {triggered:<8} | {status}")

        if is_success: 
            passed += 1
        else:
            failed_cases.append(f"{m_name} -> {t_name} (Got {triggered}, Expected {expected_behavior})")
        
        total += 1

    #SUMMARY
    print("-" * 140)
    if total > 0:
        acc = (passed / total) * 100
        print(f"SUMMARY: {passed}/{total} Passed ({acc:.1f}%)")
        
        if failed_cases:
            print("\n[!] FAILURES DETECTED:")
            for fc in failed_cases:
                print(f"    - {fc}")
    else:
        logger.info("\nNo matching test pairs found for any zone images.")
    print("="*140 + "\n")

if __name__ == "__main__":
    run_zone_based_test()