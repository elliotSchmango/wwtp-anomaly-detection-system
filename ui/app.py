import streamlit as st
import cv2
import time
import shutil
from core.state import StateManager
from services.manager import ServiceManager
from config.settings import settings

#config
st.set_page_config(
    page_title="HRSD: WWTP Anomaly Detection System",
    page_icon="⛶",
    layout="wide",
    initial_sidebar_state="expanded"
)

#for singleton architecture access
state_mgr = StateManager()
svc_mgr = ServiceManager()

def _delete_zone_data():
    removed = 0
    for zone_dir in settings.DATA_DIR.glob("zone_*"):
        if zone_dir.is_dir():
            shutil.rmtree(zone_dir, ignore_errors=True)
            removed += 1
    return removed

def _delete_model_data():
    removed = 0
    if settings.MODELS_DIR.exists():
        for path in settings.MODELS_DIR.iterdir():
            if path.is_file():
                path.unlink()
                removed += 1
    return removed

#sidebar
with st.sidebar:
    st.title("Control Panel")
    
    #dynamic captioning
    snapshot = state_mgr.get_snapshot()
    status_color = "red" if snapshot['mode'] == "ERROR" else "green" if snapshot['mode'] == "SENTRY" else "blue"
    st.markdown(f"**System Status:** :{status_color}[{snapshot['status']}]")
    
    st.divider()
    
    #sentry toggle
    is_sentry_active = (snapshot['mode'] == "SENTRY")
    toggle_sentry = st.toggle("Activate Sentry Mode", value=is_sentry_active)
    
    if toggle_sentry and not is_sentry_active:
        svc_mgr.start_sentry()
        st.rerun()
    elif not toggle_sentry and is_sentry_active:
        svc_mgr.stop_active()
        st.rerun()

    st.divider()
    
    #show available controls:
    st.subheader("Manual Operations")
    
    # Updated: use_container_width=True -> width="stretch"
    if st.button("Calibrate", disabled=is_sentry_active, width="stretch"):
        svc_mgr.start_calibration()
        st.rerun()
    
    if st.button("Train Autoencoder", disabled=is_sentry_active, width="stretch"):
        svc_mgr.start_training()
        st.rerun()
            
    if st.button("Train Classifier", disabled=is_sentry_active, width="stretch"):
        svc_mgr.start_classifier_training()
        st.rerun()

    confirm_zone_wipe = st.checkbox(
        "Confirm zone data wipe",
        key="confirm_zone_wipe",
        disabled=is_sentry_active
    )
    if st.button(
        "Wipe Zone Data",
        disabled=is_sentry_active or not confirm_zone_wipe,
        width="stretch"
    ):
        removed = _delete_zone_data()
        state_mgr.update(log=f"Wiped zone data (removed {removed} zone folder(s)).")
        st.session_state["confirm_zone_wipe"] = False
        st.rerun()

    confirm_model_wipe = st.checkbox(
        "Confirm model data wipe",
        key="confirm_model_wipe",
        disabled=is_sentry_active
    )
    if st.button(
        "Wipe Model Data",
        disabled=is_sentry_active or not confirm_model_wipe,
        width="stretch"
    ):
        removed = _delete_model_data()
        state_mgr.update(log=f"Wiped model data (removed {removed} file(s)).")
        st.session_state["confirm_model_wipe"] = False
        st.rerun()
            
    if st.button("STOP ALL TASKS", type="primary", width="stretch"):
        svc_mgr.stop_active()
        st.rerun()

#dashboard
col_main, col_stats = st.columns([2, 1])

with col_main:
    st.subheader(f"Live View - Zone {snapshot['current_zone']}")
    
    #display frame
    frame = snapshot.get('latest_frame')
    if frame is not None:
        #since opencv is BGR, streamlit also expects RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Updated: use_container_width=True -> width="stretch"
        st.image(frame_rgb, channels="RGB", width="stretch")
    else:
        st.info("Waiting for camera feed...")

with col_stats:
    st.subheader("Telemetry")

    score_mode = settings.TRAINING_MODE.lower()
    current_score_label = "Current SSIM" if score_mode == "ssim" else "Current MSE"
    current_threshold = (
        settings.AE_THRESHOLD_SSIM if score_mode == "ssim" else settings.AE_THRESHOLD_MSE
    )
    
    #alerty if anomaly detected
    if snapshot['is_anomaly']:
        st.error(f"ANOMALY DETECTED: {snapshot['last_anomaly_label']}")
        st.metric("Confidence Score (MSE)", f"{snapshot['last_anomaly_score']:.5f}")
    else:
        st.success("System Normal")
        st.metric(current_score_label, f"{snapshot['last_anomaly_score']:.5f}")
        st.caption(f"Threshold: {current_threshold:.5f}")

    st.divider()
    
    #logging
    st.subheader("System Logs")
    log_text = "\n".join(snapshot['logs'][-10:]) if snapshot['logs'] else "No logs yet."
    st.text_area("Console Output", log_text, height=200, disabled=True)
    
    #progress bar (for Calibration/Training)
    if snapshot['mode'] in ["CALIBRATING", "TRAINING", "TRAINING_CLF"]:
        st.progress(snapshot['progress'] / 100)

#auto-refresh: run script every 1 second to update UI
time.sleep(1)
st.rerun()
