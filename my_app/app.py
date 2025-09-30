import streamlit as st
import numpy as np
import cv2
import joblib
import pandas as pd
import tensorflow as tf
import os
from datetime import datetime
from tensorflow.keras.models import load_model

# ---- Page Config ----
st.set_page_config(page_title="AI TraceFinder", page_icon="🔍", layout="wide")

# --- FILE PATH FIX FOR LOGO AND MODELS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(SCRIPT_DIR, "assets", "logo.png")
# You should also use os.path.join for your model and log paths for robustness
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")

# ---- Sidebar ----
# Use the corrected path variable
st.sidebar.image(LOGO_PATH, use_container_width=True)
st.sidebar.title("⚙️ Options")
st.sidebar.success("Choose task and upload an image 🚀")

# ---- Task Selector ----
task = st.sidebar.radio("🎯 Select Objective", ["Scanner Identification", "Forgery Detection"])

st.sidebar.markdown("---")
st.sidebar.write("👨‍💻 Developed by: **Yash Kodgirwar**")

# ---- Main Title ----
st.markdown("<h1 style='text-align:center;'>📌 AI TraceFinder</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;color:lightgreen;'>🔍 Smart Document Analysis</h3>", unsafe_allow_html=True)

# ---- Objective 1: Scanner Identification ----
if task == "Scanner Identification":
    st.subheader("📌 Scanner Brand/Model Prediction")

    # Use os.path.join for model loading
    model = joblib.load(os.path.join(MODELS_DIR, "xgb_model.pkl"))
    le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))

    uploaded_file = st.file_uploader("📤 Upload scanned image", type=["jpg","jpeg","png","tif"])
    if uploaded_file is not None:
        st.image(uploaded_file, width=300)

        if st.button("🚀 Run Prediction"):
            with st.spinner("🔄 Processing image and generating prediction..."):
                # Read + preprocess
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (128, 128))
                features = img.flatten().reshape(1, -1)

                # Predict
                probs = model.predict_proba(features)[0]
                pred_idx = np.argmax(probs)
                pred_label = le.classes_[pred_idx]
                confidence = probs[pred_idx] * 100

            st.success(f"✅ Predicted Scanner: **{pred_label}** ({confidence:.2f}%)")

            # ---- Top-3 Predictions ----
            st.markdown("### 📊 Top-3 Predictions")
            top_indices = np.argsort(probs)[::-1][:3]
            for i in top_indices:
                st.write(f"🔹 {le.classes_[i]} → {probs[i]*100:.2f}%")
                st.progress(int(probs[i]*100))

            # ---- Logs ----
            # Ensure log directory exists and use os.path.join for log file path
            os.makedirs(LOGS_DIR, exist_ok=True)
            LOG_FILE = os.path.join(LOGS_DIR, "scanner_predictions.csv")
            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "file_name": uploaded_file.name,
                "predicted_label": pred_label,
                "confidence": confidence
            }
            if not os.path.exists(LOG_FILE):
                pd.DataFrame([record]).to_csv(LOG_FILE, index=False)
            else:
                df = pd.read_csv(LOG_FILE)
                df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
                df.to_csv(LOG_FILE, index=False)

            st.toast("🎉 Prediction complete!")
            st.success("📂 Prediction logged successfully!")

            # Display recent predictions
            st.markdown("### 📝 Recent Predictions")
            df = pd.read_csv(LOG_FILE).tail(5)
            st.dataframe(df, use_container_width=True)

            with open(LOG_FILE, "rb") as f:
                st.download_button("📥 Download All Logs", data=f,
                                   file_name="scanner_prediction_logs.csv", mime="text/csv")

# ---- Objective 2: Forgery Detection ----
elif task == "Forgery Detection":
    st.subheader("📌 Document Forgery/Tampering Detection")

    # Load Ensemble Models
    xgb_tampered = joblib.load(os.path.join(MODELS_DIR, "xgb_tampered.pkl"))
    cnn_tampered = load_model(os.path.join(MODELS_DIR, "tampered_cnn.keras"))

    uploaded_file = st.file_uploader("📤 Upload document image", type=["jpg","jpeg","png","tif"])
    if uploaded_file is not None:
        st.image(uploaded_file, width=300)

        if st.button("🚀 Run Detection"):
            with st.spinner("🔄 Analyzing forgery..."):
                # Preprocess
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (128,128))

                # XGB branch
                feat = img.flatten().reshape(1,-1)
                proba_xgb = xgb_tampered.predict_proba(feat)[0]

                # CNN branch
                img_cnn = img.reshape(1,128,128,1) / 255.0
                proba_cnn = cnn_tampered.predict(img_cnn)[0]

                # Weighted Ensemble (best: 0.6 CNN + 0.4 XGB)
                final_proba = 0.6*proba_cnn + 0.4*proba_xgb
                pred_idx = np.argmax(final_proba)
                classes = ["Original", "Tampered"]
                pred_label = classes[pred_idx]
                confidence = final_proba[pred_idx]*100

            st.success(f"✅ Prediction: **{pred_label}** ({confidence:.2f}%)")

            # ---- Show Class Probabilities ----
            st.markdown("### 📊 Class Probabilities")
            for i, c in enumerate(classes):
                st.write(f"🔹 {c} → {final_proba[i]*100:.2f}%")
                st.progress(int(final_proba[i]*100))

            # ---- Logs ----
            # Ensure log directory exists and use os.path.join for log file path
            os.makedirs(LOGS_DIR, exist_ok=True)
            LOG_FILE = os.path.join(LOGS_DIR, "forgery_predictions.csv")
            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "file_name": uploaded_file.name,
                "predicted_label": pred_label,
                "confidence": confidence
            }
            if not os.path.exists(LOG_FILE):
                pd.DataFrame([record]).to_csv(LOG_FILE, index=False)
            else:
                df = pd.read_csv(LOG_FILE)
                df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
                df.to_csv(LOG_FILE, index=False)

            st.toast("🎉 Forgery detection complete!")
            st.success("📂 Detection logged successfully!")

            # Display recent detections
            st.markdown("### 📝 Recent Detections")
            df = pd.read_csv(LOG_FILE).tail(5)
            st.dataframe(df, use_container_width=True)

            with open(LOG_FILE, "rb") as f:
                st.download_button("📥 Download All Logs", data=f,
                                   file_name="forgery_detection_logs.csv", mime="text/csv")
