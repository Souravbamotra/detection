import streamlit as st
import json
import time
import pandas as pd
import numpy as np
from ultralytics import YOLO
from PIL import Image
from datetime import datetime
from pathlib import Path

# ----------------------------
# PAGE CONFIG
# ----------------------------

st.set_page_config(
    page_title="AI Vision Detection",
    page_icon="🤖",
    layout="wide"
)

# ----------------------------
# TITLE
# ----------------------------

st.markdown(
"""
# 🤖 AI Vision Detection Platform
Upload an image and detect objects using YOLOv8 AI
"""
)

# ----------------------------
# SIDEBAR SETTINGS
# ----------------------------

st.sidebar.title("⚙ Settings")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["yolov8n.pt", "yolov8s.pt"]
)

confidence = st.sidebar.slider(
    "Confidence Threshold",
    0.1,
    1.0,
    0.5
)

# ----------------------------
# LOAD MODEL
# ----------------------------

@st.cache_resource
def load_model(name):
    model = YOLO(name)
    return model

model = load_model(model_choice)

# ----------------------------
# CREATE FOLDERS
# ----------------------------

Path("detections").mkdir(exist_ok=True)

history_file = "history.json"

# ----------------------------
# HISTORY FUNCTIONS
# ----------------------------

def load_history():

    try:
        with open(history_file, "r") as f:
            data = json.load(f)
    except:
        data = []

    return data


def save_history(record):

    history = load_history()

    history.append(record)

    with open(history_file, "w") as f:
        json.dump(history, f)


# ----------------------------
# DETECTION FUNCTION
# ----------------------------

def detect_objects(image):

    start = time.time()

    results = model(image, conf=confidence)

    annotated = results[0].plot()

    end = time.time()

    runtime = round(end - start, 2)

    names = results[0].names

    boxes = results[0].boxes

    detected = []

    if boxes is not None:

        for c in boxes.cls:
            detected.append(names[int(c)])

    return annotated, detected, runtime


# ----------------------------
# IMAGE UPLOAD
# ----------------------------

st.subheader("📤 Upload Image")

uploaded = st.file_uploader(
    "Upload image for AI detection",
    type=["jpg", "jpeg", "png"]
)

if uploaded:

    image = Image.open(uploaded)

    st.image(image, caption="Original Image", use_column_width=True)

    if st.button("🚀 Run Detection"):

        with st.spinner("AI analyzing image..."):

            result, objects, runtime = detect_objects(image)

        st.image(result, caption="Detection Result", use_column_width=True)

        # ----------------------------
        # STATISTICS
        # ----------------------------

        st.subheader("📊 Detection Statistics")

        col1, col2, col3 = st.columns(3)

        col1.metric("Objects Detected", len(objects))

        if objects:
            most_common = max(set(objects), key=objects.count)
        else:
            most_common = "None"

        col2.metric("Most Common", most_common)

        col3.metric("Processing Time", f"{runtime}s")

        # ----------------------------
        # SAVE RESULT
        # ----------------------------

        filename = f"detections/{datetime.now().timestamp()}.png"

        Image.fromarray(result).save(filename)

        save_history({
            "time": str(datetime.now()),
            "objects": objects,
            "file": filename
        })

        # ----------------------------
        # DOWNLOAD BUTTON
        # ----------------------------

        with open(filename, "rb") as f:

            st.download_button(
                "⬇ Download Result",
                f,
                file_name="detection.png"
            )


# ----------------------------
# HISTORY SECTION
# ----------------------------

st.divider()

st.subheader("🕘 Detection History")

history = load_history()

if history:

    df = pd.DataFrame(history)

    st.dataframe(df)

else:

    st.info("No detection history yet.")
