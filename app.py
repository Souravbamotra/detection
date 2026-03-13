import streamlit as st
import cv2
import numpy as np
import json
import time
import pandas as pd
from ultralytics import YOLO
from PIL import Image
import numpy as np
from datetime import datetime
from pathlib import Path

# -------------------------------
# PAGE CONFIG
# -------------------------------

st.set_page_config(
    page_title="AI Vision Detection",
    page_icon="🤖",
    layout="wide"
)

# -------------------------------
# LOAD CSS
# -------------------------------

def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# -------------------------------
# HEADER
# -------------------------------

st.markdown(
"""
<div class="title">
AI Vision Detection Platform
</div>
""",
unsafe_allow_html=True
)

# -------------------------------
# SIDEBAR SETTINGS
# -------------------------------

st.sidebar.title("Settings")

model_choice = st.sidebar.selectbox(
"Model",
["yolov8n.pt","yolov8s.pt","yolov8m.pt"]
)

confidence = st.sidebar.slider(
"Confidence Threshold",
0.1,1.0,0.5
)

mode = st.sidebar.radio(
"Detection Mode",
["Image Upload","Webcam"]
)

# -------------------------------
# LOAD MODEL
# -------------------------------

@st.cache_resource
def load_model(name):
    return YOLO(name)

model = load_model(model_choice)

# -------------------------------
# HISTORY FUNCTIONS
# -------------------------------

history_file = "history.json"

def save_history(data):

    try:
        with open(history_file,"r") as f:
            history = json.load(f)
    except:
        history = []

    history.append(data)

    with open(history_file,"w") as f:
        json.dump(history,f)

def load_history():
    try:
        with open(history_file) as f:
            return json.load(f)
    except:
        return []

# -------------------------------
# IMAGE DETECTION
# -------------------------------

def detect_image(image):

    start = time.time()

    results = model(image, conf=confidence)

    annotated = results[0].plot()

    end = time.time()

    names = results[0].names

    boxes = results[0].boxes

    detected = []

    if boxes is not None:
        for c in boxes.cls:
            detected.append(names[int(c)])

    return annotated, detected, round(end-start,2)

# -------------------------------
# IMAGE UPLOAD MODE
# -------------------------------

if mode == "Image Upload":

    st.subheader("Upload Image")

    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg","png","jpeg"]
    )

    if uploaded:

        image = Image.open(uploaded)

        st.image(image, caption="Original Image")

        if st.button("Run Detection"):

            with st.spinner("AI analyzing image..."):

                result, objects, runtime = detect_image(image)

            st.image(result, caption="Detection Result")

            # stats

            st.subheader("Detection Statistics")

            col1,col2,col3 = st.columns(3)

            col1.metric("Objects Detected",len(objects))

            most_common = max(set(objects), key=objects.count) if objects else "None"

            col2.metric("Most Common",most_common)

            col3.metric("Processing Time",f"{runtime}s")

            # save image

            filename = f"detections/{datetime.now().timestamp()}.png"

            cv2.imwrite(filename,result)

            save_history({
                "time":str(datetime.now()),
                "objects":objects,
                "file":filename
            })

            with open(filename,"rb") as f:
                st.download_button(
                    "Download Result",
                    f,
                    "detection.png"
                )

# -------------------------------
# WEBCAM MODE
# -------------------------------

if mode == "Webcam":

    st.subheader("Live Webcam Detection")

    st.warning("⚠ Webcam detection works only on local machine.")

    st.info("Run this app locally to use webcam detection.")

    while run:

        ret, frame = camera.read()

        if not ret:
            st.error("Camera error")
            break

        results = model(frame, conf=confidence)

        annotated = results[0].plot()

        frame_window.image(annotated)

    camera.release()

# -------------------------------
# HISTORY SECTION
# -------------------------------

st.divider()

st.subheader("Detection History")

history = load_history()

if history:

    df = pd.DataFrame(history)

    st.dataframe(df)

else:
    st.info("No detections yet.")




