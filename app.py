import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import torch
from transformers import pipeline
from PIL import Image, ImageDraw

# --- 1. Futuristic UI Setup ---
st.set_page_config(page_title="VisionPro AI", layout="wide")

st.markdown("""
    <style>
    .stApp { background: #060d16; color: #e2e8f0; }
    [data-testid="stVerticalBlock"] > div:has(div.element-container) {
        background: rgba(16, 24, 40, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid #00FFCC;
        border-radius: 15px; padding: 20px;
    }
    h1 { background: linear-gradient(90deg, #00FFCC, #0099FF); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Load AI Model (FIXED FOR CPU) ---
@st.cache_resource
def load_model():
    # We force torch_dtype to float32 to fix the 'Half' error on CPU servers
    return pipeline(
        model="google/owlv2-base-patch16", 
        task="zero-shot-object-detection",
        torch_dtype=torch.float32,
        device="cpu"
    )

detector = load_model()

# --- 3. Sidebar ---
st.sidebar.title("⚡ AI PANEL")
target_labels = st.sidebar.text_input("Look for:", "apple, orange, person, laptop")
threshold = st.sidebar.slider("Sensitivity", 0.05, 1.0, 0.20)
labels = [l.strip() for l in target_labels.split(",") if l.strip()]

# Session state to store session history
if 'history' not in st.session_state:
    st.session_state.history = []

# --- 4. Main App ---
st.title("👁️ VisionPro: Neural Scanner")
t1, t2 = st.tabs(["🎥 LIVE SCAN", "📸 UPLOAD"])

def process_frame(frame):
    img = frame.to_ndarray(format="bgr24")
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Inference
    predictions = detector(pil_img, candidate_labels=labels, threshold=threshold)
    
    draw = ImageDraw.Draw(pil_img)
    for pred in predictions:
        box = pred["box"]
        draw.rectangle((box["xmin"], box["ymin"], box["xmax"], box["ymax"]), outline="#00FFCC", width=4)
        draw.text((box["xmin"], box["ymin"]-20), f"{pred['label'].upper()}", fill="#00FFCC")
        # Save to history
        if pred['label'] not in st.session_state.history:
            st.session_state.history.append(pred['label'])
            
    return av.VideoFrame.from_ndarray(np.array(pil_img), format="rgb24")

with t1:
    webrtc_streamer(
        key="live",
        video_frame_callback=process_frame,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False}
    )

with t2:
    up = st.file_uploader("Upload Image")
    if up:
        img = Image.open(up).convert("RGB")
        res = detector(img, candidate_labels=labels, threshold=threshold)
        draw = ImageDraw.Draw(img)
        for r in res:
            box = r["box"]
            draw.rectangle((box["xmin"], box["ymin"], box["xmax"], box["ymax"]), outline="#00FFCC", width=5)
        st.image(img)

# --- 5. Session Report ---
st.sidebar.markdown("---")
if st.sidebar.button("Generate Session Report"):
    report = " \n".join(st.session_state.history)
    st.sidebar.download_button("Download Report", report, file_name="vision_report.txt")
