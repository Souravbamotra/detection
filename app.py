import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
from transformers import pipeline
from PIL import Image, ImageDraw

# --- 1. Modern Glassmorphism Theme ---
st.set_page_config(page_title="VisionPro AI", page_icon="👁️", layout="wide")

def apply_modern_style():
    st.markdown("""
    <style>
    /* Main Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #ffffff;
    }
    /* Glassmorphism Containers */
    [data-testid="stVerticalBlock"] > div:has(div.element-container) {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }
    /* Custom Sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.3) !important;
        border-right: 1px solid rgba(0, 255, 204, 0.2);
    }
    /* Titles & Headers */
    h1, h2, h3 {
        color: #00FFCC !important;
        font-family: 'Inter', sans-serif;
        text-shadow: 0 0 10px rgba(0, 255, 204, 0.3);
    }
    /* Stylish Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        color: white; border: none; border-radius: 30px;
        padding: 10px 25px; transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05); box-shadow: 0 0 15px rgba(0, 210, 255, 0.5);
    }
    </style>
    """, unsafe_allow_html=True)

apply_modern_style()

# --- 2. Load Model ---
@st.cache_resource
def load_model():
    return pipeline(model="google/owlv2-base-patch16", task="zero-shot-object-detection")

detector = load_model()

# --- 3. Sidebar UI ---
st.sidebar.title("🛠️ Vision Controls")
target_labels = st.sidebar.text_input("Look for:", "apple, orange, person, mobile phone")
threshold = st.sidebar.slider("Sensitivity", 0.05, 1.0, 0.15)
labels = [l.strip() for l in target_labels.split(",") if l.strip()]

# --- 4. Main View ---
st.title("👁️ VisionPro AI Station")
tab1, tab2 = st.tabs(["🎥 Live Stream", "📤 Upload File"])

with tab1:
    st.info("Hold an object up to the camera to see instant AI labeling.")
    
    class VideoProcessor:
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            # Image processing
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            predictions = detector(pil_img, candidate_labels=labels, threshold=threshold)
            
            draw = ImageDraw.Draw(pil_img)
            for pred in predictions:
                box = pred["box"]
                # Futuristic Cyan boxes
                draw.rectangle((box["xmin"], box["ymin"], box["xmax"], box["ymax"]), outline="#00FFCC", width=4)
                draw.text((box["xmin"], box["ymin"]-20), f"{pred['label'].upper()}", fill="#00FFCC")
            
            return av.VideoFrame.from_ndarray(np.array(pil_img), format="rgb24")

    webrtc_streamer(
        key="vision-live",
        video_frame_callback=VideoProcessor().recv,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False}
    )

with tab2:
    uploaded_file = st.file_uploader("Drop an image here", type=['jpg', 'png', 'webp'])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        with st.spinner("AI Analysis in progress..."):
            results = detector(img, candidate_labels=labels, threshold=threshold)
            draw = ImageDraw.Draw(img)
            for res in results:
                box = res["box"]
                draw.rectangle((box["xmin"], box["ymin"], box["xmax"], box["ymax"]), outline="#00FFCC", width=5)
        st.image(img, use_container_width=True)
