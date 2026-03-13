import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2
import numpy as np
from transformers import pipeline
from PIL import Image, ImageDraw

# --- 1. Futuristic Cyberpunk Theme ---
st.set_page_config(page_title="VisionPro AI", page_icon="👁️", layout="wide")

st.markdown("""
    <style>
    /* Gradient Background */
    .stApp {
        background: linear-gradient(160deg, #060d16 0%, #101828 100%);
        color: #e2e8f0;
    }
    /* Cyberpunk Glass Panels */
    [data-testid="stVerticalBlock"] > div:has(div.element-container) {
        background: rgba(16, 24, 40, 0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 255, 204, 0.2);
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
    }
    /* Neon Headlines */
    h1, h2, h3 {
        background: linear-gradient(90deg, #00FFCC, #0099FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: -1px;
    }
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #060d16 !important;
        border-right: 2px solid #00FFCC;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Load AI Model ---
@st.cache_resource
def load_model():
    # Using the base model for speed and memory efficiency
    return pipeline(model="google/owlv2-base-patch16", task="zero-shot-object-detection")

detector = load_model()

# --- 3. Sidebar UI ---
st.sidebar.title("⚡ AI CONFIG")
target_labels = st.sidebar.text_area("TARGET OBJECTS (comma separated):", "apple, orange, pomegranate, mobile phone, person")
threshold = st.sidebar.slider("AI SENSITIVITY", 0.05, 1.0, 0.20)
labels = [l.strip() for l in target_labels.split(",") if l.strip()]

# --- 4. Main View ---
st.title("👁️ VisionPro: Live Object Intelligence")

tab1, tab2 = st.tabs(["🛰️ LIVE SCANNER", "📸 STATIC ANALYSIS"])

with tab1:
    st.write("### Real-time Neural Detection")
    
    # FIXED: The callback function is now outside the class for better compatibility
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 1. Convert BGR to RGB for AI
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # 2. Run Inference
        predictions = detector(pil_img, candidate_labels=labels, threshold=threshold)
        
        # 3. Draw Results
        draw = ImageDraw.Draw(pil_img)
        for pred in predictions:
            box = pred["box"]
            # Futuristic Neon Cyan Box
            draw.rectangle((box["xmin"], box["ymin"], box["xmax"], box["ymax"]), outline="#00FFCC", width=4)
            # Label
            draw.text((box["xmin"], box["ymin"]-20), f"{pred['label'].upper()}", fill="#00FFCC")
        
        # 4. Convert back to frame
        return av.VideoFrame.from_ndarray(np.array(pil_img), format="rgb24")

    webrtc_streamer(
        key="vision-live",
        video_frame_callback=video_frame_callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False}
    )

with tab2:
    st.write("### High-Resolution Image Analysis")
    uploaded_file = st.file_uploader("Drop image files here", type=['jpg', 'png', 'webp'])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        with st.spinner("Executing Deep Scan..."):
            results = detector(img, candidate_labels=labels, threshold=threshold)
            draw = ImageDraw.Draw(img)
            for res in results:
                box = res["box"]
                draw.rectangle((box["xmin"], box["ymin"], box["xmax"], box["ymax"]), outline="#00FFCC", width=5)
        st.image(img, use_container_width=True)
