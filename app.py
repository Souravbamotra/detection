import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoHTMLAttributes
import av
from transformers import pipeline
from PIL import Image, ImageDraw

# --- Modern UI Configuration ---
st.set_page_config(page_title="VisionPro AI", page_icon="👁️", layout="wide")

st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); color: white; }
    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    .stTabs [data-baseweb="tab"] { 
        background-color: rgba(255, 255, 255, 0.05); 
        border-radius: 10px 10px 0 0; 
        padding: 10px 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Load the AI Model ---
@st.cache_resource
def load_model():
    return pipeline(model="google/owlv2-base-patch16", task="zero-shot-object-detection")

detector = load_model()

# --- Sidebar Controls ---
st.sidebar.title("🛠️ Settings")
target_labels = st.sidebar.text_input("Objects to detect (comma separated):", "apple, orange, person")
threshold = st.sidebar.slider("Sensitivity", 0.05, 1.0, 0.20)
labels = [l.strip() for l in target_labels.split(",")]

# --- Main App Interface ---
st.title("👁️ VisionPro AI")
tab1, tab2 = st.tabs(["🚀 Live Camera", "📸 Image Upload"])

# --- TAB 1: Real-Time Detection ---
with tab1:
    st.subheader("Live Feed")
    
    class VideoProcessor:
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            # Convert to RGB for the AI
            pil_img = Image.fromarray(img)
            predictions = detector(pil_img, candidate_labels=labels, threshold=threshold)
            
            # Draw on the frame
            draw = ImageDraw.Draw(pil_img)
            for pred in predictions:
                box = pred["box"]
                draw.rectangle((box["xmin"], box["ymin"], box["xmax"], box["ymax"]), outline="cyan", width=4)
                draw.text((box["xmin"], box["ymin"]-15), f"{pred['label']}", fill="cyan")
            
            return av.VideoFrame.from_ndarray(np.array(pil_img), format="rgb24")

    webrtc_streamer(
        key="vision-pro",
        video_frame_callback=VideoProcessor().recv,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False}
    )

# --- TAB 2: Image Upload (Original Mode) ---
with tab2:
    uploaded_file = st.file_uploader("Upload a photo...", type=['jpg', 'png', 'webp'])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        with st.spinner("Analyzing..."):
            results = detector(img, candidate_labels=labels, threshold=threshold)
            draw = ImageDraw.Draw(img)
            for res in results:
                box = res["box"]
                draw.rectangle((box["xmin"], box["ymin"], box["xmax"], box["ymax"]), outline="#00FFCC", width=5)
        st.image(img, use_container_width=True)
