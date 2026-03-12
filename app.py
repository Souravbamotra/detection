import streamlit as st
from transformers import pipeline
from PIL import Image, ImageDraw
import torch

# --- App Config ---
st.set_page_config(page_title="Global Fruit Finder", page_icon="🌍")
st.title("🌍 Zero-Shot Global Fruit Finder")
st.write("Type *any* fruit name in the world, and the AI will find it.")

# --- Load Zero-Shot Model ---
@st.cache_resource
def load_zero_shot_detector():
    # OWL-ViT v2 is excellent for detecting objects it has never 'officially' learned
    return pipeline(model="google/owlv2-base-patch16-ensemble", task="zero-shot-object-detection")

detector = load_zero_shot_detector()

# --- Sidebar Controls ---
st.sidebar.header("Search Settings")
search_query = st.sidebar.text_input("What fruit are we looking for?", "apple, orange, pomegranate")
threshold = st.sidebar.slider("Sensitivity (Threshold)", 0.05, 1.0, 0.15)

# --- Upload Image ---
uploaded_file = st.file_uploader("Upload a photo of mystery fruits...", type=['jpg', 'png', 'webp'])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Process Search Query
    labels = [label.strip() for label in search_query.split(",")]
    
    with st.spinner(f"Searching for {labels}..."):
        # Run Inference
        predictions = detector(image, candidate_labels=labels, threshold=threshold)
        
        # Draw Results
        draw = ImageDraw.Draw(image)
        for prediction in predictions:
            box = prediction["box"]
            label = prediction["label"]
            score = prediction["score"]
            
            # Draw Bounding Box
            draw.rectangle((box["xmin"], box["ymin"], box["xmax"], box["ymax"]), outline="red", width=3)
            draw.text((box["xmin"], box["ymin"]), f"{label} ({round(score, 2)})", fill="white")

    st.image(image, caption="AI Detection Results", use_container_width=True)
    
    # Display summary
    if predictions:
        st.success(f"Found {len(predictions)} items!")
    else:
        st.warning("No matches found. Try lowering the Sensitivity or changing the fruit name.")