import streamlit as st
from transformers import pipeline
from PIL import Image, ImageDraw

# --- App Config ---
st.set_page_config(page_title="Universal Object Finder", layout="wide")

@st.cache_resource
def load_owl():
    # OWLv2 is designed for 'Open Vocabulary' (detecting everything)
    return pipeline(model="google/owlv2-base-patch16", task="zero-shot-object-detection")

detector = load_owl()

# --- Sidebar: Choose your World ---
st.sidebar.title("🌍 Detection Universe")
mode = st.sidebar.selectbox("What are we looking for?", 
    ["Everything (Infinite)", "Fruits & Food", "Vehicles & Transport", "Tools & Hardware"])

# Define the 'Vocabulary' based on the mode
vocab_map = {
    "Everything (Infinite)": "object, thing, item", # Very broad
    "Fruits & Food": "apple, banana, orange, pomegranate, bread, milk",
    "Vehicles & Transport": "car, ship, airplane, bicycle, truck, boat",
    "Tools & Hardware": "hammer, screwdriver, wrench, nail, drill, saw"
}

# Allow the user to edit the list manually too
search_query = st.sidebar.text_area("Specific labels (comma separated):", value=vocab_map[mode])
threshold = st.sidebar.slider("Sensitivity", 0.05, 1.0, 0.15)

# --- Logic ---
uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'webp'])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    labels = [l.strip() for l in search_query.split(",")]
    
    with st.spinner(f"Scanning for {labels}..."):
        results = detector(img, candidate_labels=labels, threshold=threshold)
        
        draw = ImageDraw.Draw(img)
        for res in results:
            box = res["box"]
            label = res["label"]
            draw.rectangle((box["xmin"], box["ymin"], box["xmax"], box["ymax"]), outline="cyan", width=3)
            draw.text((box["xmin"], box["ymin"] - 10), label, fill="cyan")

    st.image(img, use_container_width=True)
