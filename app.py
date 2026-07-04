import os
import sys
import tempfile
import joblib
import pandas as pd
import streamlit as st

# Add src folder to Python path
sys.path.append("src")

from feature_extraction import extract_features_from_image

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="PixelProof",
    page_icon="📷",
    layout="centered"
)

st.title("📷 PixelProof")
st.write("Detect whether an image is a **Real Photograph** or a **Photograph of a Screen**.")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/best_model.pkl")

model = load_model()

# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "bmp", "heic", "heif"]
)

camera_image = st.camera_input("Or capture using webcam")

image = camera_image if camera_image else uploaded_file

if image is not None:

    st.image(image, caption="Input Image", use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(image.read())
        image_path = tmp.name

    with st.spinner("Extracting features..."):

        features = extract_features_from_image(image_path)

        features.pop("image", None)
        features.pop("label", None)

        X = pd.DataFrame([features])

        prediction = model.predict(X)[0]

        probability = model.predict_proba(X)[0]

    os.remove(image_path)

    screen_score = probability[1]
    real_score = probability[0]

    st.divider()

    if prediction == 0:
        st.success("✅ Prediction: REAL PHOTO")
        st.metric("Confidence", f"{real_score*100:.2f}%")
    else:
        st.error("🖥️ Prediction: SCREEN PHOTO")
        st.metric("Confidence", f"{screen_score*100:.2f}%")

    st.subheader("Prediction Scores")

    st.write(f"Real Photo : **{real_score:.4f}**")
    st.write(f"Screen Photo : **{screen_score:.4f}**")
