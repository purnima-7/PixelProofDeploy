import os
import sys
import tempfile
import joblib
import pandas as pd
import streamlit as st

# Add src folder
sys.path.append("src")

from feature_extraction import extract_features_from_image

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="PixelProof",
    page_icon="📷",
    layout="centered"
)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("📷 PixelProof")
st.sidebar.markdown(
"""
### Real vs Screen Detector

This application predicts whether an uploaded image is:

- 📸 **Real Photograph**
- 🖥️ **Photograph of a Screen**

The prediction is based on handcrafted image-processing features including:

- Texture (LBP)
- Frequency Analysis (FFT)
- Edge Statistics
- GLCM Texture Features
- Color Distribution
"""
)

# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("📷 PixelProof")
st.markdown(
"""
Upload an image or capture one using your webcam.

The model predicts whether the image is a **real photograph**
or a **photo taken of a digital screen**.
"""
)

# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("models/best_model.pkl")

model = load_model()

# --------------------------------------------------
# Input
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png", "bmp", "heic", "heif"]
)

camera_image = st.camera_input("Or Capture Image")

image = camera_image if camera_image else uploaded_file

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if image is not None:

    st.image(image, caption="Uploaded Image", use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(image.read())
        image_path = tmp.name

    with st.spinner("Analyzing image..."):

        features = extract_features_from_image(image_path)

        features.pop("image", None)
        features.pop("label", None)

        X = pd.DataFrame([features])

        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]

    os.remove(image_path)

    real_score = probability[0]
    screen_score = probability[1]

    st.divider()

    if prediction == 0:
        confidence = real_score
        st.success("### ✅ Prediction: REAL PHOTO")
    else:
        confidence = screen_score
        st.error("### 🖥️ Prediction: SCREEN PHOTO")

    st.metric("Model Confidence", f"{confidence*100:.2f}%")

    st.progress(float(confidence))

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "📸 Real Photo",
            f"{real_score*100:.2f}%"
        )

    with col2:
        st.metric(
            "🖥️ Screen Photo",
            f"{screen_score*100:.2f}%"
        )

    st.divider()

    st.caption(
        "Prediction is based on handcrafted image features extracted "
        "using texture, frequency-domain, color, and edge analysis."
    )
