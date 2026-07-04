import os
import sys
import tempfile
import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# Add src folder
# --------------------------------------------------

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
# Custom Theme
# --------------------------------------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]{
    font-family:'Inter',sans-serif;
}

.stApp{
    background:#FAFAFA;
}

.block-container{
    max-width:850px;
    padding-top:2.5rem;
    padding-bottom:3rem;
}

/* Hide Streamlit branding */

#MainMenu{
    visibility:hidden;
}

footer{
    visibility:hidden;
}

header{
    visibility:hidden;
}

/* Title */

h1{
    text-align:center;
    font-weight:700;
    color:#202020;
    letter-spacing:-1px;
}

h2,h3{
    color:#202020;
}

p{
    color:#5A5A5A;
    line-height:1.8;
}

/* Sidebar */

section[data-testid="stSidebar"]{
    background:#FFFFFF;
    border-right:1px solid #ECECEC;
}

/* Upload */

[data-testid="stFileUploader"]{
    border-radius:24px;
    border:1px solid #E8E8E8;
    padding:18px;
    background:white;
}

/* Camera */

[data-testid="stCameraInput"]{
    border-radius:24px;
    border:1px solid #E8E8E8;
    padding:18px;
    background:white;
}

/* Uploaded Image */

img{
    border-radius:24px;
}

/* Metric Cards */

[data-testid="metric-container"]{
    background:white;
    border-radius:24px;
    padding:20px;
    border:1px solid #ECECEC;
    box-shadow:0 8px 20px rgba(0,0,0,.04);
}

/* Progress */

.stProgress > div > div > div{
    background:#202020;
}

/* Success */

.stSuccess{
    background:#FFFFFF;
    color:#202020;
    border:1px solid #ECECEC;
    border-radius:24px;
}

/* Error */

.stError{
    background:#FFFFFF;
    color:#202020;
    border:1px solid #ECECEC;
    border-radius:24px;
}

hr{
    margin-top:2rem;
    margin-bottom:2rem;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("PixelProof")

st.sidebar.markdown("""
### Classical Computer Vision

PixelProof distinguishes between:

- 📸 Real Photograph
- 🖥️ Screen Photograph

The model combines handcrafted image features:

- Local Binary Patterns (LBP)
- FFT Frequency Analysis
- GLCM Texture Features
- Edge Statistics
- RGB & HSV Color Features
- Brightness & Contrast
""")

# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("📷 PixelProof")

st.markdown(
"""
<div style='text-align:center;
font-size:18px;
color:#666;
margin-bottom:40px;'>

Determine whether an uploaded image is a
<strong>real-world photograph</strong> or a
<strong>photograph of a digital display</strong>.

</div>
""",
unsafe_allow_html=True
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
    "Upload an image",
    type=["jpg","jpeg","png","bmp","heic","heif"]
)

camera_image = st.camera_input("Or capture an image")

image = camera_image if camera_image else uploaded_file

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if image is not None:

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

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

    confidence = max(real_score, screen_score)

    st.divider()

    if prediction == 0:

        st.markdown("""
        <div style="
        background:white;
        border:1px solid #ECECEC;
        border-radius:28px;
        padding:30px;
        text-align:center;
        margin-bottom:20px;
        ">

        <h2 style="margin-bottom:5px;">📸 Real Photograph</h2>

        <p style="margin:0;color:#666;">
        The uploaded image is classified as a genuine photograph.
        </p>

        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div style="
        background:white;
        border:1px solid #ECECEC;
        border-radius:28px;
        padding:30px;
        text-align:center;
        margin-bottom:20px;
        ">

        <h2 style="margin-bottom:5px;">🖥️ Screen Photograph</h2>

        <p style="margin:0;color:#666;">
        The uploaded image is classified as a photograph of a display.
        </p>

        </div>
        """, unsafe_allow_html=True)

    st.subheader("Confidence")

    st.progress(float(confidence))

    st.markdown(
        f"<h2 style='text-align:center'>{confidence*100:.2f}%</h2>",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "📸 Real",
            f"{real_score*100:.2f}%"
        )

    with col2:
        st.metric(
            "🖥️ Screen",
            f"{screen_score*100:.2f}%"
        )

    st.divider()

    st.caption(
        "Prediction is based on handcrafted image features including texture, "
        "frequency-domain analysis, edge statistics, and color distribution."
    )
