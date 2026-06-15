import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.express as px
import pandas as pd
import time

# ======================================
# PAGE CONFIG
# ======================================

st.set_page_config(
    page_title="FruitVision AI",
    page_icon="🍎",
    layout="wide"
)

# ======================================
# CUSTOM CSS
# ======================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.big-title {
    text-align:center;
    font-size:40px;
    font-weight:bold;
    color:#1E3A8A;
}

.sub-title {
    text-align:center;
    font-size:18px;
    color:gray;
}

.prediction-card {
    padding:20px;
    border-radius:15px;
    background:white;
    box-shadow:0px 4px 15px rgba(0,0,0,0.1);
}

.metric-card {
    padding:15px;
    border-radius:10px;
    background:#ffffff;
    text-align:center;
    box-shadow:0px 2px 10px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# ======================================
# LOAD MODEL
# ======================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "fruit_classifier.h5"
    )

model = load_model()

# ======================================
# CLASSES
# ======================================

classes = [
    "Apple",
    "Banana",
    "Orange"
]

emoji = {
    "Apple":"🍎",
    "Banana":"🍌",
    "Orange":"🍊"
}

# ======================================
# HEADER
# ======================================

st.markdown(
    '<p class="big-title">🍎 FruitVision AI</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-title">Deep Learning Based Fruit Classification System</p>',
    unsafe_allow_html=True
)

st.divider()

# ======================================
# LAYOUT
# ======================================

col1, col2 = st.columns([1,1])

with col1:

    st.subheader("📤 Upload Fruit Image")

    uploaded_file = st.file_uploader(
        "",
        type=["jpg","jpeg","png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file)

        st.image(
            image,
            use_container_width=True
        )

with col2:

    st.subheader("🤖 AI Prediction")

    if uploaded_file:

        start = time.time()

        img = image.resize((128,128))

        img_array = np.array(img)

        img_array = img_array/255.0

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        prediction = model.predict(
            img_array,
            verbose=0
        )[0]

        predicted_class = classes[
            np.argmax(prediction)
        ]

        confidence = (
            np.max(prediction)*100
        )

        end = time.time()

        st.markdown(
        f"""
        <div class="prediction-card">

        <h1 style='text-align:center'>
        {emoji[predicted_class]}
        </h1>

        <h2 style='text-align:center'>
        {predicted_class}
        </h2>

        <h3 style='text-align:center;color:green'>
        Confidence:
        {confidence:.2f}%
        </h3>

        </div>
        """,
        unsafe_allow_html=True
        )

        c1,c2,c3 = st.columns(3)

        c1.metric(
            "Prediction",
            predicted_class
        )

        c2.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        c3.metric(
            "Inference Time",
            f"{(end-start):.3f}s"
        )

# ======================================
# PROBABILITY CHART
# ======================================

if uploaded_file:

    st.divider()

    st.subheader(
        "📊 Prediction Probabilities"
    )

    df = pd.DataFrame({
        "Fruit": classes,
        "Probability":
        prediction*100
    })

    fig = px.bar(
        df,
        x="Fruit",
        y="Probability",
        text="Probability",
        title="Class Confidence Scores"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ======================================
# MODEL INFORMATION
# ======================================

st.divider()

st.subheader("ℹ️ Model Information")

info1,info2,info3 = st.columns(3)

info1.metric(
    "Model Type",
    "CNN"
)

info2.metric(
    "Classes",
    "3"
)

info3.metric(
    "Input Size",
    "128x128"
)

st.success(
"""
Supported Fruits:

🍎 Apple

🍌 Banana

🍊 Orange
"""
)