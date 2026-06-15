import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.express as px
import pandas as pd
import time

# ===================================================
# PAGE CONFIG
# ===================================================

st.set_page_config(
    page_title="FruitVision AI",
    page_icon="🍎",
    layout="wide"
)

# ===================================================
# CUSTOM CSS
# ===================================================

st.markdown("""
<style>

.big-title{
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:#1E3A8A;
}

.sub-title{
    text-align:center;
    color:gray;
    font-size:18px;
}

.card{
    padding:20px;
    border-radius:15px;
    background:#FFFFFF;
    box-shadow:0px 2px 10px rgba(0,0,0,0.15);
}

</style>
""", unsafe_allow_html=True)

# ===================================================
# LOAD MODEL
# ===================================================

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "fruit_classifier_deploy.keras"
    )
    return model

model = load_model()

# ===================================================
# CLASSES
# ===================================================

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

# ===================================================
# HEADER
# ===================================================

st.markdown(
    "<div class='big-title'>🍎 FruitVision AI</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>Deep Learning Based Fruit Classification System</div>",
    unsafe_allow_html=True
)

st.divider()

# ===================================================
# FILE UPLOADER
# ===================================================

uploaded_file = st.file_uploader(
    "Upload Fruit Image",
    type=["jpg","jpeg","png"]
)

# ===================================================
# PREDICTION
# ===================================================

if uploaded_file is not None:

    col1, col2 = st.columns(2)

    # ----------------------------
    # IMAGE COLUMN
    # ----------------------------

    with col1:

        image = Image.open(uploaded_file)

        image = image.convert("RGB")

        st.image(
            image,
            caption="Uploaded Image"
        )

    # ----------------------------
    # PREDICTION COLUMN
    # ----------------------------

    with col2:

        start_time = time.time()

        img = image.resize((128,128))

        img_array = np.array(
            img,
            dtype=np.float32
        )

        img_array = img_array / 255.0

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

        confidence = np.max(
            prediction
        ) * 100

        end_time = time.time()

        st.markdown(
            f"""
            <div class='card'>

            <h1 style='text-align:center'>
            {emoji[predicted_class]}
            </h1>

            <h2 style='text-align:center'>
            {predicted_class}
            </h2>

            <h3 style='text-align:center;color:green'>
            Confidence: {confidence:.2f}%
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
            f"{end_time-start_time:.3f}s"
        )

    # ===================================================
    # BAR CHART
    # ===================================================

    st.divider()

    st.subheader(
        "📊 Prediction Probabilities"
    )

    df = pd.DataFrame({
        "Fruit": classes,
        "Probability": prediction*100
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

# ===================================================
# MODEL INFO
# ===================================================

st.divider()

st.subheader("ℹ️ Model Information")

m1,m2,m3 = st.columns(3)

m1.metric(
    "Model Type",
    "CNN"
)

m2.metric(
    "Classes",
    "3"
)

m3.metric(
    "Input Size",
    "128×128"
)

st.success("""
Supported Fruits

🍎 Apple

🍌 Banana

🍊 Orange
""")
