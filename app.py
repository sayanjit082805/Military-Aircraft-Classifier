import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
from PIL import Image
import joblib
import random

st.set_page_config(
    page_title="Military Aircraft Classifier",
    page_icon="✈️",
    initial_sidebar_state="expanded",
)

aircraft_samples = [
    {
        "name": "Lockheed Martin F-35",
        "image": "public/f35.jpg",
    },
    {
        "name": "Sukhoi Su-57",
        "image": "public/su57.jpg",
    },
    {
        "name": "A-10 Warthog",
        "image": "public/a10.jpg",
    },
    {
        "name": "Dassault Rafale",
        "image": "public/rafale.jpg",
    },
    {
        "name": "General Dynamics F-16",
        "image": "public/f16.jpg",
    },
    {
        "name": "Tupolev TU-22M3",
        "image": "public/tu22m.jpg",
    },
    {
        "name": "Tornado",
        "image": "public/tornado.jpg",
    },
    {
        "name": "Mig-31",
        "image": "public/mig31.jpg",
    },
    {
        "name": "Dassault Mirage",
        "image": "public/mirage.jpg",
    },
]

model = tf.keras.models.load_model("classifier.keras")
classes = joblib.load("class_names.pkl")

st.title("Military Aircraft Classifier")

st.subheader(
    "Classify military aircraft types from aerial images using the ResNet architecture.",
    divider="gray",
)

st.write(
    """
   A deep learning project to classify military aircraft into nearly 81 distinct aircraft types. The model is trained on the Military Aircraft Detection dataset from Kaggle with PASCAL VOC format annotations and uses a ResNet (Residual Neural Network) architecture for classification.     
"""
)

st.info(
    "Some classifications may be inaccurate. The model may not generalize perfectly to all images."
)

uploaded_file = st.file_uploader("Select an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((256, 256))
    img_rgb = img.convert("RGB")
    img_array = np.array(img_rgb)
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("Classify"):

        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Preprocessing image...")
        progress_bar.progress(25)

        status_text.text("Running model inference...")
        progress_bar.progress(50)

        predictions = model.predict(img_array)
        progress_bar.progress(75)

        status_text.text("Processing results...")
        predicted_class = np.argmax(predictions, axis=1)[0]
        progress_bar.progress(100)

        status_text.text("Complete!")

        progress_bar.empty()
        status_text.empty()

        st.success(f"Prediction: **{classes[predicted_class]}**")

st.divider()

if "aircraft_samples" not in st.session_state:
    st.session_state.aircraft_samples = random.choice(aircraft_samples)

img = st.session_state.aircraft_samples
st.sidebar.image(img["image"], caption=img["name"])

st.sidebar.header("Dataset Information", divider="gray")

st.sidebar.markdown(
    """ 
    The dataset has been taken from [Kaggle](https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset/data). It contains around 36,000 images of aircraft of various different countries.

**Statistics** :

- **Total Images**: ~36,000
- **Training Set**: ~25,000 images
- **Test Set**: ~4,000 images
- **Validation Set**: ~7,000 images
- **Classes**: 81 distinct military aircraft types
"""
)


st.sidebar.header("Model Metrics", divider="gray")

st.sidebar.markdown(
    """ 
The model achieves an overall accuracy of ~75% on the validation data and ~74% on the test data.
"""
)


st.sidebar.markdown(
    """
> "All aircraft images used are sourced from publicly available search engines (Google Images, DuckDuckGo). These images are used for non-commercial, educational, and demonstration purposes only. 
"""
)

st.sidebar.markdown(
    """> The model is not intended for real-world applications and should not be used for any commercial or operational purposes."""
)
