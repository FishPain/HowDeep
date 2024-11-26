import streamlit as st
import requests
from PIL import Image
from st_clickable_images import clickable_images

from ml.inference_worker import InferenceWorker

inference_worker = InferenceWorker()
inference_worker.load_model()

# Mock process_image() method
def check_image(image):
    prediction, overlay = inference_worker.predict_and_explain(image)

    return prediction, overlay


st.set_page_config(layout="wide")

# Centered title using custom HTML and CSS
st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 20px;
    }
    </style>
    <div class="title">HowReal? Check for Deepfake Now!</div>
    """,
    unsafe_allow_html=True,
)

# File uploader
image_to_process = None  # To store the image to be sent to process_image()

c1, c2 = st.columns(2)
image = None

import os

os.environ["STREAMLIT_SERVER_ENABLE_STATIC_SERVING"] = "true"

with c1:
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    # Display clickable images if no upload
    st.subheader("Or Select an Image:")
    clicked = clickable_images(
        [
            "http://localhost:8508/app/static/valid_fake_0004573.png",
            "http://localhost:8508/app/static/valid_fake_0004634.png",
            "http://localhost:8508/app/static/valid_fake_0006713.png",
            "http://localhost:8508/app/static/valid_real_0009792.png",
            "http://localhost:8508/app/static/valid_real_0010329.png",
            "http://localhost:8508/app/static/valid_real_0011665.png",
        ],
        titles=[f"Image #{str(i)}" for i in range(5)],
        div_style={
            "display": "flex",
            "justify-content": "center",
            "flex-wrap": "wrap",
        },
        img_style={"margin": "5px", "height": "200px"},
    )
    # Display the uploaded or clickable images
    if uploaded_file is not None:
        # Open the uploaded image
        st.toast("Image uploaded", icon="✅")
        image = Image.open(uploaded_file)
    elif clicked > -1:
        # If an image is clicked, load it from the URL
        image_url = [
            "https://images.unsplash.com/photo-1565130838609-c3a86655db61?w=700",
            "https://images.unsplash.com/photo-1565372195458-9de0b320ef04?w=700",
            "https://images.unsplash.com/photo-1582550945154-66ea8fff25e1?w=700",
            "https://images.unsplash.com/photo-1591797442444-039f23ddcc14?w=700",
            "https://images.unsplash.com/photo-1518727818782-ed5341dbd476?w=700",
        ][clicked]
        image = Image.open(
            requests.get(image_url, stream=True).raw
        )  # Load the clicked image

if image:
    c2.image(image, caption="Selected Image", use_container_width=True)

    c = st.columns([8, 2])
    # Button to process the selected image
    if c[-1].button("Process Image"):
        pred, processed_image = check_image(image)

        # Show a modal with the processed image
        with st.expander("Processed Image"):
            st.image(
                processed_image, caption="Processed Image", use_container_width=True
            )
            st.success("Processing complete!")
