import streamlit as st
from PIL import Image
import cv2
import numpy as np
from datetime import datetime
import pandas as pd

st.set_page_config(layout="wide")

st.title("🚗 AI Car Damage Detection")

# Resize function
def resize_image(img, max_width=400):
    h, w = img.shape[:2]
    if w > max_width:
        ratio = max_width / w
        new_h = int(h * ratio)
        img = cv2.resize(img, (max_width, new_h))
    return img

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img = np.array(image)

    if st.button("Analyze Image"):

        # Convert
        img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        output = img_cv.copy()

        # Dummy boxes
        cv2.rectangle(output, (150, 80), (350, 230), (0, 255, 0), 2)
        cv2.putText(output, "Scratch", (150, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.rectangle(output, (300, 280), (520, 450), (0, 165, 255), 2)
        cv2.putText(output, "Dent", (300, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # Resize BOTH images
        img_small = resize_image(img_cv, 350)
        output_small = resize_image(output, 350)

        # Convert back
        img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
        output_small = cv2.cvtColor(output_small, cv2.COLOR_BGR2RGB)

        # Show side-by-side
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded")
            st.image(img_small)

        with col2:
            st.subheader("Output")
            st.image(output_small)

        # Info
        st.write("### Summary")
        st.write(f"Time: {datetime.now()}")
        st.write("Damages: 2")

        st.success("Dent → $450")
        st.success("Scratch → $120")
