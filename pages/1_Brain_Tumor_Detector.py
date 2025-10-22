import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

model = YOLO('../models/brain_tumor_model.pt')

st.title("Brain Tumor Detector")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    pil_img = Image.open(uploaded_file)
    img = np.array(pil_img)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    results = model.predict(source=img_bgr, conf=0.25)

    annotated_img = results[0].plot()

    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    st.image(annotated_img, caption="Detected Tumors", use_column_width=True)