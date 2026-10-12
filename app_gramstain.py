import cv2
import matplotlib.pyplot as plt
from pathlib import Path

from matplotlib.patches import Rectangle
import tempfile
import PIL
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import requests
import os

st.set_page_config(
    page_title="i3L AI System",
    layout="wide",
    initial_sidebar_state="auto"
)

image = Image.open('images/i3LUniversity.png')

st.image(image)
st.title("i3L AI-based Gram Staining Detection")

classes = ["EscherichiaColi","LactobacillusCasei","NeisseriaGonorrhoeae","StaphylococcusAureus"]

model_path = Path('best_gramstain_02.pt')
model = YOLO(model_path)

source_img = st.file_uploader(
    "Choose an image...", type=("jpg", "jpeg", "png"))

# with col2:
if st.button('Detect'):
    source_img = PIL.Image.open(source_img)
    res = model.predict(source_img)
    # boxes = res[0].boxes
    res_plotted = res[0].plot()[:, :, ::-1]
    st.image(res_plotted, caption='Detected Image', use_container_width =True  )
