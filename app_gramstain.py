from pathlib import Path

import streamlit as st
from PIL import Image
from ultralytics import YOLO

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "best_gramstain.pt"
LOGO_PATH = BASE_DIR / "images" / "i3LUniversity.png"

st.set_page_config(
    page_title="i3L AI System",
    layout="wide",
    initial_sidebar_state="auto",
)


@st.cache_resource
def load_model(path):
    return YOLO(str(path))


st.image(Image.open(LOGO_PATH))
st.title("i3L AI-based Gram Staining Detection")

model = load_model(MODEL_PATH)

source_img = st.file_uploader("Choose an image...", type=("jpg", "jpeg", "png"))

if st.button("Detect"):
    if source_img is None:
        st.warning("Please upload an image first.")
    else:
        img = Image.open(source_img).convert("RGB")
        res = model.predict(img)
        res_plotted = res[0].plot()[:, :, ::-1]
        st.image(res_plotted, caption="Detected Image", width="stretch")
