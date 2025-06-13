import streamlit as st
import torch
from PIL import Image
import os
import tempfile
import numpy as np
import cv2

CLASS_NAMES = ['o to', 'xe bus', 'xe dap', 'xe may', 'xe tai']
VIOLATION_CLASSES = ['xe may', 'xe dap' , 'xe bus']

@st.cache_resource(show_spinner=True)
def load_model():
    return torch.hub.load('yolov5', 'custom', path='/content/yolov5/runs/train/exp/weights/best.pt', source='local')

model = load_model()

st.title("🚦 Nhận diện phương tiện vi phạm giao thông")
st.write("Tải ảnh lên để nhận diện: o to, xe bus, xe dap, xe may, xe tai")

uploaded_file = st.file_uploader("📤 Chọn ảnh...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Ảnh gốc', use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        tmp_path = tmp_file.name
        image.save(tmp_path)

    results = model(tmp_path)
    results.render()

    # Hiển thị ảnh sau khi detect
    result_img = Image.fromarray(cv2.cvtColor(results.ims[0], cv2.COLOR_BGR2RGB))
    st.image(result_img, caption='Kết quả dự đoán', use_column_width=True)

    violated = 0
    detections = results.pandas().xyxy[0]

    st.write("### Kết quả chi tiết:")
    for idx, row in detections.iterrows():
        class_name = row['name']
        conf = row['confidence']
        st.write(f"- Đối tượng {idx+1}: `{class_name}` - Độ tin cậy: {conf:.2f}")

        if class_name in VIOLATION_CLASSES:
            violated += 1

    if violated:
        st.error(f"⛔ Phát hiện {violated} phương tiện vi phạm!")
    else:
        st.success("✅ Không phát hiện vi phạm.")
