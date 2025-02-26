import cv2
import torch
import numpy as np
import streamlit as st
from PIL import Image

import pathlib
pathlib.PosixPath = pathlib.WindowsPath

@st.cache_resource
def load_model():
    # Custom YOLOv5 model load
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
    model.eval()
    return model

def predict(frame, model):
    # Run prediction
    results = model(frame)
    return results

def draw_boxes(results, frame):
    """
    Draw bounding boxes with different colors for each class.
    """
    COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]  # Color palette
    
    for *xyxy, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, xyxy)
        color = COLORS[int(cls) % len(COLORS)]  # Assign a color to each class
        label = f"{results.names[int(cls)]} {conf:.2f}"

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)  # Thicker box
        
        # Draw label with a transparent background
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(label, font, 0.6, 1)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1 - text_height - baseline), (x1 + text_width, y1), color, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)  # Transparency effect
        cv2.putText(frame, label, (x1, y1 - baseline), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


st.title("👚 이건 무슨 옷이야? 🩳")


model = load_model()
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    with st.spinner("Processing..."):
       
        pil_image = Image.open(uploaded_file).convert("RGB")
        np_image = np.array(pil_image)
        cv_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

      
        results = predict(cv_image, model)
        
        image_with_boxes = draw_boxes(results, cv_image)
        
        st.image(image_with_boxes, channels="BGR")

