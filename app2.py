import cv2
import av
import torch
import numpy as np
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# 모델 로드
model = YOLO("best.pt")  

class YOLOVideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")  # 프레임을 OpenCV 형식으로 변환


        results = model(img, conf=0.5)

        # 결과 그리기
        if results and results[0]:
            img = results[0].plot()

        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("👚 이건 무슨 옷이야? 🩳")
st.write("📹웹캠을 통해 실시간으로 지금 입고 있는 옷의 종류를 알려드릴게요.")


webrtc_streamer(key="yolo", video_processor_factory=YOLOVideoProcessor)
