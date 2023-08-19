import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return frame, len(faces)

def app():
    st.title("リアルタイムでの顔検出")
    webrtc_streamer(key="example", video_transformer=detect_faces)

if __name__ == "__main__":
    app()
