import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile

# OpenCVの顔検出用のカスケード分類器をロード
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(img):
    # 画像をグレースケールに変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 顔の検出
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # 顔の位置に矩形を描画
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return img, len(faces)

st.title("カメラからの映像で人数を数える")

uploaded_file = st.file_uploader("映像をアップロードしてください", type=['mp4', 'mov', 'avi'])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    st.video(tfile.name)

    cap = cv2.VideoCapture(tfile.name)

    if cap.isOpened():
        frame_list = []
        total_faces = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame, num_faces = detect_faces(frame)
            total_faces += num_faces
            frame_list.append(frame)

        st.write(f"検出された人数: {total_faces}")

        for frame in frame_list:
            st.image(frame, channels="BGR", use_column_width=True)
    else:
        st.write("動画の読み込みに失敗しました。")
