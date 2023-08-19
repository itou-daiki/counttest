import streamlit as st
import cv2
import numpy as np
from PIL import Image

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

st.title("画像から人数を数える")

uploaded_file = st.file_uploader("画像をアップロードしてください", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    processed_image, num_faces = detect_faces(image_np)

    st.write(f"検出された人数: {num_faces}")
    st.image(processed_image, channels="BGR", use_column_width=True)
