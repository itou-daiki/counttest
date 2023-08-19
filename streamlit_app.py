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

# タイトルと説明を追加
st.title("顔検出アプリ")
st.write("画像をアップロードすると、その画像から顔を検出し、検出された顔の数を表示します。")

# 画像アップロードの部分
uploaded_file = st.file_uploader("画像をアップロードしてください", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    st.write("アップロードされた画像：")
    image = Image.open(uploaded_file)
    st.image(image, caption="アップロードされた画像", use_column_width=True)

    image_np = np.array(image)
    processed_image, num_faces = detect_faces(image_np)

    st.write(f"検出された人数: {num_faces}")
    st.image(processed_image, caption="顔検出後の画像", channels="BGR", use_column_width=True)
