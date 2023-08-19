import streamlit as st
import cv2
import numpy as np
from PIL import Image

# OpenCVの顔検出用のカスケード分類器をロード
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(img):
    try:
        # 画像をグレースケールに変換
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 顔の検出
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        # 顔の位置に矩形を描画
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        return img, len(faces)
    except Exception as e:
        st.write(f"エラーが発生しました: {e}")
        return img, 0

st.title("カメラからの映像で人数を数える")

uploaded_file = st.file_uploader("映像をアップロードしてください", type=['mp4', 'mov', 'avi'])

if uploaded_file is not None:
    st.video(uploaded_file)

    # メモリ上で動画を読み込む
    bytes_data = uploaded_file.read()
    nparr = np.frombuffer(bytes_data, np.uint8)
    cap = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if cap is not None:
        frame_list = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame, num_faces = detect_faces(frame)
            frame_list.append(frame)

        st.write(f"検出された人数: {num_faces}")

        for frame in frame_list:
            st.image(frame, channels="BGR", use_column_width=True)
