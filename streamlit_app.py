import streamlit as st
import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import pandas as pd
from datetime import datetime

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

def get_exif_data(image):
    exif_data = image._getexif()
    if exif_data is not None:
        for tag in list(exif_data.keys()):
            tag_name = TAGS.get(tag, tag)
            exif_data[tag_name] = exif_data.pop(tag)
    return exif_data

def get_geotagging(exif):
    if not exif:
        return None

    geotagging = {}
    for (idx, tag) in TAGS.items():
        if tag == 'GPSInfo':
            if idx not in exif:
                return None

            for (key, val) in GPSTAGS.items():
                if key in exif[idx]:
                    geotagging[val] = exif[idx][key]

    return geotagging

def save_to_csv(num_faces):
    # 現在の日時を取得
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 既存のCSVファイルを読み込むか、新しいDataFrameを作成
    try:
        df = pd.read_csv('data.csv')
    except FileNotFoundError:
        df = pd.DataFrame(columns=['Time', 'Number of Faces'])
    
    # 新しいデータを追加
    new_data = pd.DataFrame([{'Time': current_time, 'Number of Faces': num_faces}])
    df = pd.concat([df, new_data], ignore_index=True)
    
    # CSVファイルに保存
    df.to_csv('data.csv', index=False)


st.title("顔検出アプリ")
st.write("画像をアップロードすると、その画像から顔を検出し、検出された顔の数とExifデータを表示します。")

uploaded_file = st.file_uploader("画像をアップロードしてください", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    st.write("アップロードされた画像：")
    image = Image.open(uploaded_file)
    st.image(image, caption="アップロードされた画像", use_column_width=True)

    image_np = np.array(image)
    processed_image, num_faces = detect_faces(image_np)

    st.write(f"検出された人数: {num_faces}")
    st.image(processed_image, caption="顔検出後の画像", channels="BGR", use_column_width=True)

    # Exifデータの取得
    exif_data = get_exif_data(image)
    if exif_data:
        capture_time = exif_data.get("DateTimeOriginal", None)
        if capture_time:
            st.write(f"撮影日時: {capture_time}")

        geotags = get_geotagging(exif_data)
        if geotags:
            st.write(f"撮影場所のGPS情報: {geotags}")
        else:
            st.write("ExifデータにGPS情報は見つかりませんでした。")
    else:
        st.write("Exifデータが見つかりませんでした。")

    # 検出された人数をCSVに保存
    save_to_csv(num_faces)
