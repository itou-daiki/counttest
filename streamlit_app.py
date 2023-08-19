import streamlit as st
import cv2
import numpy as np
from PIL import Image, ExifTags
import pandas as pd
from datetime import datetime
import pytz

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return img, len(faces)

def get_exif_data(image):
    return {ExifTags.TAGS[k]: v for k, v in image._getexif().items() if k in ExifTags.TAGS and isinstance(v, (str, bytes))}

def get_geotagging(exif):
    if 'GPSInfo' in exif:
        return {ExifTags.GPSTAGS[k]: v for k, v in exif['GPSInfo'].items() if k in ExifTags.GPSTAGS}
    return None

def save_to_csv(num_faces):
    current_datetime_jst = datetime.utcnow().astimezone(pytz.timezone('Asia/Tokyo'))
    new_data = pd.DataFrame([{
        'Date': current_datetime_jst.strftime('%Y-%m-%d'),
        'Time': current_datetime_jst.strftime('%H:%M:%S'),
        'Number of Faces': num_faces
    }])
    try:
        df = pd.read_csv('data.csv')
        df = pd.concat([df, new_data], ignore_index=True)
    except FileNotFoundError:
        df = new_data
    df.to_csv('data.csv', index=False)

st.title("顔検出アプリ")
st.write("画像をアップロードすると、その画像から顔を検出し、検出された顔の数とExifデータを表示します。")

uploaded_file = st.file_uploader("画像をアップロードしてください", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="アップロードされた画像", use_column_width=True)

    image_np = np.array(image)
    processed_image, num_faces = detect_faces(image_np)
    st.write(f"検出された人数: {num_faces}")
    st.image(processed_image, caption="顔検出後の画像", channels="BGR", use_column_width=True)

    exif_data = get_exif_data(image)
    capture_time = exif_data.get("DateTimeOriginal", None)
    geotags = get_geotagging(exif_data)

    if capture_time:
        st.write(f"撮影日時: {capture_time}")
    if geotags:
        st.write(f"撮影場所のGPS情報: {geotags}")
    else:
        st.write("ExifデータにGPS情報は見つかりませんでした。")

    save_to_csv(num_faces)
    st.download_button("CSVをダウンロード", data=open('data.csv', 'rb').read(), file_name="data.csv", mime="text/csv")
