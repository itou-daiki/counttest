import streamlit as st
import cv2
import numpy as np
from PIL import Image, ExifTags
import pandas as pd
from datetime import datetime
import pytz

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

st.title("顔検出アプリ")
st.write("画像をアップロードすると、その画像から顔を検出し、検出された顔の数とExifデータを表示します。")

uploaded_file = st.file_uploader("画像をアップロードしてください", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="アップロードされた画像", use_column_width=True)

    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image_np, (x, y), (x+w, y+h), (255, 0, 0), 2)
    st.write(f"検出された人数: {len(faces)}")
    st.image(image_np, caption="顔検出後の画像", channels="BGR", use_column_width=True)

    exif_data = {ExifTags.TAGS[k]: v for k, v in image._getexif().items() if k in ExifTags.TAGS and isinstance(v, (str, bytes))}
    capture_time = exif_data.get("DateTimeOriginal", None)
    geotags = None
    if 'GPSInfo' in exif_data:
        geotags = {ExifTags.GPSTAGS[k]: v for k, v in exif_data['GPSInfo'].items() if k in ExifTags.GPSTAGS}

    if capture_time:
        st.write(f"撮影日時: {capture_time}")
    if geotags:
        st.write(f"撮影場所のGPS情報: {geotags}")
    else:
        st.write("ExifデータにGPS情報は見つかりませんでした。")

    current_datetime_jst = datetime.utcnow().astimezone(pytz.timezone('Asia/Tokyo'))
    new_data = pd.DataFrame([{
        'Date': current_datetime_jst.strftime('%Y-%m-%d'),
        'Time': current_datetime_jst.strftime('%H:%M:%S'),
        'Number of Faces': len(faces)
    }])
    try:
        df = pd.read_csv('data.csv')
        df = pd.concat([df, new_data], ignore_index=True)
    except FileNotFoundError:
        df = new_data
    df.to_csv('data.csv', index=False)
    st.download_button("CSVをダウンロード", data=open('data.csv', 'rb').read(), file_name="data.csv", mime="text/csv")
