# 必要なライブラリをインポート
import streamlit as st
import cv2
import numpy as np
from PIL import Image, ExifTags
import pandas as pd
from datetime import datetime
import pytz

# OpenCVの顔検出用のカスケード分類器をロード
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Streamlitのタイトルを設定
st.title("顔検出アプリ")
st.write("画像をアップロードすると、その画像から顔を検出し、検出された顔の数とExifデータを表示します。")

# Streamlitのファイルアップローダーを使用して画像をアップロード
uploaded_file = st.file_uploader("画像をアップロードしてください", type=['jpg', 'jpeg', 'png'])

# 画像がアップロードされた場合の処理
if uploaded_file:
    # PILを使用して画像を開く
    image = Image.open(uploaded_file)
    # Streamlitでアップロードされた画像を表示
    st.image(image, caption="アップロードされた画像", use_column_width=True)

    # 画像をNumPy配列に変換
    image_np = np.array(image)
    # 画像をグレースケールに変換
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    # 顔を検出
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    # 検出された顔の位置に矩形を描画
    for (x, y, w, h) in faces:
        cv2.rectangle(image_np, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # 検出された顔の数を表示
    st.write(f"検出された人数: {len(faces)}")
    # 顔が検出された後の画像を表示
    st.image(image_np, caption="顔検出後の画像", channels="BGR", use_column_width=True)

    # Exifデータを取得
    exif_raw = image._getexif()
    if exif_raw:
        exif_data = {ExifTags.TAGS[k]: v for k, v in exif_raw.items() if k in ExifTags.TAGS and isinstance(v, (str, bytes))}
    else:
        exif_data = {}

    # 撮影日時を取得
    capture_time = exif_data.get("DateTimeOriginal", None)
    # GPS情報を取得
    geotags = None
    if 'GPSInfo' in exif_data:
        geotags = {ExifTags.GPSTAGS[k]: v for k, v in exif_data['GPSInfo'].items() if k in ExifTags.GPSTAGS}

    # 撮影日時とGPS情報を表示
    if capture_time:
        st.write(f"撮影日時: {capture_time}")
    if geotags:
        st.write(f"撮影場所のGPS情報: {geotags}")
    else:
        st.write("ExifデータにGPS情報は見つかりませんでした。")

    # 検出された顔の数をCSVファイルに保存
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
    # CSVダウンロードボタンを追加
    st.download_button("CSVをダウンロード", data=open('data.csv', 'rb').read(), file_name="data.csv", mime="text/csv")
