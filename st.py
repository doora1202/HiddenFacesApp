import streamlit as st
import requests
import io
import cv2
import numpy as np
import os

# Streamlitアプリの設定
st.set_page_config(
    page_title="Hidden Faces App",
    page_icon=":smiley:",
    layout="centered"
)
st.title("Hidden Faces App")

url = os.environ.get("RENDER_URL") or 'http://localhost:8000'

# 画像ファイルのアップロード
uploaded_file = st.file_uploader("画像ファイルをアップロードしてください", type=["png", "jpg", "jpeg"])

# 画像を送信して加工された画像を受け取る
if uploaded_file is not None:
    with st.spinner('顔を検出しています...'):
        files = {"file": (uploaded_file.name, uploaded_file, "multipart/form-data")}
        response = requests.post(f"{url}/detect-face/", files=files)
        if response.ok:
            if response.headers['content-type'] == 'image/png':
                st.success("顔を隠しました。")
                # OpenCVで画像を開く
                image = np.asarray(bytearray(response.content), dtype=np.uint8)
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                st.image(image, use_column_width=True, caption='Processed Image')
                image_bytes = io.BytesIO(response.content)
                st.download_button(label="Download", data=image_bytes, file_name='processed_image.png', mime='image/png')
            else:
                st.error("顔が検出されませんでした。")
        else:
            st.error("エラーが発生しました。")
