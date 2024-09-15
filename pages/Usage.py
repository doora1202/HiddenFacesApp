import streamlit as st
import requests
import io
import cv2
import numpy as np
from PIL import Image

# Streamlitアプリの設定
st.set_page_config(
    page_title="Hidden Faces App - 使い方",
    page_icon=":smiley:",
    layout="wide"
)

with st.sidebar:
    "[View the source code](https://github.com/doora1202/HiddenFacesApp)"

# タイトルと説明
st.title("使い方 - Hidden Faces App")
st.write("""
Hidden Faces Appは、顔が写った画像をアップロードすると、顔の部分をスタンプで隠してくれるアプリです。

""")


# 実際の画像を表示（Before and After）
st.subheader("実際の画像")

# サンプル画像の表示（Before）
st.write("オリジナルの画像:")
sample_image_before = Image.open("test.jpg")  # ここで事前にサンプル画像を用意しておく
st.image(sample_image_before, caption="オリジナルの画像", use_column_width=True)

# サンプル画像の表示（After）
st.write("顔が隠された画像:")
sample_image_after = Image.open("test_after.jpg")  # スタンプで隠した画像を用意しておく
st.image(sample_image_after, caption="顔が隠された画像", use_column_width=True)

