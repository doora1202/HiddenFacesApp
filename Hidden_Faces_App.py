import streamlit as st
import requests
import cv2
import numpy as np

# Streamlitアプリの設定
st.set_page_config(
    page_title="Hidden Faces App",
    page_icon=":smiley:",
    layout="centered"
)
st.title("Hidden Faces App")

with st.sidebar:
    "[View the source code](https://github.com/doora1202/HiddenFacesApp)"

url = 'https://total-adiana-doora-74d386f3.koyeb.app/'

def resize_to_min_500px(image):
    # 元の画像のサイズを取得
    original_height, original_width = image.shape[:2]

    # 縦横の短い方を500ピクセルにする
    if original_width < original_height:
        scale_factor = 500 / original_width
    else:
        scale_factor = 500 / original_height

    # 新しいサイズを計算
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    # 画像をリサイズ
    resized_image = cv2.resize(image, (new_width, new_height))

    return resized_image, new_width, new_height, scale_factor

# 画像ファイルのアップロード
back_img = st.file_uploader("画像ファイルをアップロードしてください", type=["png", "jpg", "jpeg"])

if back_img is not None:
    # アップロードされたファイルをNumPy配列に変換
    file_bytes = np.asarray(bytearray(back_img.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)  # カラーチャンネル付きで読み込み
    
    resized_image, new_width, new_height, scale_factor = resize_to_min_500px(img)

    fore_img = cv2.imread("face.png", -1)  # アルファチャンネルを含む画像を読み込む

    # リサイズ画像を送信してbboxを取得
    with st.spinner('顔を検出しています...'):
        # 画像をエンコードしてバイナリ形式にする
        _, buffer = cv2.imencode('.jpg', resized_image)
        buffer_bytes = buffer.tobytes()

        # リクエストに送信するファイルの形式に変換
        files = {"file": (back_img.name, buffer_bytes, "image/jpeg")}  # back_img.nameを使用
        
        response = requests.post(f"{url}/detect-face/", files=files)
        if response.ok:
            response_data = response.json()
            faces = response_data.get("faces", [])
            if len(faces) == 0:
                st.error("顔が検出されませんでした。")
            else:
                for face in faces:
                    box = (np.array(face[:4], dtype=float) / scale_factor).astype(int)
                    box_w, box_h = box[2], box[3]
                    fore_size = max(box_w, box_h)
                    fore_img_resized = cv2.resize(fore_img, (fore_size, fore_size))
                    alpha_f = fore_img_resized[:, :, 3] / 255.0
                    alpha_b = 1.0 - alpha_f

                    # 合成する領域を指定する
                    x1 = max(box[0], 0)
                    y1 = max(box[1], 0)
                    x2 = min(x1 + fore_size, img.shape[1])  # 元画像の幅を超えないように調整
                    y2 = min(y1 + fore_size, img.shape[0])  # 元画像の高さを超えないように調整

                    # fore_img_resized の範囲を制限する
                    fore_x1 = 0 if x1 >= 0 else abs(x1)
                    fore_y1 = 0 if y1 >= 0 else abs(y1)
                    fore_x2 = fore_x1 + (x2 - x1)
                    fore_y2 = fore_y1 + (y2 - y1)

                    # fore_img_resized の合成範囲を切り取る
                    fore_img_crop = fore_img_resized[fore_y1:fore_y2, fore_x1:fore_x2]
                    alpha_f_crop = alpha_f[fore_y1:fore_y2, fore_x1:fore_x2]
                    alpha_b_crop = alpha_b[fore_y1:fore_y2, fore_x1:fore_x2]

                    # 合成する
                    for c in range(0, 3):
                        img[y1:y2, x1:x2, c] = (alpha_f_crop * fore_img_crop[:, :, c] +
                                                alpha_b_crop * img[y1:y2, x1:x2, c])
                # 表示用にBGRからRGBに変換
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, use_column_width=True, caption='Processed Image')

                # ダウンロード用にBGR形式のままエンコード
                _, buffer = cv2.imencode('.jpg', img)  # BGRのまま
                buffer_bytes = buffer.tobytes()
                st.download_button(label="Download", data=buffer_bytes, file_name='processed_image.jpg', mime='image/jpg')
        else:
            st.error("エラーが発生しました。")
