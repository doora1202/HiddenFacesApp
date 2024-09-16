import numpy as np
import cv2
import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORSの設定を追加
origins = [
    "https://hiddenfacesapp.streamlit.app",
    "http://localhost:8501"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 許可するオリジンを指定
    allow_credentials=True,
    allow_methods=["*"],  # すべてのHTTPメソッドを許可
    allow_headers=["*"],  # すべてのヘッダーを許可
)

@app.post("/detect-face/")
async def process_image(file: UploadFile = File(...)):
    # 顔検出
    face_detector = cv2.FaceDetectorYN_create(
        "face_detection_yunet_2023mar.onnx", "", (0, 0), 0.6, 0.3, 5000, cv2.dnn.DNN_BACKEND_DEFAULT, cv2.dnn.DNN_TARGET_CPU
    )
    bin_data = io.BytesIO(await file.read())  # ファイルのデータを取得するためにread()を使う
    back_img = read_image(bin_data)

    # 画像のサイズを顔検出器に設定
    face_detector.setInputSize((back_img.shape[1], back_img.shape[0]))

    # 顔検出
    _, faces = face_detector.detect(back_img)

    # 顔検出結果がNoneの場合は空リストにする
    faces = faces if faces is not None else []

    if len(faces) == 0:
        return JSONResponse(content={"message": "顔が検出されませんでした。"}, status_code=200)
    else:
        # facesをリスト形式に変換
        faces_list = faces.astype(float).tolist()
        print(f"Faces detected: {faces_list}")
        return JSONResponse(content={"faces": faces_list}, status_code=200)

def read_image(bin_data):
    file_bytes = np.asarray(bytearray(bin_data.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img
