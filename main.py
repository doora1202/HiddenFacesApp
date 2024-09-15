import numpy as np
import cv2
import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
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
    fore_img = cv2.imread("face.png", -1)  # アルファチャンネルを含む画像を読み込む
    face_detector.setInputSize((back_img.shape[1], back_img.shape[0]))
    _, faces = face_detector.detect(back_img)
    faces = faces if faces is not None else []
    
    for face in faces:
        box = list(map(int, face[:4]))
        box_w, box_h = box[2], box[3]
        fore_size = max(box_w, box_h)
        fore_img_resized = cv2.resize(fore_img, (fore_size, fore_size))
        alpha_f = fore_img_resized[:, :, 3] / 255.0
        alpha_b = 1.0 - alpha_f

        # 合成する領域を指定する
        x1 = max(box[0], 0)
        y1 = max(box[1], 0)
        x2 = min(x1 + fore_size, back_img.shape[1])  # 元画像の幅を超えないように調整
        y2 = min(y1 + fore_size, back_img.shape[0])  # 元画像の高さを超えないように調整

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
            back_img[y1:y2, x1:x2, c] = (alpha_f_crop * fore_img_crop[:, :, c] +
                                         alpha_b_crop * back_img[y1:y2, x1:x2, c])
    
    if len(faces) == 0:
        return {"message": "顔が検出されませんでした。"}
    
    # 画像をバイトデータに変換してレスポンスとして返す
    _, img_encoded = cv2.imencode('.png', back_img)
    return StreamingResponse(io.BytesIO(img_encoded), media_type="image/png")

def read_image(bin_data):
    file_bytes = np.asarray(bytearray(bin_data.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img
