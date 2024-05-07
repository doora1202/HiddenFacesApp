# 顔隠しアプリ

このアプリは、画像に写っている顔を検出し、顔を隠すことができるツールです。

## 1. 環境構築

まずは、以下のコマンドを使って必要なパッケージをインストールしてください。

```bash
pip install -r requirements.txt
```

## 2.モデルのインストール

[こちら](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx "url")からモデルをダウンロードしてください。

## 3.スタンプ画像のインストール

[こちら](https://x.gd/p6b4T "url")からスタンプ画像をダウンロードし、face.pngとして保存してください。

## 4. Streamlitの起動

Streamlitアプリを起動するには、以下のコマンドを実行してください。

```bash
streamlit run st.py
```

## 5. FastAPIの起動

FastAPIサーバーを起動するには、以下のコマンドを実行してください。

```bash
uvicorn main:app
```