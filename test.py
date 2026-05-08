from fastapi import FastAPI, UploadFile, File, HTTPException
import os
import requests
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
from fastapi.responses import Response
import cv2
app = FastAPI()

MODEL_URL = "https://storage.googleapis.com/models-bucket25225/models/best.onnx"
MODEL_PATH = "model.onnx"

session = None


# ---------- Load model ----------
@app.on_event("startup")
def load_model():
    global session

    if not MODEL_URL:
        raise RuntimeError("MODEL_URL not provided")
    if not os.path.exists(MODEL_PATH):
        print(f"[INFO] Downloading model from {MODEL_URL}")

        try:
            r = requests.get(MODEL_URL, timeout=30)
            r.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Failed to download model: {e}")

        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
    else:
        print("[INFO] Model already exists, using cached model")
    print("[INFO] Loading ONNX model...")
    session = ort.InferenceSession(MODEL_PATH)

    print("[INFO] Model loaded successfully")


# ---------- Health ----------
@app.get("/")
def health():
    return {"status": "ok"}


# ---------- Preprocess ----------
def preprocess(image: Image.Image):
    image = image.resize((640, 640))

    img = np.array(image).astype(np.float32)
    img = img / 255.0

    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return img

# ---------- Inference ----------
@app.post("/predict")
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    img = np.array(image)
    h, w, _ = img.shape

    # 🔥 run model
    input_tensor = preprocess(image)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})

    pred = outputs[0][0]

    # 🔥 draw boxes (basic)
    for row in pred:
        conf = row[4]
        if conf > 0.5:
            x, y, bw, bh = row[:4]

            x1 = int((x - bw / 2) * w)
            y1 = int((y - bh / 2) * h)
            x2 = int((x + bw / 2) * w)
            y2 = int((y + bh / 2) * h)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    _, buffer = cv2.imencode(".jpg", img)
    return Response(content=buffer.tobytes(), media_type="image/jpeg")
