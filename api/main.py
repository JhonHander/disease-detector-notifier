import os
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow import keras
from PIL import Image
from api.twilio_service import send_sms_diagnosis

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "model", "cnn_model.h5"
)

CLASSES = ["lung_n", "lung_aca", "lung_scc", "colon_n", "colon_aca"]
CLASS_LABELS = {
    "lung_n": "Lung Benign Tissue",
    "lung_aca": "Lung Adenocarcinoma",
    "lung_scc": "Lung Squamous Cell Carcinoma",
    "colon_n": "Colon Benign Tissue",
    "colon_aca": "Colon Adenocarcinoma",
}
IMG_SIZE = (224, 224)

ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading CNN model...")
    ml_models["cnn"] = keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully")
    yield
    ml_models.clear()


app = FastAPI(title="Disease Detector API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(image_bytes).convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image)
    img_array = keras.applications.mobilenet_v2.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    preprocessed = preprocess_image(image_bytes)

    model = ml_models["cnn"]
    predictions = model.predict(preprocessed)[0]

    class_idx = np.argmax(predictions)
    confidence = float(predictions[class_idx])
    diagnosis = CLASS_LABELS[CLASSES[class_idx]]

    all_predictions = {
        CLASS_LABELS[CLASSES[i]]: float(predictions[i]) for i in range(len(CLASSES))
    }

    return {
        "diagnosis": diagnosis,
        "confidence": confidence,
        "all_predictions": all_predictions,
    }


@app.post("/predict-and-notify")
async def predict_and_notify(file: UploadFile = File(...), phone: str = None):
    image_bytes = await file.read()
    preprocessed = preprocess_image(image_bytes)

    model = ml_models["cnn"]
    predictions = model.predict(preprocessed)[0]

    class_idx = np.argmax(predictions)
    confidence = float(predictions[class_idx])
    diagnosis = CLASS_LABELS[CLASSES[class_idx]]

    all_predictions = {
        CLASS_LABELS[CLASSES[i]]: float(predictions[i]) for i in range(len(CLASSES))
    }

    sms_result = send_sms_diagnosis(diagnosis, confidence, phone)

    return {
        "diagnosis": diagnosis,
        "confidence": confidence,
        "all_predictions": all_predictions,
        "sms_sent": sms_result["success"],
        "sms_error": sms_result.get("error"),
    }
