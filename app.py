import os
import gradio as gr
import numpy as np
from tensorflow import keras
from PIL import Image
from dotenv import load_dotenv
from api.twilio_service import send_sms_diagnosis

load_dotenv()

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "cnn_model.h5")

CLASSES = ["lung_n", "lung_aca", "lung_scc", "colon_n", "colon_aca"]
CLASS_LABELS = {
    "lung_n": "Lung Benign Tissue",
    "lung_aca": "Lung Adenocarcinoma",
    "lung_scc": "Lung Squamous Cell Carcinoma",
    "colon_n": "Colon Benign Tissue",
    "colon_aca": "Colon Adenocarcinoma",
}
IMG_SIZE = (160, 160)

model = keras.models.load_model(MODEL_PATH)


def preprocess_image(image):
    if isinstance(image, np.ndarray):
        img = Image.fromarray(image)
    else:
        img = image
    img = img.convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img)
    img_array = keras.applications.mobilenet_v2.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)


def predict_image(image, send_sms=False, phone_number=None):
    if image is None:
        return "Please upload an image", {}, "No SMS sent"

    preprocessed = preprocess_image(image)
    predictions = model.predict(preprocessed)[0]

    class_idx = np.argmax(predictions)
    confidence = float(predictions[class_idx])
    diagnosis = CLASS_LABELS[CLASSES[class_idx]]

    all_predictions = {
        CLASS_LABELS[CLASSES[i]]: float(predictions[i]) for i in range(len(CLASSES))
    }

    result_text = f"Diagnosis: {diagnosis}\nConfidence: {confidence:.2%}"

    sms_status = "No SMS sent"
    if send_sms:
        sms_result = send_sms_diagnosis(diagnosis, confidence, phone_number)
        if sms_result["success"]:
            sms_status = f"SMS sent successfully (SID: {sms_result['message_sid']})"
        else:
            sms_status = f"SMS failed: {sms_result.get('error', 'Unknown error')}"

    return result_text, all_predictions, sms_status


def create_app():
    with gr.Blocks(title="Disease Detector") as app:
        gr.Markdown("# Disease Detector - Histopathological Image Classification")
        gr.Markdown(
            "Upload a histopathological image to detect lung or colon cancer types."
        )

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")
                send_sms_checkbox = gr.Checkbox(
                    label="Send SMS Notification", value=False
                )
                phone_input = gr.Textbox(
                    label="Phone Number (optional, uses default if empty)",
                    placeholder="+15558675309",
                )
                predict_btn = gr.Button("Predict", variant="primary")

            with gr.Column():
                result_text = gr.Textbox(label="Diagnosis Result", lines=3)
                predictions_label = gr.Label(label="All Predictions", num_top_classes=5)
                sms_status = gr.Textbox(label="SMS Status", lines=2)

        predict_btn.click(
            fn=predict_image,
            inputs=[image_input, send_sms_checkbox, phone_input],
            outputs=[result_text, predictions_label, sms_status],
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(share=False)
