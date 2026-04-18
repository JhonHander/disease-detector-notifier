import os
import json
import gradio as gr
import numpy as np
from tensorflow import keras
from PIL import Image
from dotenv import load_dotenv
from api.twilio_service import send_sms_diagnosis, get_twilio_config_status

load_dotenv()

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "cnn_model.h5")
RUNS_DIR = os.path.join(os.path.dirname(__file__), "model", "training_runs")

CLASSES = ["lung_n", "lung_aca", "lung_scc", "colon_n", "colon_aca"]
CLASS_LABELS = {
    "lung_n": "Lung Benign Tissue",
    "lung_aca": "Lung Adenocarcinoma",
    "lung_scc": "Lung Squamous Cell Carcinoma",
    "colon_n": "Colon Benign Tissue",
    "colon_aca": "Colon Adenocarcinoma",
}
IMG_SIZE = (160, 160)

model = keras.models.load_model(
    MODEL_PATH,
    compile=False,
    custom_objects={
        "preprocess_input": keras.applications.mobilenet_v2.preprocess_input,
    },
)


CUSTOM_CSS = """
:root {
    --bg: #f8f5ef;
    --card: #ffffff;
    --ink: #1f2937;
    --muted: #64748b;
    --brand: #0f766e;
    --brand-soft: #ccfbf1;
    --accent: #b45309;
}

.gradio-container {
    background:
        radial-gradient(circle at 8% 12%, #ecfeff 0%, transparent 35%),
        radial-gradient(circle at 88% 20%, #fff7ed 0%, transparent 35%),
        var(--bg);
    font-family: "Manrope", "IBM Plex Sans", "Segoe UI", sans-serif;
}

.hero-card {
    background: linear-gradient(120deg, #115e59 0%, #134e4a 45%, #713f12 100%);
    border-radius: 20px;
    color: #f8fafc;
    padding: 22px;
    box-shadow: 0 18px 40px rgba(15, 23, 42, 0.2);
    margin-bottom: 8px;
}

.hero-card h1 {
    margin: 0 0 8px 0;
    font-size: 2rem;
    letter-spacing: 0.5px;
}

.hero-card p {
    margin: 0;
    opacity: 0.95;
}

.panel {
    background: var(--card);
    border-radius: 16px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.07);
}

.pill {
    display: inline-block;
    border: 1px solid rgba(255, 255, 255, 0.35);
    border-radius: 999px;
    padding: 4px 12px;
    margin-right: 8px;
    font-size: 0.8rem;
}
"""


def preprocess_image(image):
    if isinstance(image, np.ndarray):
        img = Image.fromarray(image)
    else:
        img = image
    img = img.convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    return np.expand_dims(img_array, axis=0)


def list_training_runs():
    if not os.path.isdir(RUNS_DIR):
        return []
    runs = [
        name
        for name in os.listdir(RUNS_DIR)
        if os.path.isdir(os.path.join(RUNS_DIR, name))
    ]
    return sorted(runs, reverse=True)


def _read_text_if_exists(path):
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _read_json_if_exists(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_run_dashboard(run_id):
    if not run_id:
        return "No training run selected.", [], None, []

    run_path = os.path.join(RUNS_DIR, run_id)
    if not os.path.isdir(run_path):
        return f"Run not found: {run_id}", [], None, []

    report_txt = _read_text_if_exists(os.path.join(run_path, "training_report.txt"))
    error_txt = _read_text_if_exists(os.path.join(run_path, "error_summary.txt"))
    class_json = _read_json_if_exists(os.path.join(run_path, "classification_report.json"))

    summary_md = f"## Run {run_id}\n\n"
    if report_txt:
        summary_md += "### Training Report\n"
        summary_md += f"```text\n{report_txt.strip()}\n```\n"
    if error_txt:
        summary_md += "\n### Error Summary\n"
        summary_md += f"```text\n{error_txt.strip()}\n```\n"

    rows = []
    for class_name in CLASSES:
        row = class_json.get(class_name, {})
        if row:
            rows.append(
                [
                    class_name,
                    round(float(row.get("precision", 0.0)), 4),
                    round(float(row.get("recall", 0.0)), 4),
                    round(float(row.get("f1-score", 0.0)), 4),
                    int(row.get("support", 0)),
                ]
            )

    confusion_img = os.path.join(run_path, "confusion_matrix.png")
    if not os.path.exists(confusion_img):
        confusion_img = None

    artifact_files = []
    for name in [
        "run_config.json",
        "split_manifest.json",
        "train_files.txt",
        "validation_files.txt",
        "test_files.txt",
        "training_report.txt",
        "phase_1_metrics.csv",
        "phase_2_metrics.csv",
        "classification_report.json",
        "classification_report.txt",
        "confusion_matrix.csv",
        "confusion_matrix.png",
        "error_summary.txt",
        "training_curves.png",
    ]:
        path = os.path.join(run_path, name)
        if os.path.exists(path):
            artifact_files.append(path)

    return summary_md, rows, confusion_img, artifact_files


def refresh_run_choices():
    runs = list_training_runs()
    default = runs[0] if runs else None
    return gr.Dropdown(choices=runs, value=default)


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
            sms_status = (
                f"SMS sent successfully\n"
                f"SID: {sms_result.get('message_sid', 'N/A')}\n"
                f"Status: {sms_result.get('status', 'queued')}"
            )
        else:
            sms_status = f"SMS failed: {sms_result.get('error', 'Unknown error')}"

    return result_text, all_predictions, sms_status


def send_test_sms(phone_number):
    result = send_sms_diagnosis(
        "Test diagnosis from Disease Detector",
        0.99,
        phone_number,
    )
    if result["success"]:
        return (
            f"Twilio test SMS sent\n"
            f"SID: {result.get('message_sid', 'N/A')}\n"
            f"Status: {result.get('status', 'queued')}"
        )
    return f"Twilio test SMS failed: {result.get('error', 'Unknown error')}"


def create_app():
    runs = list_training_runs()
    default_run = runs[0] if runs else None
    twilio_status = get_twilio_config_status()

    with gr.Blocks(title="Disease Detector", css=CUSTOM_CSS) as app:
        gr.HTML(
            """
            <section class='hero-card'>
              <span class='pill'>Deep Learning CNN</span>
              <span class='pill'>Gradio Frontend</span>
              <span class='pill'>Twilio SMS</span>
              <h1>Disease Detector Console</h1>
              <p>Clasificacion histopatologica de pulmon y colon con analitica de entrenamiento y notificaciones SMS.</p>
            </section>
            """
        )

        with gr.Tabs():
            with gr.TabItem("Diagnostico"):
                with gr.Row():
                    with gr.Column(scale=5):
                        image_input = gr.Image(type="pil", label="Imagen Histopatologica")
                        with gr.Row():
                            send_sms_checkbox = gr.Checkbox(
                                label="Enviar resultado por SMS", value=False
                            )
                            phone_input = gr.Textbox(
                                label="Telefono destino (E.164)",
                                placeholder="+573001234567",
                            )
                        predict_btn = gr.Button("Generar Diagnostico", variant="primary")

                    with gr.Column(scale=5):
                        result_text = gr.Textbox(label="Resultado", lines=3)
                        predictions_label = gr.Label(
                            label="Probabilidades por clase",
                            num_top_classes=5,
                        )
                        sms_status = gr.Textbox(label="Estado SMS", lines=3)

                predict_btn.click(
                    fn=predict_image,
                    inputs=[image_input, send_sms_checkbox, phone_input],
                    outputs=[result_text, predictions_label, sms_status],
                )

            with gr.TabItem("Analitica de Entrenamiento"):
                with gr.Row():
                    run_selector = gr.Dropdown(
                        choices=runs,
                        value=default_run,
                        label="Selecciona una corrida",
                    )
                    refresh_runs_btn = gr.Button("Actualizar Corridas")

                run_summary_md = gr.Markdown()
                class_table = gr.Dataframe(
                    headers=["Class", "Precision", "Recall", "F1", "Support"],
                    label="Metricas por clase",
                    interactive=False,
                )
                confusion_image = gr.Image(
                    label="Matriz de confusion",
                    type="filepath",
                )
                artifacts_files = gr.File(
                    label="Artefactos de la corrida",
                    file_count="multiple",
                )

                run_selector.change(
                    fn=load_run_dashboard,
                    inputs=[run_selector],
                    outputs=[run_summary_md, class_table, confusion_image, artifacts_files],
                )

                refresh_runs_btn.click(
                    fn=refresh_run_choices,
                    outputs=[run_selector],
                )

                app.load(
                    fn=load_run_dashboard,
                    inputs=[run_selector],
                    outputs=[run_summary_md, class_table, confusion_image, artifacts_files],
                )

            with gr.TabItem("Twilio"):
                gr.Markdown(
                    """
### Estado de Configuracion
Se valida la presencia de credenciales y numeros en variables de entorno.
"""
                )
                twilio_box = gr.Textbox(
                    value=twilio_status,
                    label="Diagnostico Twilio",
                    lines=8,
                    interactive=False,
                )
                test_phone = gr.Textbox(
                    label="Telefono para SMS de prueba (opcional)",
                    placeholder="+573001234567",
                )
                send_test_btn = gr.Button("Enviar SMS de Prueba", variant="secondary")
                test_result = gr.Textbox(label="Resultado prueba SMS", lines=4)

                send_test_btn.click(
                    fn=send_test_sms,
                    inputs=[test_phone],
                    outputs=[test_result],
                )

                gr.Markdown(
                    """
### Flujo recomendado
1. Configura `.env` con `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_PHONE_NUMBER` y opcionalmente `DESTINATION_PHONE_NUMBER`.
2. Si tu cuenta es trial, verifica el numero destino en Twilio Console.
3. Usa formato E.164 (`+57...`) para todos los telefonos.
4. Prueba con el boton de SMS antes de activar notificaciones automaticas en Diagnostico.
"""
                )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(share=False)
