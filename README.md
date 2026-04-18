# Disease Detector - Histopathological Image Classification

Aplicacion universitaria que integra:
- Prediccion de enfermedad con CNN (MobileNetV2 + transfer learning)
- Visualizacion de resultados en frontend Gradio
- Notificacion por SMS usando Twilio

## Dataset

El dataset tiene 25,000 imagenes balanceadas en 5 clases (5,000 por clase):

| Class | Directory | Description |
|-------|-----------|-------------|
| Lung Benign | `lung_image_sets/lung_n/` | Healthy lung tissue |
| Lung Adenocarcinoma | `lung_image_sets/lung_aca/` | Lung adenocarcinoma |
| Lung Squamous Cell Carcinoma | `lung_image_sets/lung_scc/` | Lung squamous cell carcinoma |
| Colon Benign | `colon_image_sets/colon_n/` | Healthy colon tissue |
| Colon Adenocarcinoma | `colon_image_sets/colon_aca/` | Colon adenocarcinoma |

## Instalacion

1. Instala dependencias:

```bash
pip install -r requirements.txt
```

2. Crea variables de entorno:

```bash
cp .env.example .env
```

3. Completa `.env` con tus credenciales:

```env
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_PHONE_NUMBER=+15017122661
DESTINATION_PHONE_NUMBER=+573001234567
TWILIO_MESSAGING_SERVICE_SID=
```

Notas Twilio:
- Usa formato E.164 (`+` y codigo pais).
- En cuenta trial solo puedes enviar a numeros verificados en Twilio Console.
- Puedes usar `TWILIO_PHONE_NUMBER` o `TWILIO_MESSAGING_SERVICE_SID` como remitente.

## Entrenamiento del Modelo

```bash
python -m model.train
```

Pipeline actual:
- Fase 1: entrenamiento de la cabeza clasificadora
- Fase 2: fine-tuning conservador del backbone
- Regularizacion con AdamW, label smoothing y L2
- Evaluacion por clase al finalizar

Artefactos por corrida en `model/training_runs/<run_id>/`:
- `training_report.txt`
- `phase_1_metrics.csv` y `phase_2_metrics.csv`
- `training_curves.png`
- `classification_report.json` y `classification_report.txt`
- `confusion_matrix.csv` y `confusion_matrix.png`
- `error_summary.txt`
- `run_config.json`

## Frontend Gradio

Ejecuta:

```bash
python app.py
```

La UI incluye 3 pestañas:
- Diagnostico: subida de imagen, prediccion y envio opcional por SMS
- Analitica de entrenamiento: dashboard de corridas guardadas con matriz de confusion
- Twilio: estado de configuracion y envio de SMS de prueba

## Backend FastAPI (Opcional)

```bash
uvicorn api.main:app --reload
```

Endpoints:
- `POST /predict`: prediccion de imagen
- `POST /predict-and-notify`: prediccion + notificacion SMS

Docs interactivas: `http://localhost:8000/docs`

## Referencias usadas para Twilio

- Twilio Python helper (`messages.create`)
- Twilio Messaging Quickstart (variables de entorno y formato de numeros)
- Guias practicas de Twilio para FastAPI/Python
