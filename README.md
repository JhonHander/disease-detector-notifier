<h1 align="center">Disease Detector</h1>

<p align="center">
  Clasificacion histopatologica con CNN (MobileNetV2), interfaz Gradio y notificaciones SMS con Twilio.
</p>

<p align="center">
	<img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
	<img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-2.15%2B-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
	<img alt="Model" src="https://img.shields.io/badge/Backbone-MobileNetV2-0EA5A5?style=for-the-badge" />
	<img alt="FastAPI" src="https://img.shields.io/badge/API-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
	<img alt="Gradio" src="https://img.shields.io/badge/Frontend-Gradio-F97316?style=for-the-badge" />
	<img alt="Twilio" src="https://img.shields.io/badge/Notifications-Twilio-F22F46?style=for-the-badge&logo=twilio&logoColor=white" />
</p>

## Resumen

**Disease Detector** es una aplicacion universitaria para clasificar imagenes histopatologicas de pulmon y colon en **5 clases balanceadas**, con dos modos de consumo:

1. **Interfaz web con Gradio** para diagnostico visual, analitica de entrenamiento y pruebas de SMS.
2. **API con FastAPI** para integracion programatica con endpoints de prediccion.

El modelo se entrena con transfer learning sobre **MobileNetV2** y guarda artefactos detallados por corrida para trazabilidad y analisis.

> [!IMPORTANT]
> Este repositorio incluye un dataset grande (25k imagenes) y multiples artefactos de entrenamiento. Asegura espacio en disco suficiente antes de entrenar desde cero.

## Caracteristicas

- Clasificacion de histopatologia en 5 categorias clinicas.
- Pipeline de entrenamiento en 2 fases (head training + fine-tuning).
- Registro de metricas, reportes por clase y matriz de confusion por corrida.
- Frontend en Gradio con pestañas de diagnostico, analitica y Twilio.
- Envio opcional de diagnosticos por SMS via Twilio.
- API REST para prediccion individual y prediccion con notificacion.

## Dataset

El dataset tiene **25,000 imagenes** balanceadas (**5,000 por clase**):

| Clase | Directorio | Descripcion |
|---|---|---|
| Lung Benign | `lung_image_sets/lung_n/` | Tejido pulmonar sano |
| Lung Adenocarcinoma | `lung_image_sets/lung_aca/` | Adenocarcinoma de pulmon |
| Lung Squamous Cell Carcinoma | `lung_image_sets/lung_scc/` | Carcinoma escamoso de pulmon |
| Colon Benign | `colon_image_sets/colon_n/` | Tejido de colon sano |
| Colon Adenocarcinoma | `colon_image_sets/colon_aca/` | Adenocarcinoma de colon |

## Arquitectura Rapida

```text
Images -> Preprocessing (160x160 RGB) -> CNN MobileNetV2 -> Prediccion
                                                         |
                                                         +-> Gradio UI
                                                         +-> FastAPI endpoints
                                                         +-> Twilio SMS (opcional)
```

## Instalacion

### 1) Crear entorno e instalar dependencias

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Configurar variables de entorno

```bash
cp .env.example .env
```

Completa `.env`:

```env
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_PHONE_NUMBER=+15017122661
DESTINATION_PHONE_NUMBER=+573001234567
TWILIO_MESSAGING_SERVICE_SID=
```

> [!NOTE]
> Puedes enviar con `TWILIO_PHONE_NUMBER` o con `TWILIO_MESSAGING_SERVICE_SID`.

> [!TIP]
> Usa formato E.164 (`+` y codigo de pais). En cuentas trial de Twilio, solo se puede enviar a numeros verificados.

## Entrenamiento del Modelo

```bash
python -m model.train
```

Pipeline actual:

1. **Fase 1**: entrenamiento de la cabeza clasificadora.
2. **Fase 2**: fine-tuning conservador del backbone.
3. Regularizacion con AdamW, label smoothing y L2.
4. Evaluacion final con accuracy, macro F1 y reportes por clase.

Artefactos por corrida en `model/training_runs/<run_id>/`:

- `training_report.txt`
- `phase_1_metrics.csv` y `phase_2_metrics.csv`
- `training_curves.png`
- `classification_report.json` y `classification_report.txt`
- `confusion_matrix.csv` y `confusion_matrix.png`
- `error_summary.txt`
- `run_config.json`
- `split_manifest.json` y listas de archivos (`train_files.txt`, `validation_files.txt`, `test_files.txt`)

## Ejecutar la Aplicacion

### Opcion A: Frontend Gradio

```bash
python app.py
```

Pestanas disponibles:

- **Diagnostico**: carga de imagen, prediccion y envio opcional por SMS.
- **Analitica de entrenamiento**: dashboard de corridas guardadas, metricas y matriz de confusion.
- **Twilio**: estado de configuracion y envio de SMS de prueba.

### Opcion B: Backend FastAPI

```bash
uvicorn api.main:app --reload
```

API docs: `http://localhost:8000/docs`

Endpoints:

- `POST /predict`: prediccion de imagen.
- `POST /predict-and-notify`: prediccion y envio de SMS.

## Ejemplos de Uso API

```bash
curl -X POST "http://localhost:8000/predict" \
	-F "file=@/ruta/a/imagen.png"
```

```bash
curl -X POST "http://localhost:8000/predict-and-notify" \
	-F "file=@/ruta/a/imagen.png" \
	-F "phone=+573001234567"
```

## Estructura del Proyecto

```text
.
|-- app.py
|-- api/
|   |-- main.py
|   `-- twilio_service.py
|-- model/
|   |-- model.py
|   |-- train.py
|   |-- cnn_model.h5
|   `-- training_runs/
|-- lung_image_sets/
|-- colon_image_sets/
|-- requirements.txt
|-- TWILIO_SETUP_GUIA_PASO_A_PASO.md
`-- README.md
```

## Referencias

- Twilio Python Helper Library (`messages.create`)
- Twilio Messaging Quickstart
- Buenas practicas FastAPI + Python para integraciones SMS
