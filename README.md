# Disease Detector - Histopathological Image Classification

A deep learning application that uses CNN to classify lung and colon cancer histopathological images, with a Gradio frontend and Twilio SMS notifications.

## Dataset

The dataset contains 25,000 images across 5 classes (5,000 each):

| Class | Directory | Description |
|-------|-----------|-------------|
| Lung Benign | `lung_image_sets/lung_n/` | Healthy lung tissue |
| Lung Adenocarcinoma | `lung_image_sets/lung_aca/` | Lung adenocarcinoma |
| Lung Squamous Cell Carcinoma | `lung_image_sets/lung_scc/` | Lung squamous cell carcinoma |
| Colon Benign | `colon_image_sets/colon_n/` | Healthy colon tissue |
| Colon Adenocarcinoma | `colon_image_sets/colon_aca/` | Colon adenocarcinoma |

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Twilio

1. Sign up at [twilio.com](https://www.twilio.com)
2. Get your Account SID, Auth Token, and phone number from the console
3. Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env`:
```
TWILIO_ACCOUNT_SID=your_account_sid_here
TWILIO_AUTH_TOKEN=your_auth_token_here
TWILIO_PHONE_NUMBER=+15017122661
DESTINATION_PHONE_NUMBER=+15558675309
```

### 3. Train the Model

```bash
python -m model.train
```

This will:
- Load all 25,000 images from both datasets
- **Phase 1**: Train classifier head with MobileNetV2 backbone (frozen)
- **Phase 2**: Fine-tune last 30 layers of MobileNetV2
- Save the best model to `model/cnn_model.h5`

## Model Architecture

The CNN uses **transfer learning with MobileNetV2** (pre-trained on ImageNet), which is the standard best practice for image classification:

- **Backbone**: MobileNetV2 (frozen initially, then fine-tuned)
- **Data Augmentation**: Random flip, rotation, zoom, translation
- **Classifier Head**: GlobalAveragePooling → Dropout → Dense(256) → BatchNorm → Dropout → Dense(5, softmax)
- **Training**: Two-phase approach (head training + fine-tuning)
- **Optimizations**: Adam optimizer, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

## Model Architecture

The CNN uses **transfer learning with MobileNetV2** (pre-trained on ImageNet), which is the standard best practice for image classification:

- **Backbone**: MobileNetV2 (frozen initially, then fine-tuned)
- **Data Augmentation**: Random flip, rotation, zoom, translation
- **Classifier Head**: GlobalAveragePooling → Dropout → Dense(256) → BatchNorm → Dropout → Dense(5, softmax)
- **Training**: Two-phase approach (head training + fine-tuning)
- **Optimizations**: Adam optimizer, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

### 4. Run the Gradio App

```bash
python app.py
```

Open the URL shown in the terminal (default: `http://localhost:7860`)

### 5. Run the FastAPI Backend (Optional)

```bash
uvicorn api.main:app --reload
```

API endpoints:
- `POST /predict` - Upload image, get prediction
- `POST /predict-and-notify` - Upload image, get prediction, send SMS

Interactive docs at: `http://localhost:8000/docs`

## Project Structure

```
├── model/
│   ├── model.py          # CNN architecture
│   ├── train.py          # Training script
│   └── cnn_model.h5      # Trained model (after training)
├── api/
│   ├── main.py           # FastAPI server
│   └── twilio_service.py # SMS notification service
├── app.py                # Gradio frontend
├── requirements.txt
├── .env.example
└── lung_image_sets/      # Dataset
└── colon_image_sets/     # Dataset
```
