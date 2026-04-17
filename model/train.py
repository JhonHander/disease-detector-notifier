import os
import json
from datetime import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from model.model import create_cnn_model

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Using GPU: {gpus}")
else:
    print("Using CPU")

DATASET_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_SAVE_PATH = os.path.join(DATASET_DIR, "model", "cnn_model.h5")
RUNS_DIR = os.path.join(DATASET_DIR, "model", "training_runs")

CLASSES = ["lung_n", "lung_aca", "lung_scc", "colon_n", "colon_aca"]
IMG_SIZE = (160, 160)
BATCH_SIZE = 12
EPOCHS = 20


def get_runtime_info():
    gpu_devices = tf.config.list_physical_devices("GPU")
    gpu_names = [device.name for device in gpu_devices]
    return {
        "tensorflow_version": tf.__version__,
        "gpu_count": len(gpu_devices),
        "gpu_devices": gpu_names,
        "training_device": "GPU" if gpu_devices else "CPU",
    }


def build_callbacks(run_dir, phase_name):
    os.makedirs(run_dir, exist_ok=True)
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7
        ),
        keras.callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH, save_best_only=True, monitor="val_accuracy"
        ),
        keras.callbacks.CSVLogger(os.path.join(run_dir, f"{phase_name}_metrics.csv")),
    ]


def save_metrics_plot(history_head, history_fine, run_dir):
    os.makedirs(run_dir, exist_ok=True)

    head_acc = history_head.history.get("accuracy", [])
    head_val_acc = history_head.history.get("val_accuracy", [])
    fine_acc = history_fine.history.get("accuracy", [])
    fine_val_acc = history_fine.history.get("val_accuracy", [])

    acc = head_acc + fine_acc
    val_acc = head_val_acc + fine_val_acc

    epochs = list(range(1, len(acc) + 1))

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, acc, label="Train Accuracy", marker="o")
    plt.plot(epochs, val_acc, label="Validation Accuracy", marker="o")
    plt.axvline(x=len(head_acc), color="gray", linestyle="--", linewidth=1)
    plt.title("Model Accuracy During Training")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "training_curves.png"), dpi=100)
    plt.close()


def save_training_report(
    run_dir,
    started_at,
    runtime_info,
    history_head,
    history_fine,
    test_acc,
):
    os.makedirs(run_dir, exist_ok=True)

    ended_at = datetime.now()
    duration_seconds = (ended_at - started_at).total_seconds()

    report = f"""
========================================
TRAINING REPORT - Disease Detector CNN
========================================
Run ID: {os.path.basename(run_dir)}
Started: {started_at.strftime('%Y-%m-%d %H:%M:%S')}
Ended: {ended_at.strftime('%Y-%m-%d %H:%M:%S')}
Duration: {duration_seconds:.1f} seconds

Device: {runtime_info['training_device']}
GPU Count: {runtime_info['gpu_count']}
TensorFlow: {runtime_info['tensorflow_version']}

Configuration:
  Image Size: {IMG_SIZE}
  Batch Size: {BATCH_SIZE}
  Classes: {len(CLASSES)}

Phase 1 (Head Training):
  Epochs: {len(history_head.history.get('loss', []))}
  Final Accuracy: {history_head.history.get('accuracy', [0])[-1]:.4f}
  Final Val Accuracy: {history_head.history.get('val_accuracy', [0])[-1]:.4f}

Phase 2 (Fine-tuning):
  Epochs: {len(history_fine.history.get('loss', []))}
  Final Accuracy: {history_fine.history.get('accuracy', [0])[-1]:.4f}
  Final Val Accuracy: {history_fine.history.get('val_accuracy', [0])[-1]:.4f}

Validation Accuracy: {test_acc:.4f}

Model saved to: {MODEL_SAVE_PATH}
========================================
"""

    with open(os.path.join(run_dir, "training_report.txt"), "w") as f:
        f.write(report.strip())
    
    print(report)


def main():
    started_at = datetime.now()
    run_id = started_at.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RUNS_DIR, run_id)
    runtime_info = get_runtime_info()

    print(f"\n★ Starting training run: {run_id}")
    print(f"★ Artifacts will be saved to: {run_dir}\n")

    lung_dir = os.path.join(DATASET_DIR, "lung_image_sets")
    colon_dir = os.path.join(DATASET_DIR, "colon_image_sets")
    global_class_to_idx = {name: idx for idx, name in enumerate(CLASSES)}

    def load_split_dataset(data_dir, subset):
        return keras.utils.image_dataset_from_directory(
            data_dir,
            labels="inferred",
            label_mode="categorical",
            batch_size=BATCH_SIZE,
            image_size=IMG_SIZE,
            shuffle=True,
            seed=42,
            validation_split=0.2,
            subset=subset,
        )

    def remap_to_global_labels(dataset, local_class_names):
        local_to_global = tf.constant(
            [global_class_to_idx[name] for name in local_class_names], dtype=tf.int32
        )

        def mapper(images, labels):
            local_ids = tf.argmax(labels, axis=1, output_type=tf.int32)
            global_ids = tf.gather(local_to_global, local_ids)
            remapped = tf.one_hot(global_ids, depth=len(CLASSES), dtype=tf.float32)
            return images, remapped

        return dataset.map(mapper, num_parallel_calls=tf.data.AUTOTUNE)

    print("Loading lung train/validation datasets...")
    lung_train_raw = load_split_dataset(lung_dir, "training")
    lung_val_raw = load_split_dataset(lung_dir, "validation")

    print("Loading colon train/validation datasets...")
    colon_train_raw = load_split_dataset(colon_dir, "training")
    colon_val_raw = load_split_dataset(colon_dir, "validation")

    print(f"Lung local classes: {lung_train_raw.class_names}")
    print(f"Colon local classes: {colon_train_raw.class_names}")

    lung_train = remap_to_global_labels(lung_train_raw, lung_train_raw.class_names)
    lung_val = remap_to_global_labels(lung_val_raw, lung_val_raw.class_names)
    colon_train = remap_to_global_labels(colon_train_raw, colon_train_raw.class_names)
    colon_val = remap_to_global_labels(colon_val_raw, colon_val_raw.class_names)

    train_dataset = lung_train.concatenate(colon_train)
    train_dataset = train_dataset.shuffle(buffer_size=500).prefetch(tf.data.AUTOTUNE)
    val_dataset = lung_val.concatenate(colon_val).prefetch(tf.data.AUTOTUNE)

    print("Creating model...")
    model = create_cnn_model(
        input_shape=(*IMG_SIZE, 3), num_classes=len(CLASSES), trainable=False
    )

    print("\n=== Phase 1: Training Classifier Head ===")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        train_dataset,
        epochs=10,
        validation_data=val_dataset,
        callbacks=build_callbacks(run_dir, "phase_1"),
    )

    print("\n=== Phase 2: Fine-tuning Last Layers ===")
    base_model = next(
        layer
        for layer in model.layers
        if isinstance(layer, keras.Model) and "mobilenetv2" in layer.name.lower()
    )
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history_fine = model.fit(
        train_dataset,
        epochs=15,
        validation_data=val_dataset,
        callbacks=build_callbacks(run_dir, "phase_2"),
    )

    print("\nEvaluating on validation dataset...")
    test_loss, test_acc = model.evaluate(val_dataset)
    print(f"Validation Accuracy: {test_acc:.4f}")

    save_metrics_plot(history, history_fine, run_dir)
    save_training_report(run_dir, started_at, runtime_info, history, history_fine, test_acc)

    print(f"\n✓ Model saved to: {MODEL_SAVE_PATH}")
    print(f"✓ Training artifacts saved to: {run_dir}\n")


if __name__ == "__main__":
    main()
