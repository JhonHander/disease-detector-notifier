import os
import json
from datetime import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
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
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1
PHASE_1_EPOCHS = 10
PHASE_2_EPOCHS = 15
LABEL_SMOOTHING = 0.1


def get_runtime_info():
    gpu_devices = tf.config.list_physical_devices("GPU")
    gpu_names = [device.name for device in gpu_devices]
    return {
        "tensorflow_version": tf.__version__,
        "gpu_count": len(gpu_devices),
        "gpu_devices": gpu_names,
        "training_device": "GPU" if gpu_devices else "CPU",
    }


def save_run_config(run_dir):
    os.makedirs(run_dir, exist_ok=True)
    config = {
        "classes": CLASSES,
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "validation_split": VALIDATION_SPLIT,
        "test_split": TEST_SPLIT,
        "phase_1_epochs": PHASE_1_EPOCHS,
        "phase_2_epochs": PHASE_2_EPOCHS,
        "label_smoothing": LABEL_SMOOTHING,
        "optimizer_phase_1": {"name": "AdamW", "learning_rate": 5e-4, "weight_decay": 1e-4},
        "optimizer_phase_2": {"name": "AdamW", "learning_rate": 1e-5, "weight_decay": 5e-5},
    }
    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)


def save_split_manifest(run_dir, train_paths, val_paths, test_paths):
    os.makedirs(run_dir, exist_ok=True)

    def summarize(paths):
        per_class = {class_name: 0 for class_name in CLASSES}
        for path in paths:
            class_name = os.path.basename(os.path.dirname(path))
            if class_name in per_class:
                per_class[class_name] += 1
        return {
            "total": len(paths),
            "per_class": per_class,
        }

    manifest = {
        "train": summarize(train_paths),
        "validation": summarize(val_paths),
        "test": summarize(test_paths),
        "note": "Validation is used for model selection during training. Test is only used for final evaluation.",
    }

    with open(os.path.join(run_dir, "split_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    with open(os.path.join(run_dir, "train_files.txt"), "w") as f:
        f.write("\n".join(train_paths))

    with open(os.path.join(run_dir, "validation_files.txt"), "w") as f:
        f.write("\n".join(val_paths))

    with open(os.path.join(run_dir, "test_files.txt"), "w") as f:
        f.write("\n".join(test_paths))


def build_callbacks(run_dir, phase_name):
    os.makedirs(run_dir, exist_ok=True)
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=4, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7
        ),
        keras.callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH,
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(run_dir, f"{phase_name}_best.keras"),
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
        ),
        keras.callbacks.CSVLogger(os.path.join(run_dir, f"{phase_name}_metrics.csv")),
        keras.callbacks.TensorBoard(log_dir=os.path.join(run_dir, "logs", phase_name)),
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
    val_acc,
    test_acc,
    macro_f1,
    class_summary,
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

Validation Accuracy (monitoring split): {val_acc:.4f}
Test Accuracy (final holdout split): {test_acc:.4f}
Test Macro F1: {macro_f1:.4f}

Class Summary:
{class_summary}

Model saved to: {MODEL_SAVE_PATH}
========================================
"""

    with open(os.path.join(run_dir, "training_report.txt"), "w") as f:
        f.write(report.strip())
    
    print(report)


def build_class_summary(cm, report_dict):
    class_rows = []
    for class_name in CLASSES:
        row = report_dict.get(class_name, {})
        class_rows.append(
            {
                "name": class_name,
                "precision": float(row.get("precision", 0.0)),
                "recall": float(row.get("recall", 0.0)),
                "f1": float(row.get("f1-score", 0.0)),
                "support": int(row.get("support", 0)),
            }
        )

    worst_recall = min(class_rows, key=lambda x: x["recall"])

    confusions = []
    for i, src in enumerate(CLASSES):
        for j, dst in enumerate(CLASSES):
            if i == j:
                continue
            count = int(cm[i, j])
            if count > 0:
                confusions.append((count, src, dst))
    confusions.sort(reverse=True, key=lambda x: x[0])
    top_confusions = confusions[:3]

    class_lines = [
        f"- {row['name']}: precision={row['precision']:.4f}, recall={row['recall']:.4f}, f1={row['f1']:.4f}, support={row['support']}"
        for row in class_rows
    ]

    confusion_lines = [
        f"- {src} -> {dst}: {count} casos"
        for count, src, dst in top_confusions
    ]
    if not confusion_lines:
        confusion_lines = ["- Sin confusiones fuera de la diagonal."]

    summary = "\n".join(
        [
            "Per-class metrics:",
            *class_lines,
            f"Worst recall class: {worst_recall['name']} ({worst_recall['recall']:.4f})",
            "Top confusions:",
            *confusion_lines,
        ]
    )
    return summary


def save_classification_artifacts(model, eval_dataset, run_dir):
    os.makedirs(run_dir, exist_ok=True)

    y_true_batches = []
    y_pred_batches = []
    for images, labels in eval_dataset:
        probs = model(images, training=False).numpy()
        y_true_batches.append(tf.argmax(labels, axis=1).numpy())
        y_pred_batches.append(np.argmax(probs, axis=1))

    y_true = np.concatenate(y_true_batches)
    y_pred = np.concatenate(y_pred_batches)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    cm_path = os.path.join(run_dir, "confusion_matrix.csv")
    np.savetxt(cm_path, cm, delimiter=",", fmt="%d")

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    report_path = os.path.join(run_dir, "classification_report.json")
    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=2)

    report_text = classification_report(
        y_true,
        y_pred,
        target_names=CLASSES,
        zero_division=0,
    )
    with open(os.path.join(run_dir, "classification_report.txt"), "w") as f:
        f.write(report_text)

    # Visual summary for quick error inspection by class.
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(CLASSES))
    plt.xticks(ticks, CLASSES, rotation=45, ha="right")
    plt.yticks(ticks, CLASSES)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "confusion_matrix.png"), dpi=120)
    plt.close()

    macro_f1 = report_dict.get("macro avg", {}).get("f1-score", 0.0)
    class_summary = build_class_summary(cm, report_dict)
    with open(os.path.join(run_dir, "error_summary.txt"), "w") as f:
        f.write(class_summary)

    return macro_f1, class_summary


def main():
    started_at = datetime.now()
    run_id = started_at.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RUNS_DIR, run_id)
    runtime_info = get_runtime_info()

    print(f"\n★ Starting training run: {run_id}")
    print(f"★ Artifacts will be saved to: {run_dir}\n")
    save_run_config(run_dir)

    lung_dir = os.path.join(DATASET_DIR, "lung_image_sets")
    colon_dir = os.path.join(DATASET_DIR, "colon_image_sets")
    class_to_idx = {name: idx for idx, name in enumerate(CLASSES)}

    def collect_labeled_paths():
        all_paths = []
        all_labels = []
        for class_name in CLASSES:
            if class_name.startswith("lung_"):
                class_dir = os.path.join(lung_dir, class_name)
            else:
                class_dir = os.path.join(colon_dir, class_name)

            for file_name in sorted(os.listdir(class_dir)):
                file_path = os.path.join(class_dir, file_name)
                if os.path.isfile(file_path):
                    all_paths.append(file_path)
                    all_labels.append(class_to_idx[class_name])
        return all_paths, all_labels

    def print_split_stats(name, labels):
        counts = np.bincount(np.array(labels), minlength=len(CLASSES))
        parts = [f"{CLASSES[i]}={int(counts[i])}" for i in range(len(CLASSES))]
        print(f"{name} split counts: " + ", ".join(parts))

    def build_dataset(paths, labels, training):
        paths_tensor = tf.constant(paths)
        labels_tensor = tf.constant(labels, dtype=tf.int32)
        dataset = tf.data.Dataset.from_tensor_slices((paths_tensor, labels_tensor))

        if training:
            dataset = dataset.shuffle(
                buffer_size=min(len(paths), 4000),
                seed=42,
                reshuffle_each_iteration=True,
            )

        def load_image(path, label):
            image_bytes = tf.io.read_file(path)
            image = tf.io.decode_image(
                image_bytes,
                channels=3,
                expand_animations=False,
            )
            image.set_shape([None, None, 3])
            image = tf.image.resize(image, IMG_SIZE)
            image = tf.cast(image, tf.float32)
            label_one_hot = tf.one_hot(label, depth=len(CLASSES), dtype=tf.float32)
            return image, label_one_hot

        dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        return dataset

    all_paths, all_labels = collect_labeled_paths()
    train_paths, holdout_paths, train_labels, holdout_labels = train_test_split(
        all_paths,
        all_labels,
        test_size=VALIDATION_SPLIT + TEST_SPLIT,
        random_state=42,
        shuffle=True,
        stratify=all_labels,
    )

    val_paths, test_paths, val_labels, test_labels = train_test_split(
        holdout_paths,
        holdout_labels,
        test_size=TEST_SPLIT / (VALIDATION_SPLIT + TEST_SPLIT),
        random_state=42,
        shuffle=True,
        stratify=holdout_labels,
    )

    print(f"Total images: {len(all_paths)}")
    print(
        f"Train images: {len(train_paths)} | Validation images: {len(val_paths)} | Test images: {len(test_paths)}"
    )
    print_split_stats("Train", train_labels)
    print_split_stats("Validation", val_labels)
    print_split_stats("Test", test_labels)

    save_split_manifest(run_dir, train_paths, val_paths, test_paths)

    train_dataset = build_dataset(train_paths, train_labels, training=True)
    val_dataset = build_dataset(val_paths, val_labels, training=False)
    test_dataset = build_dataset(test_paths, test_labels, training=False)

    print("Creating model...")
    model = create_cnn_model(
        input_shape=(*IMG_SIZE, 3), num_classes=len(CLASSES), trainable=False
    )

    print("\n=== Phase 1: Training Classifier Head ===")
    head_optimizer = keras.optimizers.AdamW(learning_rate=5e-4, weight_decay=1e-4)
    classification_loss = keras.losses.CategoricalCrossentropy(
        label_smoothing=LABEL_SMOOTHING
    )
    model.compile(
        optimizer=head_optimizer,
        loss=classification_loss,
        metrics=["accuracy"],
    )

    history = model.fit(
        train_dataset,
        epochs=PHASE_1_EPOCHS,
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
    for layer in base_model.layers[:-10]:
        layer.trainable = False

    fine_optimizer = keras.optimizers.AdamW(learning_rate=1e-5, weight_decay=5e-5)
    model.compile(
        optimizer=fine_optimizer,
        loss=classification_loss,
        metrics=["accuracy"],
    )

    history_fine = model.fit(
        train_dataset,
        epochs=PHASE_2_EPOCHS,
        validation_data=val_dataset,
        callbacks=build_callbacks(run_dir, "phase_2"),
    )

    print("\nEvaluating on validation dataset...")
    _, val_acc = model.evaluate(val_dataset)
    print(f"Validation Accuracy: {val_acc:.4f}")

    print("\nEvaluating on test dataset...")
    _, test_acc = model.evaluate(test_dataset)
    print(f"Test Accuracy: {test_acc:.4f}")
    macro_f1, class_summary = save_classification_artifacts(model, test_dataset, run_dir)
    print(f"Test Macro F1: {macro_f1:.4f}")
    print(class_summary)

    save_metrics_plot(history, history_fine, run_dir)
    save_training_report(
        run_dir,
        started_at,
        runtime_info,
        history,
        history_fine,
        val_acc,
        test_acc,
        macro_f1,
        class_summary,
    )

    print(f"\n✓ Model saved to: {MODEL_SAVE_PATH}")
    print(f"✓ Training artifacts saved to: {run_dir}\n")


if __name__ == "__main__":
    main()
