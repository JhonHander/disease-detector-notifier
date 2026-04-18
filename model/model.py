import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers


def create_cnn_model(input_shape=(224, 224, 3), num_classes=5, trainable=False):
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    base_model.trainable = trainable

    data_augmentation = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.2),
            keras.layers.RandomZoom(0.2),
            keras.layers.RandomTranslation(0.2, 0.2),
        ]
    )

    model = keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            data_augmentation,
            # MobileNetV2 expects inputs in [-1, 1].
            keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1.0),
            base_model,
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(
                256,
                activation="relu",
                kernel_regularizer=regularizers.l2(1e-4),
            ),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(
                num_classes,
                activation="softmax",
                kernel_regularizer=regularizers.l2(1e-4),
            ),
        ]
    )
    return model
