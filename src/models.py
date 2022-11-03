import os
import configs

from typing import List, Any
from tensorflow.keras import models, layers, losses
from tensorflow.keras.models import Sequential


def create_model_with_layers() -> Sequential:
    model = Sequential()

    model.add(
        layers.Conv2D(
            35,
            (3, 3),
            activation="relu",
            input_shape=(configs.image_width, configs.image_height, 3)
        )
    )
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(2))

    return model


def compile_model_with_default_params(model: Sequential):
    model.compile(
        optimizer="adam",
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )


def train_model(
    model: Sequential,
    train_data: List[Any],
    train_labels: List[Any],
    test_data: List[Any],
    test_labels: List[Any]
):
    model.fit(
        train_data,
        train_labels,
        epochs=configs.number_of_training_epochs,
        validation_data=(test_data, test_labels)
    )


def evaluate_model_and_show_metrics(model: Sequential, data, labels):
    evaluation_result = model.evaluate(data, labels)

    metrics_title = f'\n{"=" * 20} Metrics {"=" * 20}\n'
    print(metrics_title)

    for i in range(len(model.metrics_names)):
        print(model.metrics_names[i], ":", evaluation_result[i])

    print(f'\n{"=" * len(metrics_title)}')


def save_model_to_file(model: Sequential):
    model.save(configs.saved_model_file_path)


def load_model_from_file() -> Sequential | None:
    if os.path.exists(configs.saved_model_file_path):
        return models.load_model(configs.saved_model_file_path)

    return None
