import os
import configs
from aliases import Model

from typing import List, Any
from tensorflow.keras import models, layers, losses


class Models:
    @staticmethod
    def compile_model_with_default_params(model: Model):
        model.compile(
            optimizer="adam",
            loss=losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

    @staticmethod
    def train_model(model: Model, train_data: List[Any], train_labels: List[Any], test_data: List[Any], test_labels: List[Any]):
        model.fit(
            train_data,
            train_labels,
            epochs=configs.number_of_training_epochs,
            validation_data=(test_data, test_labels)
        )

    @staticmethod
    def evaluate_model_and_show_metrics(model: Model, data, labels):
        evaluation_result = model.evaluate(data, labels)

        metrics_title = f'\n{"=" * 20} Metrics {"=" * 20}\n'
        print(metrics_title)

        for i in range(len(model.metrics_names)):
            print(model.metrics_names[i], ":", evaluation_result[i])

        print(f'\n{"=" * len(metrics_title)}')

    @staticmethod
    def load() -> Model:
        if os.path.exists(configs.saved_model_file_path):
            return models.load_model(configs.saved_model_file_path)

        return None

    @staticmethod
    def save(model: Model):
        model.save(configs.saved_model_file_path)

    @staticmethod
    def _add_layers_to_model(model: Model):
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

    @staticmethod
    def create() -> Model:
        model = models.Sequential()
        Models._add_layers_to_model(model)
        return model
