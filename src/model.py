from typing import Any, List
from keras import layers, models, losses

import configs


class Model(models.Sequential):
    def __init__(self):
        models.Sequential.__init__(self)
        self._add_layers()

    def _add_layers(self):
        self.add(
            layers.Conv2D(
                35,
                (3, 3),
                activation="relu",
                input_shape=(configs.image_width, configs.image_height, 3)
            )
        )
        self.add(layers.MaxPooling2D((2, 2)))
        self.add(layers.Conv2D(64, (3, 3), activation="relu"))
        self.add(layers.MaxPooling2D((2, 2)))
        self.add(layers.Conv2D(64, (3, 3), activation="relu"))
        self.add(layers.Flatten())
        self.add(layers.Dense(64, activation="relu"))
        self.add(layers.Dense(2))

    def compile_with_default_params(self):
        self.compile(
            optimizer="adam",
            loss=losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

    def train(self, train_data: List[Any], train_labels: List[Any], test_data: List[Any], test_labels: List[Any]):
        self.fit(
            train_data,
            train_labels,
            epochs=configs.number_of_training_epochs,
            validation_data=(test_data, test_labels)
        )

    def evaluate_and_show_metrics(self, images, labels):
        evaluation_result = self.evaluate(images, labels)

        metrics_title = f'\n{"=" * 20} Metrics {"=" * 20}\n'
        print(metrics_title)

        for i in range(len(self.metrics_names)):
            print(self.metrics_names[i], ":", evaluation_result[i])

        print(f'\n{"=" * len(metrics_title)}')
