import configs
from keras import layers, models, losses


class Model(models.Sequential):
    def __init__(self):
        models.Sequential.__init__(self)
        self._add_layers()

    def _add_layers(self):
        image_width = configs.image_width
        image_height = configs.image_height
        self.add(
            layers.Conv2D(
                35,
                (3, 3),
                activation="relu",
                input_shape=(image_width, image_height, 3),
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
