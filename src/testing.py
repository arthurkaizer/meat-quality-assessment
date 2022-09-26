import cv2
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from typing import List

from model import Model
from image_classes import ImageClasses


class ModelTesting:
    def __init__(self, model: Model, image_data: np.ndarray[cv2.Mat], image_classes: List[ImageClasses], label_encoder: LabelEncoder, number_of_items_to_test: int):
        self._model = model
        self._image_data = image_data
        self._image_classes = image_classes
        self._label_encoder = label_encoder
        self._number_of_items_to_test = number_of_items_to_test

    def _get_random_image(self):
        index = random.randint(1, self._image_data.shape[0])
        image = self._image_data[index]
        return (image, index)

    def _predict(self, image: cv2.Mat, index: int):
        real_class = self._image_classes[index]
        result = self._model.predict(np.array([image])).argmax()
        predicted_class = self._label_encoder.inverse_transform([result])[0]
        return (predicted_class, real_class)

    def test_and_show_result(self):
        plt.figure(figsize=(15, 15))
        rows = math.ceil(math.sqrt(self._number_of_items_to_test))

        for index in range(1, self._number_of_items_to_test):
            image, image_index = self._get_random_image()
            predicted_class, real_class = self._predict(image, image_index)
            color = "green" if real_class == ImageClasses.Fresh else "red"
            plt.subplot(rows, rows, index)
            plt.imshow(image)
            plt.title(real_class, color=color)
            plt.ylabel(f"Predicted: {predicted_class}",
                       fontsize=10, color=color)
            plt.xticks([])
            plt.yticks([])

        plt.show()
