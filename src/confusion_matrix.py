import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, List
from sklearn.metrics import confusion_matrix as create_confusion_matrix

from aliases import Model


class ConfusionMatrix:
    def __init__(self, model: Model, test_data: List[Any], test_labels: List[Any]):
        self._model = model
        self._test_data = test_data
        self._test_labels = test_labels
        self._confusion_matrix = None
        self._predict_and_create_matrix()

    def _predict_and_create_matrix(self):
        prediction_result = self._model.predict(self._test_data)
        self._confusion_matrix = create_confusion_matrix(
            self._test_labels,
            self._handle_prediction_result(prediction_result)
        )

    def _handle_prediction_result(self, result: Any):
        length = len(result)
        classes = np.zeros(length)

        for i in range(length):
            classes[i] = result[i].argmax()

        return classes

    def _create_data_frame_and_heat_map(self):
        data_frame = pd.DataFrame(
            columns=["Fresh", "Spoiled"],
            index=["Fresh", "Spoiled"],
            data=self._confusion_matrix
        )

        _, axes = plt.subplots(figsize=(4, 4))

        sns.heatmap(
            data_frame,
            annot=True,
            cmap="Greens",
            fmt=".0f",
            ax=axes,
            linewidths=5,
            cbar=False,
            annot_kws={"size": 16},
        )

    def show(self):
        self._create_data_frame_and_heat_map()
        plt.xlabel("Predicted Label")
        plt.xticks(size=16)
        plt.yticks(size=16, rotation=0)
        plt.ylabel("True Label")
        plt.title("Confusion Matrix", size=12)
        plt.show()
