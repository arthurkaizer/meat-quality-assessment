import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from IPython.display import clear_output

# Imports for CNN
import tensorflow as tf
from model import Model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import configs
from image_classes import ImageClasses
from images import Images


def main():
    images = Images()
    image_classes = images.get_classes()
    image_data = images.get_files()
    size = image_data.shape[0]
    image_data.shape

    labels = LabelEncoder()
    labels.fit(image_classes)

    x = image_data / 255.0
    y = labels.transform(image_classes)
    train_images, test_images, train_labels, test_labels = train_test_split(
        x, y, test_size=0.3, random_state=123
    )

    model = Model()

    model.compile_with_default_params()

    history = model.fit(
        train_images, train_labels, epochs=2, validation_data=(test_images, test_labels)
    )

    result = model.evaluate(test_images, test_labels)

    for file in range(len(model.metrics_names)):
        print(model.metrics_names[file], ":", result[file])

    model.summary()

    y_pred = model.predict(test_images)

    def toClass(pred):
        class_ = np.zeros(len(pred))
        for i in range(len(pred)):
            index = pred[i].argmax()
            class_[i] = index

        return class_

    cm = confusion_matrix(test_labels, toClass(y_pred))

    df1 = pd.DataFrame(
        columns=["Fresh", "Spoiled"], index=["Fresh", "Spoiled"], data=cm
    )

    f, ax = plt.subplots(figsize=(6, 6))

    sns.heatmap(
        df1,
        annot=True,
        cmap="Greens",
        fmt=".0f",
        ax=ax,
        linewidths=5,
        cbar=False,
        annot_kws={"size": 16},
    )

    plt.xlabel("Predicted Label")
    plt.xticks(size=12)
    plt.yticks(size=12, rotation=0)
    plt.ylabel("True Label")
    plt.title("YSA Confusion Matrix", size=12)
    plt.show()

    def Prediction(image):
        img = cv2.resize(image, (configs.image_width, configs.image_height))
        test = img / 255.0
        pred = model.predict(np.array([image])).argmax()
        return labels.inverse_transform([pred])[0]

    plt.figure(figsize=(15, 15))
    for file in range(1, 10):
        fig = np.random.choice(np.arange(size))
        plt.subplot(4, 4, file)
        plt.imshow(image_data[fig])
        color = "green" if image_classes[fig] == ImageClasses.Fresh else "red"
        prediction = f"| Pred:{Prediction(image_data[fig])} |"
        plt.title(image_classes[fig], color=color)
        plt.ylabel(prediction, fontsize=17, color=color)
        plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == "__main__":
    main()
