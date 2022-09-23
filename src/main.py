import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import os
from IPython.display import clear_output

# Imports for CNN
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import configs
from images import Images
from image_filenames import ImageFilenames


def main():
    filenames = ImageFilenames()
    # table = pd.DataFrame(image_filenames.grouped_filenames).head()

    # sizes = [len(data['Fresh']), len(data['Spoiled'])]

    # plt.figure(figsize=(10,5), dpi=100)

    # plt.pie(x=sizes,autopct='%1.0f%%',shadow=False, textprops={'color':"w","fontsize":15}, startangle=90,explode=(0,.01))
    # plt.legend(files,bbox_to_anchor=(0.4, 0, .7, 1))
    # plt.title("Data Split")
    # plt.show()

    images = Images(filenames)
    image_data = np.array(images.files)
    size = image_data.shape[0]
    image_data.shape

    # plt.figure(figsize=(15,15))
    # for i in range(1,17):
    #    fig = np.random.choice(np.arange(size))
    #    plt.subplot(4,4,i)
    #    plt.imshow(image_data[fig])
    #    if image_target[fig]=='Fresh':
    #        c='green'
    #    else:
    #        c='red'
    #    plt.title(image_target[fig], color=c)
    #    plt.xticks([]), plt.yticks([])
    # plt.show()

    labels = LabelEncoder()
    labels.fit(images.classes)

    x = image_data / 255.0
    y = labels.transform(images.classes)
    train_images, test_images, train_labels, test_labels = train_test_split(
        x, y, test_size=0.3, random_state=123
    )

    model = models.Sequential()
    model.add(
        layers.Conv2D(
            35,
            (3, 3),
            activation="relu",
            input_shape=(configs.image_width, configs.image_height, 3),
        )
    )
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(2))

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    history = model.fit(
        train_images, train_labels, epochs=2, validation_data=(test_images, test_labels)
    )

    # plt.style.use('ggplot')
    # plt.figure(figsize=(10, 5))
    ###plt.plot(history.history['accuracy'], label='accuracy')
    ##lt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.ylim([0.5, 1.01])
    # plt.legend(loc='lower right')

    # test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

    result = model.evaluate(test_images, test_labels)
    print(result)

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
    for file in range(1, 17):
        fig = np.random.choice(np.arange(size))
        plt.subplot(4, 4, file)
        plt.imshow(image_data[fig])
        if images.classes[fig] == "Fresh":
            c = "green"
        else:
            c = "red"
        plt.title(images.classes[fig], color=c)
        plt.ylabel(
            "| Pred:{} |".format(Prediction(image_data[fig])), fontsize=17, color=c
        )
        plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == "__main__":
    main()
