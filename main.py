import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from model import Model
from images import Images
from confusion_matrix import ConfusionMatrix
from testing import ModelTesting

import time


def main():
    images = Images()
    image_classes = images.classes
    image_data = np.array(images.images)

    labels = LabelEncoder()
    labels.fit(image_classes)

    x = image_data / 255.0
    y = labels.transform(image_classes)

    (train_images,
     test_images,
     train_labels,
     test_labels) = train_test_split(x, y, test_size=0.3, random_state=123)

    model = Model()
    model.compile_with_default_params()
    timeStart = time.time()
    model.train(train_images, train_labels, test_images, test_labels)
    timeEnd = time.time() - timeStart
    model.evaluate_and_show_metrics(test_images, test_labels)
    model.summary()

    confusion_matrix = ConfusionMatrix(model, test_images, test_labels)
    confusion_matrix.show()

    model_testing = ModelTesting(
        model=model,
        label_encoder=labels,
        image_data=image_data,
        image_classes=image_classes,
        number_of_items_to_test=30
    )
    model_testing.test_and_show_result()

    print ("\nResolved in " + str(timeEnd) + " seconds!\n")

if __name__ == "__main__":
    main()
