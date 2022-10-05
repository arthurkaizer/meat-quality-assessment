import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from fastapi import FastAPI

from models import Models
from files import Files

model = Models.load()

if model == None:
    files, classes = Files.get_files_and_corresponding_classes()

    labels = LabelEncoder()
    labels.fit(classes)

    image_data = np.array(files) / 255.0
    image_classes = labels.transform(classes)

    result = train_test_split(
         image_data, 
         image_classes, 
         test_size=0.3, 
         random_state=123)

    train_images = result[0]
    test_images = result[1]
    train_labels = result[2]
    test_labels = result[3]

    model = Models.create()
    Models.compile_model_with_default_params(model)
    Models.train_model(model, train_images, train_labels, test_images, test_labels)
    Models.evaluate_model_and_show_metrics(model, test_images, test_labels)
    Models.save(model)

model.summary()

api = FastAPI()

@api.get("/")
def root():
    return model.summary()
