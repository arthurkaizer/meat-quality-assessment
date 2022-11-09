import os
import cv2
import numpy as np
from uvicorn import run
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

import configs
import models
from files import Files


api = FastAPI()

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
    allow_credentials=True
)


model = models.load_model_from_file()

files, classes = Files.get_files_and_corresponding_classes()

labels = LabelEncoder()
labels.fit(classes)

if model is None:
    images_as_np_array = np.array(files) / 255.0
    normalized_classes = labels.transform(classes)

    result = train_test_split(
        images_as_np_array,
        normalized_classes,
        test_size=0.3,
        random_state=123)

    train_images, test_images = result[0], result[1]
    train_labels, test_labels = result[2], result[3]

    model = models.create_model_with_layers()
    models.compile_model_with_default_params(model)
    models.train_model(model, train_images, train_labels,
                       test_images, test_labels)
    models.evaluate_model_and_show_metrics(model, test_images, test_labels)
    models.save_model_to_file(model)


async def _get_resized_image_from_file(file: UploadFile) -> np.ndarray:
    try:
        file_content = await file.read()
        content_as_np_array = np.frombuffer(file_content, np.uint8)
        image = cv2.imdecode(content_as_np_array, cv2.COLOR_BGR2RGB)
        target_size = (configs.image_width, configs.image_height)
        resized_image = cv2.resize(image, target_size)
        return np.array([resized_image])
    except Exception:
        return np.array([])
    finally:
        file.file.close()


def _predict_class_for_image(image: np.ndarray) -> str:
    prediction_result = model.predict(image)
    argmax = prediction_result.argmax()
    return labels.inverse_transform([argmax])[0]


@api.post("/classify")
async def classify_image(file: UploadFile):
    allowed_file_extensions = [".jpg", ".jpeg", ".png"]
    uploaded_file_extension = Path(file.filename).suffix.lower()

    if uploaded_file_extension not in allowed_file_extensions:
        joined = ", ".join(allowed_file_extensions)
        error_msg = f"File type not allowed. Allowed file extensions: {joined}"
        return {"error": error_msg}

    resized_image = await _get_resized_image_from_file(file)

    if resized_image.size == 0:
        return {"error": "Failed to read file content"}

    predicted_class = _predict_class_for_image(resized_image)

    return {"class": predicted_class}


if __name__ == "__main__":
    api_port = int(os.environ.get("PORT", 5000))
    run(api, host="0.0.0.0", port=api_port)
