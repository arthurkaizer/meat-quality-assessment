import os
import cv2
import numpy
from typing import List

import configs
from image_classes import ImageClasses
from image import Image


_classes_and_dirs = {
    ImageClasses.Fresh: configs.fresh_images_folder_path,
    ImageClasses.Spoiled: configs.spoiled_images_folder_path
}


class Images:
    def __init__(self):
        self._images: List[Image] = []
        self._fill_images_for_class(ImageClasses.Fresh)
        self._fill_images_for_class(ImageClasses.Spoiled)

    def _fill_images_for_class(self, c: ImageClasses):
        for file in os.listdir(_classes_and_dirs[c]):
            if not file.endswith(".jpg"):
                continue

            class_folder_path = _classes_and_dirs[c]
            file_path = os.path.join(class_folder_path, file)
            image = Images._get_image_at(file_path)
            self._images.append(Image(file, image, c))

    def _get_image_at(path: str) -> cv2.Mat:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resize_to = (configs.image_width, configs.image_height)
        return cv2.resize(image, resize_to)

    def get_classes(self) -> List[str]:
        return list(map(lambda i: i.corresponding_class.value, self._images))

    def get_files(self) -> numpy.ndarray[cv2.Mat]:
        files = list(map(lambda i: i.mat, self._images))
        return numpy.array(files)
