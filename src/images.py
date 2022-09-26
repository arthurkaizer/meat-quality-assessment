import os
import cv2
from typing import List

import configs
from image_classes import ImageClasses


_classes_and_dirs = {
    ImageClasses.Fresh: configs.fresh_images_folder_path,
    ImageClasses.Spoiled: configs.spoiled_images_folder_path
}


class Images:
    def __init__(self):
        self._images: List[cv2.Mat] = []
        self._classes: List[ImageClasses] = []
        self._fill_images_for_class(ImageClasses.Fresh)
        self._fill_images_for_class(ImageClasses.Spoiled)

    def _get_image_at(path: str) -> cv2.Mat:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resize_to = (configs.image_width, configs.image_height)
        return cv2.resize(image, resize_to)

    def _fill_images_for_class(self, c: ImageClasses):
        dir_path = _classes_and_dirs[c]

        for file in os.listdir(dir_path):
            if file.endswith(".jpg"):
                image_path = os.path.join(dir_path, file)
                image = Images._get_image_at(image_path)
                self._images.append(image)
                self._classes.append(c.value)

    @property
    def classes(self):
        return self._classes

    @property
    def images(self):
        return self._images
