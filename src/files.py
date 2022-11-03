import os
import cv2
from typing import List, Tuple

import configs
from image_classes import ImageClasses


_classes_and_dirs = {
    ImageClasses.Fresh: configs.fresh_images_folder_path,
    ImageClasses.Spoiled: configs.spoiled_images_folder_path
}


class Files:
    @staticmethod
    def _get_image_at(path: str) -> cv2.Mat:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resize_to = (configs.image_width, configs.image_height)
        return cv2.resize(image, resize_to)

    @staticmethod
    def get_files_and_corresponding_classes() -> Tuple[List[cv2.Mat], List[ImageClasses]]:
        images: List[cv2.Mat] = []
        classes: List[ImageClasses] = []

        for c, dir in _classes_and_dirs.items():
            for file in os.listdir(dir):
                if file.endswith(".jpg"):
                    image_path = os.path.join(dir, file)
                    image = Files._get_image_at(image_path)
                    images.append(image)
                    classes.append(c.value)

        return (images, classes)
