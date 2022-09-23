import os
import cv2
import configs
from typing import Dict, List
from image_classes import ImageClasses
from image_filenames import ImageFilenames


image_dirs: Dict[ImageClasses, str] = {
    ImageClasses.Fresh: configs.fresh_images_folder_path,
    ImageClasses.Spoiled: configs.spoiled_images_folder_path,
}


def _get_resized_image(folder_path: str, filename: str) -> cv2.Mat:
    image_path = os.path.join(folder_path, filename)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(image, (configs.image_width, configs.image_height))


class Images:
    def __init__(self, image_filenames: ImageFilenames):
        self._all_filenames = image_filenames.grouped_filenames
        self._images: List[cv2.Mat] = []
        self._image_classes: List[ImageClasses] = []
        self._fill()

    def _fill_images_for_class(self, c: ImageClasses):
        slice_size = configs.training_sample_size
        for filename in self._all_filenames[c][0:slice_size]:
            image = _get_resized_image(image_dirs[c], filename)
            self._images.append(image)
            self._image_classes.append(c.value)

    def _fill(self):
        self._fill_images_for_class(ImageClasses.Fresh)
        self._fill_images_for_class(ImageClasses.Spoiled)

    @property
    def files(self) -> List[cv2.Mat]:
        return self._images

    @property
    def classes(self) -> List[ImageClasses]:
        return self._image_classes
