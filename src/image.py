import cv2
from image_classes import ImageClasses


class Image:
    def __init__(self, filename: str, mat: cv2.Mat, corresponding_class: ImageClasses):
        self._filename = filename
        self._mat = mat
        self._corresponding_class = corresponding_class

    @property
    def filename(self):
        return self._filename

    @property
    def mat(self):
        return self._mat

    @property
    def corresponding_class(self):
        return self._corresponding_class
