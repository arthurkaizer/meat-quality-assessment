import os
import configs
from typing import Dict, List
from image_classes import ImageClasses


def _get_jpg_files_in_dir(dir_path: str) -> List[str]:
    files: List[str] = []

    for file in os.listdir(dir_path):
        if file.endswith(".jpg"):
            files.append(file)

    return files


class ImageFilenames:
    def __init__(self):
        self._files: Dict[ImageClasses, List[str]] = {}
        self._fill()

    def _fill(self):
        fresh_images = _get_jpg_files_in_dir(configs.fresh_images_folder_path)
        spoiled_images = _get_jpg_files_in_dir(configs.spoiled_images_folder_path)
        self._files[ImageClasses.Fresh] = fresh_images
        self._files[ImageClasses.Spoiled] = spoiled_images

    @property
    def grouped_filenames(self) -> Dict[ImageClasses, List[str]]:
        return self._files
