from enum import Enum


class ImageClasses(str, Enum):
    Fresh = "Fresh"
    Spoiled = "Spoiled"