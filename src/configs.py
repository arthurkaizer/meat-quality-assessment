import os

_working_dir = os.getcwd()

training_sample_size = 500
number_of_training_epochs = 2
image_width = 100
image_height = 100
assets_folder_path = os.path.join(_working_dir, "assets")
fresh_images_folder_path = os.path.join(assets_folder_path, "Fresh")
spoiled_images_folder_path = os.path.join(assets_folder_path, "Spoiled")
