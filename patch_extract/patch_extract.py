import os
import numpy as np
from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


class PatchExtractor:
    def __init__(self, patch_size: tuple = (224, 224), threshold: int = 240):
        self.patch_size = patch_size
        self.threshold = threshold

    def extract(self, img_path, save_folder):
        os.makedirs(save_folder, exist_ok=True)

        img = Image.open(img_path)

        w, h = img.size
        patch_count = 0

        for i in range(0, w, self.patch_size[0]):
            for j in range(0, h, self.patch_size[1]):
                try:
                    patch = img.crop((i, j, i + self.patch_size[0], j + self.patch_size[1]))
                    patch_arr = np.array(patch.convert("RGB"))

                    if np.mean(patch_arr) < self.threshold:
                        patch_count += 1
                        patch.save(os.path.join(save_folder, f'patch_{patch_count}.jpg'))
                except OSError:
                    continue
