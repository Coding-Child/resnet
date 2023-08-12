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
                    # 패치 크기 계산 (경계를 벗어나는 경우 크기 조정)
                    patch_width = min(self.patch_size[0], w - i)
                    patch_height = min(self.patch_size[1], h - j)

                    patch = img.crop((i, j, i + patch_width, j + patch_height))
                    padded_patch = Image.new("RGB", self.patch_size, (255, 255, 255))
                    padded_patch.paste(patch, (0, 0, patch_width, patch_height))

                    patch_arr = np.array(padded_patch)

                    if np.mean(patch_arr) < self.threshold:
                        patch_count += 1
                        patch_path = os.path.join(save_folder, f'patch_{patch_count}.jpg')
                        padded_patch.save(patch_path)
                except OSError:
                    continue
        return patch_count
