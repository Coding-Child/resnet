import os
from tqdm import tqdm
from patch_extract.patch_extract import PatchExtractor

root_dir = 'cancer_data/LN'
paths = os.listdir(root_dir)
full_path = list()

for img in paths:
    full_path.append(os.path.join(root_dir, img))

extractor = PatchExtractor()

with tqdm(total=len(full_path), desc="Extract") as pbar:
    for img in full_path:
        extractor.extract(img, 'data/1')
        pbar.update(1)
