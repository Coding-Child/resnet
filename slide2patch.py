import os
from tqdm import tqdm
from patch_extract.patch_extract import PatchExtractor

root_path = os.listdir('cancer_data')
for direct in root_path:
    root_dir = os.path.join('cancer_data/', direct)
    paths = os.listdir(root_dir)
    full_path = list()

    for img in paths:
        full_path.append(os.path.join(root_dir, img))

    extractor = PatchExtractor(threshold=245)

    with tqdm(total=len(full_path), desc=f"Extract in {direct}") as pbar:
        for img in full_path:
            if direct == 'LN':
                idx = 1
            else:
                idx = 0
            extractor.extract(img, f'data/{idx}')
            pbar.update(1)
