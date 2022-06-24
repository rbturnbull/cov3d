from PIL import Image
from pathlib import Path
import numpy as np

directory = ".."

directory = Path(directory).resolve()
paths = []

subdirs = ["train/covid", "train/non-covid", "validation/covid", "validation/non-covid"]


for s in subdirs:
    subdir = directory/s
    if not subdir.exists():
        raise FileNotFoundError(f"Cannot find directory '{subdir}'.")
    subdir_paths = [path for path in subdir.iterdir() if path.name.startswith("ct_scan")]
    if len(subdir_paths) == 0:
        raise FileNotFoundError(f"Cannot file directories with prefix 'ct_scan' in {subdir}")
    paths += subdir_paths

pixel_count = 0
pixel_sum = 0.0

pixel_sum_squared = 0.0

for path in paths:
    print(path)
    slices = sorted([x for x in path.glob('*.jpg') if x.stem.isnumeric()], key=lambda x:int(x.stem))
    for slice in slices:
        with Image.open(slice) as im:
            pixel_count += im.size[0] * im.size[1]
            pixels = np.asarray(im)/255.0
            pixel_sum += np.sum(pixels)
            pixel_sum_squared += np.sum(pixels**2)


mean = pixel_sum/pixel_count
var = (pixel_sum_squared / pixel_count) - (mean ** 2)
std = np.sqrt(var)

print("mean:", mean)
print("std:", std)