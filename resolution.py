from PIL import Image
from pathlib import Path

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

for path in paths:
    slices = sorted([x for x in path.glob('*.jpg') if x.stem.isnumeric()], key=lambda x:int(x.stem))
    with Image.open(slices[0]) as im:
        size = im.size
    print(path,len(slices), size[1], size[0], sep=",")