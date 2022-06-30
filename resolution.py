from pathlib import Path
from cov3d.transforms import ReadCTScan, ReadCTScanTricubic, ReadCTScanMapping
import random

directory = ".."

directory = Path(directory).resolve()
paths = []

# subdirs = ["train/covid", "train/non-covid", "validation/covid", "validation/non-covid"]
subdirs = [
    "/data/gpfs/projects/punim1293/rob/covid-ct/testseverity/part_2_test_set_ECCV_22",
    "/data/gpfs/projects/punim0639/rob/test/presence",
    "/data/gpfs/projects/punim0639/rob/test/presence/part1_1_set_test_ECCV",
    "/data/gpfs/projects/punim0639/rob/test/presence/part1_2_test_set_ECCV",
    "/data/gpfs/projects/punim0639/rob/test/presence/part_3_test_set_ECCV",
]


for s in subdirs:
    subdir = directory/s
    if not subdir.exists():
        raise FileNotFoundError(f"Cannot find directory '{subdir}'.")
    subdir_paths = [path for path in subdir.iterdir() if path.name.startswith("ct_scan") or path.name.startswith("test_ct_scan")]
    if len(subdir_paths) == 0:
        raise FileNotFoundError(f"Cannot file directories with prefix 'ct_scan' or 'test_ct_scan' in {subdir}")
    paths += subdir_paths

reader = ReadCTScanTricubic(width=320, depth=160)

random.shuffle(paths)
for path in paths:
    print(path)
    reader(path)
    # slices = sorted([x for x in path.glob('*.jpg') if x.stem.isnumeric()], key=lambda x:int(x.stem))
    # with Image.open(slices[0]) as im:
    #     size = im.size
    # print(path,len(slices), size[1], size[0], sep=",")
