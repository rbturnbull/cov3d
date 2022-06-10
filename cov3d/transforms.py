from fastai.data.block import TransformBlock
from pathlib import Path
from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor

def read_ct_scans(path:Path):
    slices = sorted(path.glob('*.jpg'), key=lambda x:int(x.stem))
    num_slices = len(slices)
    assert num_slices > 0

    # Get resolution from first slice
    with Image.open(slices[0]) as im:
        size = im.size

    tensor = torch.zeros( (1, num_slices, size[1], size[0]), dtype=torch.float16 )
    for index, slice in enumerate(slices):
        with Image.open(slice) as im:
            im = im.convert('L')
            im_data = to_tensor(im).half()
            tensor[:,index,:,:] = im_data

    return tensor
    

def BinaryBlock():
    return TransformBlock(
        # type_tfms=read3D,
    )


def CTScanBlock():
    return TransformBlock(
        type_tfms=read_ct_scans,
    )

