from fastai.data.block import TransformBlock
from pathlib import Path
from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor
from fastai.torch_core import TensorBase


class TensorBool(TensorBase):   
    pass


def read_ct_scans(path:Path):
    slices = sorted(path.glob('*.jpg'), key=lambda x:int(x.stem))
    num_slices = len(slices)
    assert num_slices > 0

    # Get resolution from first slice
    with Image.open(slices[0]) as im:
        size = im.size

    tensor = torch.zeros( (1, num_slices, size[1], size[0]) )
    for index, slice in enumerate(slices):
        with Image.open(slice) as im:
            im = im.convert('L')
            im_data = to_tensor(im)
            tensor[:,index,:,:] = im_data

    return tensor
    

def bool_to_tensor(input:bool):
    return TensorBool([input]).float()


def BoolBlock():
    return TransformBlock(
        item_tfms=[bool_to_tensor],
    )


def CTScanBlock():
    return TransformBlock(
        type_tfms=read_ct_scans,
    )

