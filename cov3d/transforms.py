from fastai.data.block import TransformBlock
from pathlib import Path
from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor
from fastai.torch_core import TensorBase


class TensorBool(TensorBase):   
    pass


def read_ct_scans(path:Path):
    slices = sorted([x for x in path.glob('*.jpg') if x.stem.isnumeric()], key=lambda x:int(x.stem))
    num_slices = len(slices)
    assert num_slices > 0

    max_slices = 0
    if max_slices:
        if num_slices < max_slices:
            factor = max_slices//num_slices
            slices += slices * factor
            slices = slices[:max_slices]
            num_slices = len(slices)

        if num_slices > max_slices:
            start = (num_slices-max_slices)//2
            slices = slices[start:start+max_slices]
            num_slices = len(slices)
    
        if num_slices != max_slices:
            raise ValueError(f"{num_slices} != {max_slices}")

    size = (128,128)
    if not size:
        # Get resolution from first slice
        with Image.open(slices[0]) as im:
            size = im.size

    tensor = torch.zeros( (1, num_slices, size[1], size[0]) )
    for index, slice in enumerate(slices):
        with Image.open(slice) as im:
            im = im.convert('L')
            if im.size != size:
                im = im.resize(size,Image.BICUBIC)
                # raise ValueError(f"The size of image {path} ({im.size}) is not consistent with the first image in this scan {size}")
            im_data = to_tensor(im)
            tensor[:,index,:,:] = im_data

    return tensor
    

def bool_to_tensor(input:bool):
    return torch.FloatTensor([input])


def BoolBlock():
    return TransformBlock(
        item_tfms=[bool_to_tensor],
    )


def CTScanBlock():
    return TransformBlock(
        type_tfms=read_ct_scans,
    )

