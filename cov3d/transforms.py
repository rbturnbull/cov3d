from fastcore.transform import Transform
from fastai.data.block import TransformBlock
from pathlib import Path
from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor
from fastai.torch_core import TensorBase
import random
import numpy as np
import tricubic


class TensorBool(TensorBase):   
    pass


class ReadCTScan(Transform):
    def __init__(self, width:int = None, height:int = None, max_slices:int=128, channels:int = 3, **kwargs):
        super().__init__(**kwargs)
        if height is None:
            height = width
        
        self.size = None if width is None else (height, width)
        self.max_slices = max_slices
        self.channels = channels

    def encodes(self, path:Path):
        slices = sorted([x for x in path.glob('*.jpg') if x.stem.isnumeric()], key=lambda x:int(x.stem))
        num_slices = len(slices)
        assert num_slices > 0

        max_slices = self.max_slices
        if max_slices:
            if num_slices < max_slices:
                factor = max_slices//num_slices
                slices += slices * factor
                slices = slices[:max_slices]
                num_slices = len(slices)

            if num_slices > max_slices:
                stride = num_slices//max_slices
                slices = slices[::stride][:max_slices]
                num_slices = len(slices)
        
            if num_slices != max_slices:
                raise ValueError(f"{num_slices} != {max_slices}")

        size = self.size
        if not size:
            # Get resolution from first slice
            with Image.open(slices[0]) as im:
                size = im.size

        tensor = torch.zeros( (self.channels, num_slices, size[1], size[0]) )
        for index, slice in enumerate(slices):
            with Image.open(slice) as im:
                im = im.convert('RGB' if self.channels == 3 else "L")
                if im.size != size:
                    im = im.resize(size,Image.BICUBIC)
                    # raise ValueError(f"The size of image {path} ({im.size}) is not consistent with the first image in this scan {size}")
                im_data = to_tensor(im)
                tensor[:,index,:,:] = im_data

        return tensor
    

class ReadCTScanTricubic(Transform):
    def __init__(self, width:int = None, height:int = None, depth:int=128, channels:int = 1, **kwargs):
        super().__init__(**kwargs)
        if height is None:
            height = width
        
        self.width = width
        self.height = height
        self.depth = depth
        self.channels = channels

    def encodes(self, path:Path):
        filename = f"{self.depth}x{self.height}x{self.width}.pt"
        tensor_path = path/filename
        if tensor_path.exists():
            return torch.load(str(tensor_path))

        slices = sorted([x for x in path.glob('*.jpg') if x.stem.isnumeric()], key=lambda x:int(x.stem))
        depth = self.depth
        if depth is None:
            depth = len(slices)
        assert depth > 0

        size = (self.height, self.width)
        with Image.open(slices[0]) as im:
            original_size = im.size

        original = np.zeros( (len(slices), original_size[1], original_size[0]) )
        for index, slice in enumerate(slices):
            with Image.open(slice) as im:
                im = im.convert("L")
                original[index,:,:] = np.asarray(im)/255.0


        interpolator = tricubic.tricubic(list(original), list(original.shape))
        
        xs = np.linspace(0.0, original.shape[0]-1, num=depth)
        ys = np.linspace(0.0, original.shape[1]-1, num=size[1])
        zs = np.linspace(0.0, original.shape[2]-1, num=size[0])

        del original

        tensor = torch.zeros( (self.channels, depth, size[1], size[0]) )
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                for k, z in enumerate(zs):
                    tensor[:,i,j,k] = interpolator.ip( [x,y,z] )

        torch.save(tensor, str(tensor_path))

        return tensor    

def read_ct_slice(path:Path):
    path = Path(path)
    if path.is_dir():
        slices = sorted([x for x in path.glob('*.jpg') if x.stem.isnumeric()], key=lambda x:int(x.stem))
        num_slices = len(slices)
        assert num_slices > 0

        if "validation" == path.parent.parent.name:
            slice_index = num_slices//2
            path = slices[slice_index]
        else:
            path = random.choice(slices)

    size = (256,256)
    with Image.open(str(path)) as im:
        im = im.convert('RGB')
        if im.size != size:
            im = im.resize(size,Image.BICUBIC)
            # raise ValueError(f"The size of image {path} ({im.size}) is not consistent with the first image in this scan {size}")
        return to_tensor(im)


def bool_to_tensor(input:bool):
    return torch.FloatTensor([input])


def BoolBlock():
    return TransformBlock(
        item_tfms=[bool_to_tensor],
    )


def CTScanBlock(**kwargs):
    return TransformBlock(
        type_tfms=ReadCTScan(**kwargs),
    )


def CTSliceBlock():
    return TransformBlock(
        type_tfms=read_ct_slice,
    )
