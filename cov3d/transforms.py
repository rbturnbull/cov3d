from fastcore.transform import Transform
from fastai.data.block import TransformBlock
from pathlib import Path
from PIL import Image
import torch
from fastai.torch_core import TensorBase
import random
import numpy as np

from skimage.segmentation import clear_border
from skimage.morphology import ball, binary_closing, binary_dilation
from skimage.measure import label
from skimage.transform import resize



class TensorBool(TensorBase):
    pass



class ReadCTScanCrop(Transform):
    def __init__(
        self,
        width: int = None,
        height: int = None,
        depth: int = 128,
        channels: int = 1,
        threshold: int = 70,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if height is None:
            height = width

        self.width = width
        self.height = height
        self.depth = depth
        self.channels = channels
        self.threshold = threshold

    def encodes(self, path: Path):
        """
        Code used for reference: 
            https://github.com/bbrister/ctOrganSegmentation/blob/master/findLungsCT.m
            https://www.kaggle.com/code/kmader/dsb-lung-segmentation-algorithm
        """
        filename = f"{path.name}-{self.depth}x{self.height}x{self.width}.pt"
        tensor_path = path / filename
        if tensor_path.exists():
            return torch.load(str(tensor_path)).half()

        slices = sorted(
            [x for x in path.glob("*.jpg") if x.stem.isnumeric()],
            key=lambda x: int(x.stem),
        )
        depth = self.depth
        if depth is None:
            depth = len(slices)
        assert depth > 0

        original_size = (512,512)
        data = np.zeros( (slices, original_size[1], original_size[0]), dtype=int )
        for i in range(slices):
            im = Image.open(f"{i}.jpg").convert('L')
            if im.size != original_size:
                im = im.resize(original_size, Image.BICUBIC)
            im_data = np.asarray(im)
            data[i,:,:] = im_data

        # filter for air which is under a pixel value of a certain threshold
        binary = data < self.threshold

        # Remove exam table (see https://github.com/bbrister/ctOrganSegmentation/blob/master/findLungsCT.m)
        table_size = 5
        binary = binary_closing(binary, np.ones( (1,table_size,1) ) )


        # Remove air connected to the boundary to the front, back, right and left (not head and feet)
        end_mask = np.ones_like(binary, dtype=bool)
        end_mask[:,0,:] = False
        end_mask[:,-1,:] = False
        end_mask[:,:,0] = False
        end_mask[:,:,-1] = False
        end_mask
        cleared = clear_border(binary, mask=end_mask)

        # Dilate a bit to fill in holes and to join the lungs if necessary
        dilated = binary_dilation(cleared, ball(3, decomposition="sequence"))

        # label regions
        label_image = label(dilated)

        # find the biggest region which we will call the lungs
        lungs = label_image == np.argmax(np.bincount(label_image.flat)[1:]) + 1

        # find bounds of segmented lungs
        for start_i in range(lungs.shape[0]):
            if lungs[start_i,:,:].sum() > 0:
                print('start_i', start_i)
                break

        for end_i in reversed(range(lungs.shape[0])):
            if lungs[end_i,:,:].sum() > 0:
                print('end_i', end_i)
                break


        for start_j in range(lungs.shape[1]):
            if lungs[:,start_j,:].sum() > 0:
                print('start_j', start_j)
                break

        for end_j in reversed(range(lungs.shape[1])):
            if lungs[:,end_j,:].sum() > 0:
                print('end_j', end_j)
                break

        for start_k in range(lungs.shape[2]):
            if lungs[:,:,start_k].sum() > 0:
                print('start_k', start_k)
                break

        for end_k in reversed(range(lungs.shape[2])):
            if lungs[:,:,end_k].sum() > 0:
                print('end_k', end_k)
                break

        # crop original data according to the bounds of the segmented lungs
        # also scale from zero to one
        lungs_cropped = data[ start_i:end_i+1, start_j:end_j+1, start_k:end_k+1]/255.0

        lungs_resized = resize(lungs_cropped, (self.depth,self.height,self.width), order=3)        

        tensor = torch.as_tensor(tensor)
        torch.save(tensor, str(tensor_path))

        return tensor.half()


def bool_to_tensor(input: bool):
    return torch.FloatTensor([input])


def BoolBlock():
    return TransformBlock(
        item_tfms=[bool_to_tensor],
    )


def CTScanBlock(**kwargs):
    reader = ReadCTScanCrop
    return TransformBlock(
        type_tfms=reader(**kwargs),
    )


class Normalize(Transform):
    def __init__(self, mean=0.39467953936257016, std=0.32745144936428694):
        self.mean = mean
        self.std = std

    def encodes(self, x):
        if (
            len(x.shape) < 4
        ):  # hack so that it just works on the 3d input. This should be done with type dispatching
            return x

        return (x - self.mean) / self.std

    def decodes(self, x):
        if (
            len(x.shape) < 4
        ):  # hack so that it just works on the 3d input. This should be done with type dispatching
            return x

        return (x * self.std) + self.mean


class Flip(Transform):
    def __init__(self, always: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.always = always

    def encodes(self, x):
        if (
            len(x.shape) < 4
        ):  # hack so that it just works on the 3d input. This should be done with type dispatching
            return x

        dims = []
        # if random.getrandbits(1):
        #     dims.append(1)
        # if random.getrandbits(1):
        #     dims.append(2)
        if random.getrandbits(1) or self.always:
            dims.append(3)

        return torch.flip(x, dims=dims)
