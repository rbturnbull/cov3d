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
        autocrop:bool = True,
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
        self.autocrop = autocrop

    def encodes(self, path: Path):
        """
        Code used for reference: 
            https://stackoverflow.com/a/66382706
            https://github.com/bbrister/aimutil
            https://github.com/bbrister/ctOrganSegmentation/blob/master/findLungsCT.m
            https://www.kaggle.com/code/kmader/dsb-lung-segmentation-algorithm
        """
        filename = f"{path.name}-{self.depth}x{self.height}x{self.width}.pt"
        root_dir = path.parent.parent.parent
        relative_dir = path.relative_to(root_dir)
        autocrop_str = "autocrop" if self.autocrop else "no-autocrop"
        tensor_path = root_dir/ f"{self.depth}x{self.height}x{self.width}-{autocrop_str}" / relative_dir / filename
        if tensor_path.exists():
            return torch.load(str(tensor_path)).half()

        tensor_path.parent.mkdir(exist_ok=True, parents=True)

        slices = sorted(
            [x for x in path.glob("*.jpg") if x.stem.isnumeric()],
            key=lambda x: int(x.stem),
        )
        depth = self.depth
        assert depth > 0

        original_size = (512,512)
        data = np.zeros( (len(slices), original_size[1], original_size[0]), dtype=int )
        slices_to_keep = []
        for i in range(len(slices)):
            im = Image.open(path/f"{i}.jpg").convert('L')
            if im.size != original_size:
                im = im.resize(original_size, Image.BICUBIC)
            im_data = np.asarray(im)
            if np.max(im_data) > 30:
                data[i,:,:] = im_data
                slices_to_keep.append(i)

        data = data[slices_to_keep, :, :]
        if self.autocrop:

            # filter for air which is under a pixel value of a certain threshold
            binary = data < self.threshold

            # Remove exam table (see https://github.com/bbrister/ctOrganSegmentation/blob/master/findLungsCT.m)
            table_size = 5
            binary = binary_closing(binary, np.ones( (1,table_size,1) ) )

            # Find 2D seed
            a = 0.33
            b = 0.5
            centre1 = (int(data.shape[1]*a), int(data.shape[2] * b))
            centre2 = (int(data.shape[1]*(1.0-a)), int(data.shape[2] * b))

            xv, yv = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[2]))

            distance_from_center1 = np.sqrt(((xv - centre1[0])/(a*data.shape[1]))**2 + ((yv - centre1[1])/(b*data.shape[2]))**2)
            distance_from_center2 = np.sqrt(((xv - centre2[0])/(a*data.shape[1]))**2 + ((yv - centre2[1])/(b*data.shape[2]))**2)

            weights1 = 0.5*np.cos(distance_from_center1.clip(max=1.0)*np.pi) + 0.5
            weights2 = 0.5*np.cos(distance_from_center2.clip(max=1.0)*np.pi) + 0.5

            weights = weights1 + weights2

            max_result = 0.0
            max_index = -1
            seed = None
            for slice_index, slice in enumerate(binary):
                # clear border
                cleared = clear_border(slice)

                label_image, labels_count = label(cleared, return_num=True)

                max_label_result = 0.0
                for label_index in range(1, labels_count):
                    result = np.sum(weights*(label_image==label_index))
                    if result >= max_result:
                        max_index = slice_index
                        max_result = result
                        seed = label_image==label_index

            assert seed is not None

            # Remove air connected to the boundary to the front, back, right and left (not head and feet)
            end_mask = np.ones_like(binary, dtype=bool)
            end_mask[:,0,:] = False
            end_mask[:,-1,:] = False
            end_mask[:,:,0] = False
            end_mask[:,:,-1] = False
            boundary = clear_border(binary, mask=end_mask) ^ binary
            boundary = binary_closing( boundary, ball(3, decomposition="sequence") )
            removed_boundary = False
            if boundary[max_index, seed].sum() == 0:
                binary[boundary] = 0
                removed_boundary = True

            # Dilate a bit to fill in holes and to join the lungs if necessary
            dilated = binary_dilation(binary, ball(3, decomposition="sequence"))

            # label regions
            label_image = label(dilated)

            # find label with seed
            seed_label = label_image[max_index, seed]
            assert seed_label.min() == seed_label.max()
            seed_label = seed_label[0]

            lungs = label_image == seed_label

            # find bounds of segmented lungs
            for start_i in range(lungs.shape[0]):
                if lungs[start_i,:,:].sum() > 0:
                    break

            for end_i in reversed(range(lungs.shape[0])):
                if lungs[end_i,:,:].sum() > 0:
                    break

            for start_j in range(lungs.shape[1]):
                if lungs[:,start_j,:].sum() > 0:
                    break

            for end_j in reversed(range(lungs.shape[1])):
                if lungs[:,end_j,:].sum() > 0:
                    break

            for start_k in range(lungs.shape[2]):
                if lungs[:,:,start_k].sum() > 0:
                    break

            for end_k in reversed(range(lungs.shape[2])):
                if lungs[:,:,end_k].sum() > 0:
                    break

            # crop original data according to the bounds of the segmented lungs
            # also scale from zero to one
            lungs_cropped = data[ start_i:end_i+1, start_j:end_j+1, start_k:end_k+1]/255.0

            crop_log = tensor_path.parent/f"{path.name}.crop.txt"
            crop_log.write_text(f"{relative_dir},{len(slices)},{start_i},{end_i},{start_j},{end_j},{start_k},{end_k},{data.size},{lungs_cropped.size},{lungs_cropped.size/data.size*100.0},{removed_boundary}\n")

            # Save seed
            rgb = np.repeat( np.expand_dims(data[max_index], axis=2), 3, axis=2)
            rgb[start_j:end_j+1, start_k:end_k+1, 2 ] = 0
            rgb[seed, 2 ] = 0
            rgb[seed, 1 ] = 0
            im = Image.fromarray(rgb.astype(np.uint8))
            seed_dir = root_dir/ f"{self.depth}x{self.height}x{self.width}" / "seed" / relative_dir
            seed_dir.mkdir(exist_ok=True, parents=True)
            seed_path = seed_dir/f"{path.name}.seed-i.jpg"
            im.save(seed_path)    

            rgb = np.repeat( np.expand_dims(data[:,int(start_j/2+end_j/2),:], axis=2), 3, axis=2)
            rgb[start_i:end_i+1, start_k:end_k+1, 2 ] = 0
            rgb[max_index-start_i, start_k:end_k+1, 1 ] = 0
            im = Image.fromarray(rgb.astype(np.uint8))
            seed_path = seed_dir/f"{path.name}.seed-j.jpg"
            im.save(seed_path)    

            rgb = np.repeat( np.expand_dims(data[:,:,int(start_k/4+end_k/4)], axis=2), 3, axis=2)
            rgb[start_i:end_i+1, start_j:end_j+1, 2 ] = 0
            rgb[max_index-start_i, start_j:end_j+1, 1 ] = 0
            im = Image.fromarray(rgb.astype(np.uint8))
            seed_path = seed_dir/f"{path.name}.seed-k.jpg"
            im.save(seed_path)    

            data = lungs_cropped

        data = resize(data, (self.depth,self.height,self.width), order=3)        

        tensor = torch.unsqueeze(torch.as_tensor(data), dim=0)
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
