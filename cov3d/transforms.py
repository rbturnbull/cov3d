from fastcore.transform import Transform
from fastai.data.block import TransformBlock
from pathlib import Path
from PIL import Image
import torch
from fastai.torch_core import TensorBase
import random
import numpy as np

from skimage.segmentation import clear_border
from skimage.morphology import ball, binary_closing, binary_dilation, binary_opening
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
        threshold: int = 90,
        fp16: bool = True,
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
        self.fp16 = fp16

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
        autocrop_str = "-autocrop4" if self.autocrop else ""
        fp16_str = "-fp16" if self.fp16 else ""
        preprossed_dir = root_dir/ f"{self.depth}x{self.height}x{self.width}{autocrop_str}{fp16_str}"
        tensor_path = preprossed_dir / relative_dir / filename
        if tensor_path.exists():
            x = torch.load(str(tensor_path))
            return x

        tensor_path.parent.mkdir(exist_ok=True, parents=True)

        if path.is_dir():
            assert path.name.startswith("ct_scan")
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

            slice_count = len(slices)
            
        elif path.suffix == ".mha": # stoic dataset
            from medpy.io import load
            data, image_header = load(path)
            data = data.transpose(2,0,1)

            # scale so that it is the same as the challenge dataset
            data = (data.clip(min=-1150, max=350)+1150)/(350+1150)*255.0
            slice_count = data.shape[0]
            # Rotate and flip so that it is the same as the challenge dataset
            data[0] = torch.rot90(data[0], dims=(2,1))
            data = torch.flip(data, dims=[1])

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
            
            max_result = [0.0,0.0]
            max_index = [-1, -1]
            seed = [None,None]
            for slice_index, slice in enumerate(binary):
                # clear border
                cleared = clear_border(slice)

                label_image, labels_count = label(cleared, return_num=True)

                max_label_result = 0.0
                for label_index in range(1, labels_count):
                    for i, weights in enumerate([weights1, weights2]):
                        result = np.sum(weights*(label_image==label_index))
                        if result >= max_result[i]:
                            max_index[i] = slice_index
                            max_result[i] = result
                            seed[i] = label_image==label_index

            assert seed[0] is not None
            assert seed[1] is not None

            # Remove air connected to the boundary to the front, back, right and left (not head and feet)
            def get_boundary(data, threshold=70, closing=True, erosion_radius=None, jstart=False, jend=False, kstart=False, kend=False, show=False):
                binary_strict = data < threshold
                if erosion_radius:
                    binary_strict = binary_opening(binary_strict, ball(erosion_radius, decomposition="sequence"))

                end_mask = np.ones_like(binary, dtype=bool)
                end_mask[:,0,:] = jstart
                end_mask[:,-1,:] = jend
                end_mask[:,:,0] = kstart
                end_mask[:,:,-1] = kend
                boundary = clear_border(binary_strict, mask=end_mask) ^ binary_strict
                if closing:
                    boundary = binary_closing( boundary, ball(3, decomposition="sequence") )

                if show:
                    plot_volume(boundary, n_steps=3).show()
                    sys.exit()

                return boundary
                
            # Try to remove boundary if not in seed
            removed_boundary = False
            # Try to remove boundary if not in seed
            removed_boundary = False
            threshold = self.threshold
            def remove_boundary(data, binary, threshold):
                # Try to remove boundary if not in seed
                total_boundary = np.zeros_like(binary, dtype=bool)
                while threshold > 0:
                    for erosion_radius in [0,5,10,15]:
                        print(f"trying to clear boundary at threshold {threshold}, erosion {erosion_radius}")
                        
                        boundary = get_boundary(data, threshold=threshold, jstart=False, jend=False, kstart=False, kend=False, erosion_radius=erosion_radius)
                        if boundary[max_index[0], seed[0]].sum() == 0 and boundary[max_index[1], seed[1]].sum() == 0:
                            print("clearing boundary")
                            binary[boundary] = 0
                            total_boundary = total_boundary | boundary
                            break
                    
                    if erosion_radius == 0:
                        break
                    
                    threshold -= 10
                binary[total_boundary] = 0
                return binary

            binary = remove_boundary(data, binary, threshold=threshold)

            # Dilate a bit to fill in holes and to join the lungs if necessary
            dilated = binary_dilation(binary, ball(3, decomposition="sequence"))

            # label regions
            label_image = label(dilated)

            # find label with seed
            seed_label = label_image[max_index[0], seed[0]]
            assert seed_label.min() == seed_label.max()
            seed_label_a = seed_label[0]

            seed_label = label_image[max_index[1], seed[1]]
            assert seed_label.min() == seed_label.max()
            seed_label_b = seed_label[0]

            lungs = (label_image == seed_label_a) | (label_image == seed_label_b)

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
            crop_log.write_text(f"{relative_dir},{slice_count},{start_i},{end_i},{start_j},{end_j},{start_k},{end_k},{data.size},{lungs_cropped.size},{lungs_cropped.size/data.size*100.0}\n")

            # Save seeds
            for i in range(2):
                rgb = np.repeat( np.expand_dims(data[max_index[i]], axis=2), 3, axis=2)
                rgb[start_j:end_j+1, start_k:end_k+1, 2 ] = 0
                rgb[seed[i], 2 ] = 0
                rgb[seed[i], 1 ] = 0
                im = Image.fromarray(rgb.astype(np.uint8))
                seed_dir = preprossed_dir / "seed" / relative_dir
                seed_dir.mkdir(exist_ok=True, parents=True)
                seed_path = seed_dir/f"{path.name}.seed-i-{i}.jpg"
                im.save(seed_path)    

                rgb = np.repeat( np.expand_dims(data[:,int(start_j/2+end_j/2),:], axis=2), 3, axis=2)
                rgb[start_i:end_i+1, start_k:end_k+1, 2 ] = 0
                rgb[max_index[i]-start_i, start_k:end_k+1, 1 ] = 0
                im = Image.fromarray(rgb.astype(np.uint8))
                seed_path = seed_dir/f"{path.name}.seed-j-{i}.jpg"
                im.save(seed_path)    

                rgb = np.repeat( np.expand_dims(data[:,:,int(start_k/4+end_k/4)], axis=2), 3, axis=2)
                rgb[start_i:end_i+1, start_j:end_j+1, 2 ] = 0
                rgb[max_index[i]-start_i, start_j:end_j+1, 1 ] = 0
                im = Image.fromarray(rgb.astype(np.uint8))
                seed_path = seed_dir/f"{path.name}.seed-k-{i}.jpg"
                im.save(seed_path)    

            data = lungs_cropped

        data = resize(data, (self.depth,self.height,self.width), order=3)        

        tensor = torch.unsqueeze(torch.as_tensor(data), dim=0)
        if self.fp16:
            tensor = tensor.half()
        print("save", str(tensor_path))
        torch.save(tensor, str(tensor_path))

        return tensor


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
            not isinstance(x, torch.Tensor) or len(x.shape) < 4
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


class AdjustContrast(Transform):
    def __init__(self, sigma: float = 0.03, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma

    def encodes(self, x):
        if (
            not isinstance(x, torch.Tensor) or len(x.shape) < 4
        ):  # hack so that it just works on the 3d input. This should be done with type dispatching
            return x

        return np.random.lognormal(sigma=self.sigma) * (x - 0.5) + 0.5


class AdjustBrightness(Transform):
    def __init__(self, std: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.std = std

    def encodes(self, x):
        if (
            not isinstance(x, torch.Tensor) or len(x.shape) < 4
        ):  # hack so that it just works on the 3d input. This should be done with type dispatching
            return x

        return x + np.random.normal(0.0, self.std)


class Clip(Transform):
    def encodes(self, x):
        if (
            not isinstance(x, torch.Tensor) or len(x.shape) < 4
        ):  # hack so that it just works on the 3d input. This should be done with type dispatching
            return x

        return x.clip(min=0.0,max=1.0)
