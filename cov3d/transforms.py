import scipy.ndimage
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
from scipy.interpolate import CubicSpline


class TensorBool(TensorBase):
    pass


class ReadCTScan(Transform):
    def __init__(
        self,
        width: int = None,
        height: int = None,
        max_slices: int = 128,
        channels: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if height is None:
            height = width

        self.size = None if width is None else (height, width)
        self.max_slices = max_slices
        self.channels = channels

    def encodes(self, path: Path):
        slices = sorted(
            [x for x in path.glob("*.jpg") if x.stem.isnumeric()],
            key=lambda x: int(x.stem),
        )
        num_slices = len(slices)
        assert num_slices > 0

        max_slices = self.max_slices
        if max_slices:
            if num_slices < max_slices:
                factor = max_slices // num_slices
                slices += slices * factor
                slices = slices[:max_slices]
                num_slices = len(slices)

            if num_slices > max_slices:
                stride = num_slices // max_slices
                slices = slices[::stride][:max_slices]
                num_slices = len(slices)

            if num_slices != max_slices:
                raise ValueError(f"{num_slices} != {max_slices}")

        size = self.size
        if not size:
            # Get resolution from first slice
            with Image.open(slices[0]) as im:
                size = im.size

        tensor = torch.zeros((self.channels, num_slices, size[1], size[0]))
        for index, slice in enumerate(slices):
            with Image.open(slice) as im:
                im = im.convert("RGB" if self.channels == 3 else "L")
                if im.size != size:
                    im = im.resize(size, Image.BICUBIC)
                    # raise ValueError(f"The size of image {path} ({im.size}) is not consistent with the first image in this scan {size}")
                im_data = to_tensor(im)
                tensor[:, index, :, :] = im_data

        return tensor


class ReadCTScanTricubic(Transform):
    def __init__(
        self,
        width: int = None,
        height: int = None,
        depth: int = 128,
        channels: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if height is None:
            height = width

        self.width = width
        self.height = height
        self.depth = depth
        self.channels = channels

    def encodes_old(self, path: Path):
        filename = f"{self.depth}x{self.height}x{self.width}.pt"
        tensor_path = path / filename
        if tensor_path.exists():
            return torch.load(str(tensor_path))

        slices = sorted(
            [x for x in path.glob("*.jpg") if x.stem.isnumeric()],
            key=lambda x: int(x.stem),
        )
        depth = self.depth
        if depth is None:
            depth = len(slices)
        assert depth > 0

        size = (self.height, self.width)
        with Image.open(slices[0]) as im:
            original_size = im.size

        original = np.zeros((len(slices), original_size[1], original_size[0]))
        for index, slice in enumerate(slices):
            with Image.open(slice) as im:
                im = im.convert("L")
                original[index, :, :] = np.asarray(im) / 255.0

        interpolator = tricubic.tricubic(list(original), list(original.shape))

        xs = np.linspace(0.0, original.shape[0] - 1, num=depth)
        ys = np.linspace(0.0, original.shape[1] - 1, num=size[1])
        zs = np.linspace(0.0, original.shape[2] - 1, num=size[0])

        del original

        tensor = torch.zeros((self.channels, depth, size[1], size[0]))
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                for k, z in enumerate(zs):
                    tensor[:, i, j, k] = interpolator.ip([x, y, z])

        torch.save(tensor, str(tensor_path))

        return tensor

    def encodes(self, path: Path):
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

        size = (self.height, self.width)
        original = np.zeros((len(slices), self.height, self.width))
        for index, slice in enumerate(slices):
            with Image.open(slice) as im:
                im = im.convert("L")
                if im.size != size:
                    im = im.resize(size, Image.BICUBIC)
                original[index, :, :] = np.asarray(im) / 255.0

        tensor = np.zeros((self.channels, depth, self.height, self.width))
        assert self.channels == 1
        for i in range(self.height):
            for j in range(self.width):
                if len(slices) == 1:
                    tensor[0, :, i, j] = original[0, i, j]
                else:
                    # Build interpolator
                    interpolator = CubicSpline(
                        np.linspace(0.0, 1.0, len(slices)), original[:, i, j]
                    )

                    # Interpolate along depth axis
                    tensor[0, :, i, j] = interpolator(np.linspace(0.0, 1.0, depth))

        tensor = torch.as_tensor(tensor)
        torch.save(tensor, str(tensor_path))

        return tensor.half()


class ReadCTScanMapping(Transform):
    def __init__(
        self,
        width: int = None,
        height: int = None,
        depth: int = 128,
        channels: int = 1,
        x_factor: float = 0.5,
        y_factor: float = 0.5,
        z_factor: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if height is None:
            height = width

        self.width = width
        self.height = height
        self.depth = depth
        self.channels = channels
        self.x_factor = x_factor
        self.y_factor = y_factor
        self.z_factor = z_factor

    def encodes(self, path: Path):
        filename = f"{path.name}-{self.depth}x{self.height}x{self.width}-{self.x_factor}-{self.y_factor}-{self.z_factor}.pt"
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

        size = (self.height, self.width)
        original = np.zeros((len(slices), self.height, self.width))
        for index, slice in enumerate(slices):
            with Image.open(slice) as im:
                im = im.convert("L")

                x, y = np.meshgrid(
                    np.linspace(-1.0, 1.0, im.size[0]),
                    np.linspace(-1.0, 1.0, im.size[1]),
                )  # meshgrid for interpolation mapping
                x = self.x_factor * x ** 3 + (1 - self.x_factor) * x
                y = self.y_factor * y ** 3 + (1 - self.y_factor) * y

                x = (x + 1) * im.size[0] / 2
                y = (y + 1) * im.size[1] / 2

                distorted = scipy.ndimage.map_coordinates(im, [y.ravel(), x.ravel()])
                distorted.resize(im.size)
                original[index, :, :] = (
                    np.asarray(Image.fromarray(distorted).resize(size, Image.BICUBIC))
                    / 255.0
                )

        tensor = np.zeros((self.channels, depth, self.height, self.width))
        assert self.channels == 1
        for i in range(self.height):
            for j in range(self.width):
                if len(slices) == 1:
                    tensor[0, :, i, j] = original[0, i, j]
                else:
                    # Build interpolator
                    interpolator = CubicSpline(
                        np.linspace(-1.0, 1.0, len(slices)), original[:, i, j]
                    )

                    z = np.linspace(-1.0, 1.0, depth)
                    z = self.z_factor * z ** 3 + (1 - self.z_factor) * z
                    # Interpolate along depth axis
                    tensor[0, :, i, j] = interpolator(z)

        tensor = torch.as_tensor(tensor)
        torch.save(tensor, str(tensor_path))

        return tensor.half()


def read_ct_slice(path: Path):
    path = Path(path)
    if path.is_dir():
        slices = sorted(
            [x for x in path.glob("*.jpg") if x.stem.isnumeric()],
            key=lambda x: int(x.stem),
        )
        num_slices = len(slices)
        assert num_slices > 0

        if "validation" == path.parent.parent.name:
            slice_index = num_slices // 2
            path = slices[slice_index]
        else:
            path = random.choice(slices)

    size = (256, 256)
    with Image.open(str(path)) as im:
        im = im.convert("RGB")
        if im.size != size:
            im = im.resize(size, Image.BICUBIC)
            # raise ValueError(f"The size of image {path} ({im.size}) is not consistent with the first image in this scan {size}")
        return to_tensor(im)


def bool_to_tensor(input: bool):
    return torch.FloatTensor([input])


def BoolBlock():
    return TransformBlock(
        item_tfms=[bool_to_tensor],
    )


def CTScanBlock(distortion=True, **kwargs):
    reader = ReadCTScanMapping if distortion else ReadCTScanTricubic
    return TransformBlock(
        type_tfms=reader(**kwargs),
    )


def CTSliceBlock():
    return TransformBlock(
        type_tfms=read_ct_slice,
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
