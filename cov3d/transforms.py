from fastai.data.block import TransformBlock


def BinaryBlock():
    return TransformBlock(
        # type_tfms=read3D,
    )


def CTScanBlock():
    return TransformBlock(
        type_tfms=read_ct_scans,
    )

