from pathlib import Path
from torch import nn
from fastai.metrics import accuracy
from fastai.data.core import DataLoaders
from fastai.data.transforms import GrandparentSplitter
from fastai.data.core import DataLoaders
from fastai.data.block import DataBlock
from fastai.metrics import accuracy, Precision, Recall, F1Score
import torch
from fastapp.util import call_func
import fastapp as fa
from rich.console import Console
console = Console()

from .transforms import CTScanBlock, BoolBlock
from .models import ResNet3d

def get_y(scan_path:Path):
    parent_name = scan_path.parent.name
    if parent_name == "covid":
        return True
    if parent_name == "non-covid":
        return False
    raise Exception(f"Cannot determine whether sample '{scan_path}' has covid or not from the path.")


class Cov3d(fa.FastApp):
    """
    A deep learning model to detect the presence and severity of COVID19 in patients from CT-scans.
    """
    def dataloaders(
        self,
        directory:Path = fa.Param(help="The data directory."), 
        batch_size:int = fa.Param(default=4, help="The batch size."),
    ) -> DataLoaders:
        """
        Creates a FastAI DataLoaders object which Cov3d uses in training and prediction.

        Args:
            directory (Path): The data directory.
            batch_size (int, optional): The number of elements to use in a batch for training and prediction. Defaults to 32.

        Returns:
            DataLoaders: The DataLoaders object.
        """
        directory = Path(directory).resolve()
        subdirs = ["train/covid", "train/non-covid", "validation/covid", "validation/non-covid"]
        paths = []
        for s in subdirs:
            subdir = directory/s
            if not subdir.exists():
                raise FileNotFoundError(f"Cannot find directory '{subdir}'.")
            subdir_paths = [path for path in subdir.iterdir() if path.name.startswith("ct_scan")]
            if len(subdir_paths) == 0:
                raise FileNotFoundError(f"Cannot file directories with prefix 'ct_scan' in {subdir}")
            paths += subdir_paths

        datablock = DataBlock(
            blocks=(CTScanBlock, BoolBlock),
            splitter=GrandparentSplitter(train_name='train', valid_name='validation'),
            get_y=get_y,
        )

        dataloaders = DataLoaders.from_dblock(
            datablock, 
            source=paths,
            bs=batch_size,
        )

        return dataloaders    

    def model(
        self,
        initial_features:int = fa.Param(default=64, tune=True, tune_min=16, tune_max=256, help="The number of features in the initial CNN layer."),
    ) -> nn.Module:
        """
        Creates a deep learning model for the Cov3d to use.

        Returns:
           nn.Module: The created model.
        """ 
        return ResNet3d(
            initial_features=initial_features,
        )

    def loss_func(self):
        return nn.BCEWithLogitsLoss()

    def metrics(self):
        average = "macro"
        return [
            accuracy,
            F1Score(average=average),
            Precision(average=average),
            Recall(average=average),
        ]

    def monitor(self):
        return "f1_score"
