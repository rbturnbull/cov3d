from pathlib import Path
from torch import nn
from fastai.data.core import DataLoaders
from fastai.data.transforms import GrandparentSplitter
from fastai.data.core import DataLoaders
from fastai.data.block import DataBlock, CategoryBlock
from fastai.metrics import accuracy, Precision, Recall, F1Score
import torch
import pandas as pd
from fastapp.util import call_func
import fastapp as fa
from rich.console import Console
console = Console()
from fastapp.metrics import logit_f1, logit_accuracy

from .transforms import CTScanBlock, BoolBlock
from .models import ResNet3d


def get_y(scan_path:Path):
    parent_name = scan_path.parent.name
    if parent_name == "covid":
        return True
    if parent_name == "non-covid":
        return False
    raise Exception(f"Cannot determine whether sample '{scan_path}' has covid or not from the path.")


class DictionaryGetter():
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __call__(self, key):
        return self.dictionary[key]


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
        return [
            logit_f1,
            logit_accuracy,
        ]

    def monitor(self):
        return "logit_f1"



class Cov3dSeverity(fa.FastApp):
    """
    A deep learning model to detect the presence and severity of COVID19 in patients from CT-scans.
    """
    def dataloaders(
        self,
        directory:Path = fa.Param(help="The data directory."),
        batch_size:int = fa.Param(default=4, help="The batch size."),
        training_csv:Path = fa.Param(help="The path to the training CSV file with severity information."),
        validation_csv:Path = fa.Param(help="The path to the validation CSV file with severity information."),
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
        paths = []

        # subdirs = ["train/covid", "train/non-covid", "validation/covid", "validation/non-covid"]
        # for s in subdirs:
        #     subdir = directory/s
        #     if not subdir.exists():
        #         raise FileNotFoundError(f"Cannot find directory '{subdir}'.")
        #     subdir_paths = [path for path in subdir.iterdir() if path.name.startswith("ct_scan")]
        #     if len(subdir_paths) == 0:
        #         raise FileNotFoundError(f"Cannot file directories with prefix 'ct_scan' in {subdir}")
        #     paths += subdir_paths

        severity = dict()

        df = pd.read_csv(training_csv)
        for _, row in df.iterrows():
            path = directory/"train/covid"/row['Name']
            if not path.exists():
                raise FileNotFoundError(f"Cannot find directory {path}")
            paths.append(path)
            severity[path] = row['Category']
        
        df = pd.read_csv(validation_csv)
        for _, row in df.iterrows():
            path = directory/"validation/covid"/row['Name']
            if not path.exists():
                raise FileNotFoundError(f"Cannot find directory {path}")
            paths.append(path)
            severity[path] = row['Category']


        datablock = DataBlock(
            blocks=(CTScanBlock, CategoryBlock),
            splitter=GrandparentSplitter(train_name='train', valid_name='validation'),
            get_y=DictionaryGetter(severity),
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
            num_classes=4,
        )

    def loss_func(self):
        return nn.CrossEntropyLoss()

    def metrics(self):
        return [
            accuracy,
        ]

    def monitor(self):
        return "accuracy"
