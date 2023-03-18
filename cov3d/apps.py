import random
import re
from pathlib import Path
from torch import nn
from typing import List
from fastai.data.core import DataLoaders
from fastai.data.transforms import GrandparentSplitter, get_image_files, FuncSplitter, IndexSplitter
from fastai.data.core import DataLoaders
from fastai.data.block import DataBlock, CategoryBlock, TransformBlock
from fastai.metrics import accuracy, Precision, Recall, F1Score
import torch
import pandas as pd
from torchapp.util import call_func
from fastai.learner import load_learner
import torchapp as ta
from fastcore.foundation import L
from rich.console import Console

console = Console()
from torchapp.metrics import logit_f1, logit_accuracy
from pytorchvideo.models.head import create_res_basic_head
from fastcore.transform import Pipeline
from fastcore.foundation import mask2idxs
from fastai.callback.preds import MCDropoutCallback

from torchvision.models import video
from .callbacks import ExportLearnerCallback
from .transforms import (
    CTScanBlock,
    BoolBlock,
    Normalize,
    Flip,
    AdjustBrightness,
    AdjustContrast,
    Clip,
)
from .models import ResNet3d, update_first_layer, PositionalEncoding3D, adapt_stoic_model
from .loss import Cov3dLoss, EarthMoverLoss, FocalLoss, FocalEMDLoss
from .metrics import (
    SeverityF1,
    PresenceF1,
    SeverityAccuracy,
    PresenceAccuracy,
    # severity_probability_to_category,
    MildF1,
    ModerateF1,
    SevereF1,
    CriticalF1,
    NonCovidF1,
    CovidF1,

)


def get_y(scan_path: Path):
    parent_name = scan_path.parent.name
    if parent_name == "covid":
        return True
    if parent_name == "non-covid":
        return False
    raise Exception(
        f"Cannot determine whether sample '{scan_path}' has covid or not from the path."
    )


class DictionaryGetter:
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __call__(self, key):
        if key not in self.dictionary:
            raise ValueError(f"key {key} ({type(key)}) not in dictionary")
        return self.dictionary[key]


class DictionarySplitter:
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __call__(self, objects):
        validation_indexes = mask2idxs(self.dictionary[object] for object in objects)
        return IndexSplitter(validation_indexes)(objects)


def is_validation(scan_path: Path):
    if not scan_path.is_dir():
        scan_path = scan_path.parent

    return scan_path.parent.parent.name.startswith("validation")


class Cov3dCombinedGetter:
    def __init__(self, severity_dictionary):
        self.severity_dictionary = severity_dictionary

    def __call__(self, scan_path):
        if not scan_path.is_dir():
            scan_path = scan_path.parent
        parent_name = scan_path.parent.name
        if parent_name == "covid":
            has_covid = True
        elif parent_name == "non-covid":
            has_covid = False
        else:
            raise Exception(
                f"Cannot determine whether sample '{scan_path}' has covid or not from the path."
            )

        return torch.as_tensor([has_covid, self.severity_dictionary.get(scan_path, 0)])


class Cov3dCombinedGetter:
    def __init__(self, severity_dictionary):
        self.severity_dictionary = severity_dictionary

    def __call__(self, scan_path):
        if not scan_path.is_dir():
            scan_path = scan_path.parent
        parent_name = scan_path.parent.name
        if parent_name == "covid":
            has_covid = True
        elif parent_name == "non-covid":
            has_covid = False
        else:
            raise Exception(
                f"Cannot determine whether sample '{scan_path}' has covid or not from the path."
            )

        return torch.as_tensor([has_covid, self.severity_dictionary.get(scan_path, 0)])


class Cov3d(ta.TorchApp):
    """
    A deep learning model to detect the presence and severity of COVID19 in patients from CT-scans.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.severity_factor = 0.5 # will be overridden by dataloader
        self.depth = 64 # will be overridden by dataloader
        self.width = 128 # will be overridden by dataloader
        self.height = 128 # will be overridden by dataloader
        self.train_non_covid_count = 1 # will be overridden by dataloader
        self.train_covid_count = 1 # will be overridden by dataloader
        self.train_mild_count = 1 # will be overridden by dataloader
        self.train_moderate_count = 1 # will be overridden by dataloader
        self.train_severe_count = 1 # will be overridden by dataloader
        self.train_critical_count = 1 # will be overridden by dataloader

    def dataloaders(
        self,
        directory: Path = ta.Param(help="The data directory."),
        batch_size: int = ta.Param(default=4, help="The batch size."),
        splits_csv: Path = ta.Param(
            None,
            help="The path to a file which contains the cross-validation splits."
        ),
        split: int = ta.Param(
            0,
            help="The cross-validation split to use. The default (i.e. 0) is the original validation set."
        ),
        training_severity: Path = ta.Param(
            None,
            help="The path to the training Excel file with severity information."
        ),
        validation_severity: Path = ta.Param(
            None,
            help="The path to the validation Excel file with severity information."
        ),
        width: int = ta.Param(default=128, help="The width to convert the images to."),
        height: int = ta.Param(
            default=None,
            help="The height to convert the images to. If None, then it is the same as the width.",
        ),
        depth: int = ta.Param(
            default=128, help="The depth of the 3d volume to interpolate to."
        ),
        normalize: bool = ta.Param(
            False,
            help="Whether or not to normalize the pixel data by the mean and std of the dataset.",
        ),
        severity_factor: float = 0.5,
        flip: bool = False,
        brightness: float = 0.0,
        contrast: float = 0.0,
        distortion: bool = True,
        autocrop:bool = True,
        max_scans:int = 0,
    ) -> DataLoaders:
        """
        Creates a FastAI DataLoaders object which Cov3d uses in training and prediction.

        Returns:
            DataLoaders: The DataLoaders object.
        """
        directory = Path(directory).resolve()
        paths = []
        self.severity_factor = severity_factor

        # Try loading cross validation csv
        if splits_csv is None:
            splits_csv = Path("cross-validation.csv")
            if not splits_csv.exists():
                splits_csv = None

        if splits_csv:
            splits_df = pd.read_csv(splits_csv)
            paths = [directory/path for path in splits_df['path']]
            validation_dict = {path:split == s for path, s in zip(paths, splits_df['split'])}
            splitter = DictionarySplitter(validation_dict)

            train_df = splits_df[splits_df['split']!=split]

            def get_count(train_df, string, value):
                try:
                    return len(train_df[train_df['category'].str.lower() == string])
                except Exception:
                    return len(train_df[train_df['category'].astype(int) == value])
            
            self.train_non_covid_count = get_count(train_df, "non-covid", 0)
            self.train_mild_count = get_count(train_df, "mild", 1)
            self.train_moderate_count = get_count(train_df, "moderate", 2)
            self.train_severe_count = get_count(train_df, "severe", 3)
            self.train_critical_count = get_count(train_df, "critical", 4)
            self.train_covid_count = get_count(train_df, "covid", 5)

            assert self.train_non_covid_count > 0
            assert self.train_mild_count > 0
            assert self.train_moderate_count > 0
            # assert self.train_severe_count > 0
            # assert self.train_critical_count > 0
            # assert self.train_covid_count > 0

            # self.counts = torch.tensor([
            #     self.train_non_covid_count,
            #     self.train_mild_count,
            #     self.train_moderate_count,
            #     self.train_severe_count,
            #     self.train_critical_count,
            #     self.train_covid_count,
            # ])
            self.counts_binary = torch.tensor([
                self.train_non_covid_count,
                self.train_mild_count +
                self.train_moderate_count +
                self.train_severe_count +
                self.train_critical_count +
                self.train_covid_count
            ])

            # self.weights = self.counts.sum()/(len(self.counts)*self.counts)
            self.weights_binary = self.counts_binary.sum()/(len(self.counts_binary)*self.counts_binary)
            self.categories_count = 6 if self.train_severe_count or self.train_critical_count or self.train_covid_count else 3
            self.weights = torch.zeros( (self.categories_count,) )
            self.weights[0] = self.weights_binary[0]
            self.weights[1:] = self.weights_binary[1]

            category_dict = dict()
            for path, category in zip(paths, splits_df['category'].astype(str).str.lower()):
                c = 0
                if category.isdigit():
                    c = int(category)
                elif category == "non-covid":
                    c = 0
                elif category == "mild":
                    c = 1
                elif category == "moderate":
                    c = 2
                elif category == "severe":
                    c = 3
                elif category == "critical":
                    c = 4
                elif category == "covid":
                    c = 5
                else:
                    raise ValueError(f"Cannot understand category {category}")

                category_dict[path] = c
        else:
            if split != 0:
                raise ValueError(f"Not cross validation split file found so using split {split} is not possible.")
            raise Exception("use a cross validation split file")
            subdirs = [
                "train/covid",
                "train/non-covid",
                "validation/covid",
                "validation/non-covid",
            ]

            severity = dict()

            def read_severity(file: Path, dir: str):
                df = pd.read_excel(file)
                for _, row in df.iterrows():
                    path = directory / f"{dir}/covid" / row["Name"]
                    if not path.exists():
                        raise FileNotFoundError(f"Cannot find directory {path}")
                    severity[path] = row["Category"]

            training_severity = training_severity or directory/"ICASSP_severity_train_partition.xlsx"
            validation_severity = validation_severity or directory/"ICASSP_severity_validation_partition.xlsx"
            read_severity(training_severity, dir="train")
            read_severity(validation_severity, dir="validation")

            for s in subdirs:
                subdir = directory / s
                if not subdir.exists():
                    raise FileNotFoundError(f"Cannot find directory '{subdir}'.")
                subdir_paths = [
                    path for path in subdir.iterdir() if path.name.startswith("ct_scan")
                ]
                if len(subdir_paths) == 0:
                    raise FileNotFoundError(
                        f"Cannot file directories with prefix 'ct_scan' in {subdir}"
                    )

                # if self.severity_factor >= 1.0:
                #     subdir_paths = [p for p in subdir_paths if p in severity]

                paths += subdir_paths

                if s == "train/covid":
                    self.train_covid_count = len(subdir_paths)
                elif s == "train/non-covid":
                    self.train_non_covid_count = len(subdir_paths)

            splitter = FuncSplitter(is_validation)

        batch_tfms = []

        if normalize:
            batch_tfms.append(Normalize())

        self.width = width
        self.height = height or width
        self.depth = depth

        item_tfms = []

        if contrast > 0.0:
            item_tfms.append(AdjustContrast(sigma=contrast))

        if brightness > 0.0:
            item_tfms.append(AdjustBrightness(std=brightness))

        # item_tfms.append(Clip())

        if flip:
            item_tfms.append(Flip)

        datablock = DataBlock(
            blocks=(
                CTScanBlock(
                    width=width, 
                    height=height, 
                    depth=depth,
                    autocrop=autocrop,
                ),
                TransformBlock,
            ),
            splitter=splitter,
            get_y=DictionaryGetter(category_dict),
            batch_tfms=batch_tfms,
            item_tfms=item_tfms,
        )

        if max_scans:
            random.seed(1)
            random.shuffle(paths)
            paths = paths[:max_scans]

        dataloaders = DataLoaders.from_dblock(
            datablock,
            source=paths,
            bs=batch_size,
        )

        dataloaders.c = 2 # self.categories_count # is this used??
        return dataloaders

    # def dataloaders_old(
    #     self,
    #     directory: Path = ta.Param(help="The data directory."),
    #     batch_size: int = ta.Param(default=4, help="The batch size."),
    #     training_severity: Path = ta.Param(
    #         None,
    #         help="The path to the training Excel file with severity information."
    #     ),
    #     validation_severity: Path = ta.Param(
    #         None,
    #         help="The path to the validation Excel file with severity information."
    #     ),
    #     width: int = ta.Param(default=128, help="The width to convert the images to."),
    #     height: int = ta.Param(
    #         default=None,
    #         help="The height to convert the images to. If None, then it is the same as the width.",
    #     ),
    #     depth: int = ta.Param(
    #         default=128, help="The depth of the 3d volume to interpolate to."
    #     ),
    #     normalize: bool = ta.Param(
    #         False,
    #         help="Whether or not to normalize the pixel data by the mean and std of the dataset.",
    #     ),
    #     severity_factor: float = 0.5,
    #     flip: bool = False,
    #     distortion: bool = True,
    #     autocrop:bool = True,
    # ) -> DataLoaders:
    #     """
    #     Creates a FastAI DataLoaders object which Cov3d uses in training and prediction.

    #     Returns:
    #         DataLoaders: The DataLoaders object.
    #     """
    #     directory = Path(directory).resolve()
    #     paths = []
    #     self.severity_factor = severity_factor

    #     subdirs = [
    #         "train/covid",
    #         "train/non-covid",
    #         "validation/covid",
    #         "validation/non-covid",
    #     ]

    #     severity = dict()

    #     def read_severity(file: Path, dir: str):
    #         df = pd.read_excel(file)
    #         for _, row in df.iterrows():
    #             path = directory / f"{dir}/covid" / row["Name"]
    #             if not path.exists():
    #                 raise FileNotFoundError(f"Cannot find directory {path}")
    #             severity[path] = row["Category"]

    #     training_severity = training_severity or directory/"ICASSP_severity_train_partition.xlsx"
    #     validation_severity = validation_severity or directory/"ICASSP_severity_validation_partition.xlsx"
    #     read_severity(training_severity, dir="train")
    #     read_severity(validation_severity, dir="validation")

    #     for s in subdirs:
    #         subdir = directory / s
    #         if not subdir.exists():
    #             raise FileNotFoundError(f"Cannot find directory '{subdir}'.")
    #         subdir_paths = [
    #             path for path in subdir.iterdir() if path.name.startswith("ct_scan")
    #         ]
    #         if len(subdir_paths) == 0:
    #             raise FileNotFoundError(
    #                 f"Cannot file directories with prefix 'ct_scan' in {subdir}"
    #             )

    #         # if self.severity_factor >= 1.0:
    #         #     subdir_paths = [p for p in subdir_paths if p in severity]

    #         paths += subdir_paths

    #         if s == "train/covid":
    #             self.train_covid_count = len(subdir_paths)
    #         elif s == "train/non-covid":
    #             self.train_non_covid_count = len(subdir_paths)

    #     batch_tfms = []
    #     if normalize:
    #         batch_tfms.append(Normalize())

    #     self.width = width
    #     self.height = height or width
    #     self.depth = depth

    #     item_tfms = []
    #     if flip:
    #         item_tfms.append(Flip)

    #     datablock = DataBlock(
    #         blocks=(
    #             CTScanBlock(
    #                 width=width, 
    #                 height=height, 
    #                 depth=depth,
    #                 autocrop=autocrop,
    #             ),
    #             TransformBlock,
    #         ),
    #         splitter=FuncSplitter(is_validation),
    #         get_y=Cov3dCombinedGetter(severity),
    #         batch_tfms=batch_tfms,
    #         item_tfms=item_tfms,
    #     )

    #     dataloaders = DataLoaders.from_dblock(
    #         datablock,
    #         source=paths,
    #         bs=batch_size,
    #     )

    #     dataloaders.c = 2
    #     return dataloaders

    def model(
        self,
        model_name: str = "r3d_18",
        pretrained:bool = True,
        penultimate: int = 512,
        dropout: float = 0.5,
        max_pool: bool = False,
        severity_regression: bool = False,
        final_bias: bool = False,
        fine_tune: bool = False,
        flatten: bool = False,
        even_stride: bool = False,
        positional_encoding: bool = False,
        cov3d_trained: Path = None,
        severity_everything: bool = False,
    ) -> nn.Module:
        """
        Creates a deep learning model for the Cov3d to use.

        Returns:
           nn.Module: The created model.
        """

        in_channels = 1
        self.positional_encoding = positional_encoding
        if self.positional_encoding:
            in_channels += 3

        self.severity_regression = severity_regression
        self.severity_everything = severity_everything
        out_features = 3 if self.categories_count == 3 else 5
        # if self.severity_factor > 0.0:
        #     if severity_everything:
        #         out_features += 5
        #     elif severity_regression:
        #         out_features += 1
        #     else:
        #         out_features += 4

        self.fine_tune = fine_tune and pretrained

        if Path(model_name).exists():
            model_path = Path(model_name)
            import dill
            pretrained_learner = load_learner(model_path, cpu=False, pickle_module=dill)
            model = pretrained_learner.model
            adapt_stoic_model(model)
            return model
        elif model_name in ("r3d_18", "mc3_18", "r2plus1d_18", "mvit_v1_b", "mvit_v2_s", "s3d"):
            # https://pytorch.org/vision/stable/models.html
            # https://pytorch.org/vision/stable/models/video_resnet.html
            # https://pytorch.org/vision/stable/models/video_mvit.html
            # https://pytorch.org/vision/stable/models/video_s3d.html
            # https://pytorch.org/vision/master/models/video_swin_transformer.html
            get_model = getattr(video, model_name)
            model = get_model(weights='DEFAULT' if pretrained else None)
            update_first_layer(model, in_channels, pretrained=pretrained)

            if even_stride:
                first_layer = next(model.stem.children())
                first_layer.stride = (2, 2, 2)

            if max_pool:
                model.avgpool = torch.nn.AdaptiveMaxPool3d((1, 1, 1))

            if self.fine_tune:
                for param in model.parameters():
                    param.requires_grad = False

            blocks = (self.depth // 8) * (self.width // 16) * (self.height // 16)
            if flatten:
                model = nn.Sequential(
                    model.stem,
                    model.layer1,
                    nn.Dropout(dropout),
                    model.layer2,
                    nn.Dropout(dropout),
                    model.layer3,
                    nn.Dropout(dropout),
                    model.layer4,
                    nn.Dropout(dropout),
                    nn.Conv3d(
                        in_channels=512, out_channels=4, kernel_size=1, bias=True
                    ),
                    nn.ReLU(),
                    nn.Flatten(1),
                    nn.Linear(
                        in_features=4 * blocks,
                        out_features=out_features,
                        bias=final_bias,
                    ),
                )
            else:
                model.layer1 = nn.Sequential(
                    model.layer1,
                    nn.Dropout(dropout),
                )
                model.layer2 = nn.Sequential(
                    model.layer2,
                    nn.Dropout(dropout),
                )
                model.layer3 = nn.Sequential(
                    model.layer3,
                    nn.Dropout(dropout),
                )
                model.layer4 = nn.Sequential(
                    model.layer4,
                    nn.Dropout(dropout),
                )

                if penultimate:
                    model.fc = nn.Sequential(
                        nn.Linear(
                            in_features=model.fc.in_features,
                            out_features=penultimate,
                            bias=True,
                        ),
                        nn.Dropout(dropout),
                        nn.ReLU(),
                        nn.Linear(
                            in_features=penultimate,
                            out_features=out_features,
                            bias=final_bias,
                        ),
                    )
                else:
                    model.fc = nn.Linear(
                        in_features=model.fc.in_features,
                        out_features=out_features,
                        bias=final_bias,
                    )
        elif model_name in ("swin3d_t", "swin3d_s", "swin3d_b"):
            # https://pytorch.org/vision/master/models/video_swin_transformer.html
            get_model = getattr(video, model_name)
            model = get_model(weights='DEFAULT' if pretrained else None)
            update_first_layer(model, in_channels, pretrained=pretrained) # model.patch_embed.proj
            model.head = nn.Linear(
                in_features=model.head.in_features,
                out_features=out_features,
                bias=final_bias,
            )
        else:
            model = torch.hub.load(
                "facebookresearch/pytorchvideo", model=model_name, pretrained=pretrained
            )
            update_first_layer(model, in_channels, pretrained=pretrained)
            if self.fine_tune:
                for param in model.parameters():
                    param.requires_grad = False
            model.blocks[-1] = create_res_basic_head(
                in_features=2048, out_features=out_features, pool_kernel_size=(4, 4, 4)
            )

        if self.positional_encoding:
            model = nn.Sequential(
                PositionalEncoding3D(),
                model,
            )

        if cov3d_trained is not None:
            state_dict = torch.load(cov3d_trained)
            model.load_state_dict(state_dict)
            return model

        return model

    def extra_callbacks(self):
        return [ExportLearnerCallback(monitor="presence_f1"), ExportLearnerCallback(monitor="severity_f1")]

    def loss_func(
        self,
        presence_smoothing: float = 0.1,
        severity_smoothing: float = 0.1,
        neighbour_smoothing: bool = False,
        mse: bool = False,
        emd_weight:float=0.1,
    ):
        if emd_weight == 0.0:
            return FocalLoss()

        return FocalEMDLoss(
            distances=[1,1] if self.categories_count == 3 else [1,1,1,1],
            distance_negative_to_positive=1,
            square=False,
            emd_weight=emd_weight,
            # weights=self.weights
        )
        # return EarthMoverLoss(
        #     distances=[1,1,1,1],
        #     distance_negative_to_positive=1,
        #     square=False,
        #     # weights=self.weights,
        # )
        # pos_weight = self.train_non_covid_count / self.train_covid_count
        # return Cov3dLoss(
        #     pos_weight=torch.as_tensor(
        #         [pos_weight]
        #     ).cuda(),  # hack - this should be to the device of the other tensors
        #     severity_factor=self.severity_factor,
        #     severity_regression=self.severity_regression,
        #     presence_smoothing=presence_smoothing,
        #     severity_smoothing=severity_smoothing,
        #     neighbour_smoothing=neighbour_smoothing,
        #     severity_everything=self.severity_everything,
        #     mse=mse,
        # )

    def metrics(self):
        metrics = [
            PresenceF1(),
            SeverityF1(),
            PresenceAccuracy(),
            SeverityAccuracy(),
            MildF1(),
            ModerateF1(),
            NonCovidF1(),
            CovidF1(),
        ]

        if self.categories_count > 3:
            metrics += [
                SevereF1(),
                CriticalF1(),
            ]

        return metrics

    def monitor(self):
        return "presence_f1"

    def fit(
        self,
        learner,
        callbacks,
        epochs: int = ta.Param(default=20, help="The number of epochs."),
        freeze_epochs: int = ta.Param(
            default=3,
            help="The number of epochs to train when the learner is frozen and the last layer is trained by itself. Only if `fine_tune` is set on the app.",
        ),
        learning_rate: float = ta.Param(
            default=1e-4,
            help="The base learning rate (when fine tuning) or the max learning rate otherwise.",
        ),
        **kwargs,
    ):
        if self.fine_tune:
            learner.fine_tune(
                0,
                freeze_epochs=freeze_epochs,
                base_lr=learning_rate,
                cbs=callbacks,
                **kwargs,
            )  # hack
            for param in learner.model.parameters():
                param.requires_grad = True
            return learner.fine_tune(
                epochs, freeze_epochs=0, base_lr=learning_rate, cbs=callbacks, **kwargs
            )  # hack

        return learner.fit_one_cycle(
            epochs, lr_max=learning_rate, cbs=callbacks, **kwargs
        )

    def learner_kwargs(
        self,
        output_dir: Path = ta.Param(
            "./outputs", help="The location of the output directory."
        ),
        weight_decay: float = ta.Param(
            None,
            help="The amount of weight decay. If None then it uses the default amount of weight decay in fastai.",
        ),
        **kwargs,
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        return dict(
            loss_func=call_func(self.loss_func, **kwargs),
            metrics=call_func(self.metrics, **kwargs),
            path=output_dir,
            wd=weight_decay,
        )

    def __call__(
        self,
        gpu: bool = ta.Param(
            True, help="Whether or not to use a GPU for processing if available."
        ),
        mc_samples: int = ta.Param(
            0, help="The number of Monte Carlo samples of the results to get."
        ),
        mc_dropout: bool = ta.Param(
            False,
            help="Whether or not to use Monte Carlo dropout if doing MC sampling.",
        ),
        **kwargs,
    ):
        # Open the exported learner from a pickle file
        path = call_func(self.pretrained_local_path, **kwargs)
        try:
            learner = self.learner_obj = load_learner(path, cpu=not gpu)
        except Exception:
            import dill
            learner = self.learner_obj = load_learner(path, cpu=not gpu, pickle_module=dill)

        callbacks_to_remove = [
            "ExportLearnerCallback",
            "TorchAppWandbCallback",
            "SaveModelCallback",
        ]
        learner.cbs = L([callback for callback in learner.cbs if type(callback).__name__ not in callbacks_to_remove])

        # Create a dataloader for inference
        dataloader = call_func(self.inference_dataloader, learner, **kwargs)
        results_samples = []

        if mc_samples:
            callbacks = [MCDropoutCallback()] if mc_dropout else []

            for index in range(mc_samples):
                print(f"Monte Carlo sample run {index}")
                results, _ = learner.get_preds(
                    dl=dataloader,
                    reorder=False,
                    with_decoded=False,
                    act=self.activation(),
                    cbs=callbacks,
                )
                results_samples += [results]

        else:
            dataloader.after_item = Pipeline()
            results, _ = learner.get_preds(
                dl=dataloader, reorder=False, with_decoded=False, act=self.activation()
            )
            results_samples.append(results)

            dataloader.after_item = Pipeline(Flip(always=True))
            results, _ = learner.get_preds(
                dl=dataloader, reorder=False, with_decoded=False, act=self.activation()
            )
            results_samples.append(results)

        results = torch.stack(results_samples, dim=1)

        # Output results
        return call_func(self.output_results, results, **kwargs)

    def inference_dataloader(
        self,
        learner,
        scan: List[Path] = ta.Param(None, help="A directory of a CT scan."),
        scan_dir: List[Path] = ta.Param(
            None,
            help="A directory with CT scans in subdirectories. Subdirectories must start with 'ct_scan' or 'test_ct_scan'.",
        ),
        directory:Path=None,
        **kwargs,
    ):
        self.scans = []
        if isinstance(scan, (str, Path)):
            scan = [Path(scan)]
        if isinstance(scan_dir, (str, Path)):
            scan_dir = [Path(scan_dir)]

        for s in scan:
            self.scans.append(Path(s))

        if directory:
            directory = Path(directory)
            self.scans = [directory/path for path in self.scans]

        if scan_dir:
            for sdir in scan_dir:
                sdir = Path(sdir)
                if directory:
                    sdir = directory/sdir
                self.scans += [
                    path
                    for path in sdir.iterdir()
                    if path.is_dir() or path.suffix == ".mha"
                ]

        

        dataloader = learner.dls.test_dl(self.scans)

        return dataloader

    def output_results(
        self,
        results,
        output_csv: Path = ta.Param(
            default=None, help="A path to output the results as a CSV."
        ),
        output_mc: Path = ta.Param(
            default=None,
            help="A path to output all MC inference runs as a PyTorch tensor.",
        ),
        covid_txt: Path = ta.Param(
            default=None,
            help="A path to output the names of the predicted COVID positive scans.",
        ),
        noncovid_txt: Path = ta.Param(
            default=None,
            help="A path to output the names of the predicted COVID negative scans.",
        ),
        mild_txt: Path = ta.Param(
            default=None,
            help="A path to output the names of the predicted mild COVID scans.",
        ),
        moderate_txt: Path = ta.Param(
            default=None,
            help="A path to output the names of the predicted moderate COVID scans.",
        ),
        severe_txt: Path = ta.Param(
            default=None,
            help="A path to output the names of the predicted severe COVID scans.",
        ),
        critical_txt: Path = ta.Param(
            default=None,
            help="A path to output the names of the predicted critical COVID scans.",
        ),
        **kwargs,
    ):
        results_df_data = []

        if output_mc:
            print(f"Saving MC inference output to: {output_mc}")
            torch.save(results, str(output_mc))

        columns = [
            "name",
            "COVID19 positive",
            "probability",
            "mc_samples_positive",
            "severity",
            "mild_samples",
            "moderate_samples",
            "severe_samples",
            "critical_samples",
            "mild_probability",
            "moderate_probability",
            "severe_probability",
            "critical_probability",
            "mc_samples_total",
            "path",
        ]
        for path, result in zip(self.scans, results):
            sample_probabilties = torch.softmax(result, dim=1)
            average_probabilties = sample_probabilties.mean(dim=0)
            
            positive = average_probabilties[0] < 0.5
            probability_positive = 1.0-average_probabilties[0]

            mc_samples_total = result.shape[0]
            mc_samples_positive = (sample_probabilties[:, 0] < 0.5).sum() / mc_samples_total

            severity_categories = ["mild", "moderate", "severe", "critical"]
            average_severity_probabilities = average_probabilties[1:]/(1.0-average_probabilties[:1])
            sample_severity_probabilities = sample_probabilties[:,1:]/(1.0-sample_probabilties[:,:1])
            severity_id = torch.argmax(average_severity_probabilities, dim=-1).item()
            sample_severity_ids = torch.argmax(sample_severity_probabilities, dim=1)

            severity = severity_categories[severity_id]
            mild_samples = (sample_severity_ids == 0).sum() / mc_samples_total
            moderate_samples = (sample_severity_ids == 1).sum() / mc_samples_total
            severe_samples = (sample_severity_ids == 2).sum() / mc_samples_total
            critical_samples = (sample_severity_ids == 3).sum() / mc_samples_total

            results_df_data.append(
                [
                    path.name,
                    positive.item(),
                    probability_positive.item(),
                    mc_samples_positive.item(),
                    severity,
                    mild_samples.item(),
                    moderate_samples.item(),
                    severe_samples.item(),
                    critical_samples.item(),
                    average_severity_probabilities[0].item(),
                    average_severity_probabilities[1].item(),
                    average_severity_probabilities[2].item(),
                    average_severity_probabilities[3].item(),
                    mc_samples_total,
                    path,
                ]
            )

        results_df = pd.DataFrame(results_df_data, columns=columns)

        if output_csv:
            console.print(
                f"Writing results for {len(results_df)} sequences to: {output_csv}"
            )
            results_df.to_csv(output_csv, index=False)

        def get_digits(string):
            m = re.search(r"\d+", string)
            if m:
                return int(m.group(0))
            return -1

        def write_scans_txt(filename, mask):
            if filename:
                scans = results_df[mask]["name"].tolist()
                scans = sorted(scans, key=get_digits)
                print(f"writing to {filename}")
                with open(filename, "w") as f:
                    f.write("\n".join(scans) + "\n")

        write_scans_txt(covid_txt, results_df["COVID19 positive"] == True)
        write_scans_txt(noncovid_txt, results_df["COVID19 positive"] == False)
        write_scans_txt(mild_txt, results_df["severity"] == "mild")
        write_scans_txt(moderate_txt, results_df["severity"] == "moderate")
        write_scans_txt(severe_txt, results_df["severity"] == "severe")
        write_scans_txt(critical_txt, results_df["severity"] == "critical")

        print(results_df)
        print(f"COVID19 Positive: {results_df['COVID19 positive'].sum()}")
        print(
            f"COVID19 Negative: {len(results_df) - results_df['COVID19 positive'].sum()}"
        )

        return results_df


class Cov3dSeverity(Cov3d):
    def monitor(self):
        return "severity_f1"
