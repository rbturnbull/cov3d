from pathlib import Path
from torch import nn
from fastai.data.core import DataLoaders
from fastai.data.transforms import GrandparentSplitter, get_image_files, FuncSplitter
from fastai.data.core import DataLoaders
from fastai.data.block import DataBlock, CategoryBlock, TransformBlock
from fastai.metrics import accuracy, Precision, Recall, F1Score
import torch
import pandas as pd
from fastapp.util import call_func
from fastapp.vision import VisionApp
import fastapp as fa
from rich.console import Console
console = Console()
from fastapp.metrics import logit_f1, logit_accuracy
from pytorchvideo.models.head import create_res_basic_head

from torchvision.models import video

from .transforms import CTScanBlock, BoolBlock, CTSliceBlock, ReadCTScanTricubic, Normalize, Flip
from .models import ResNet3d, update_first_layer, PositionalEncoding3D
from .loss import Cov3dLoss
from .metrics import SeverityF1, PresenceF1, SeverityAccuracy, PresenceAccuracy



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
        return self.dictionary[key]-1


def is_validation(scan_path:Path):
    if not scan_path.is_dir():
        scan_path = scan_path.parent
    
    return  scan_path.parent.parent.name.startswith("validation")


class Cov3dCombinedGetter():
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
            raise Exception(f"Cannot determine whether sample '{scan_path}' has covid or not from the path.")

        return torch.as_tensor([has_covid, self.severity_dictionary.get(scan_path,0)])


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
        width:int = fa.Param(default=256, help="The width to convert the images to."),
        height:int = fa.Param(default=None, help="The height to convert the images to. If None, then it is the same as the width."),
        depth:int = fa.Param(default=128, help="The depth of the 3d volume to interpolate to."),
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

        severity = dict()

        df = pd.read_csv(training_csv, delimiter=";")
        for _, row in df.iterrows():
            path = directory/"train/covid"/row['Name']
            if not path.exists():
                raise FileNotFoundError(f"Cannot find directory {path}")
            paths.append(path)
            severity[path] = row['Category']
        
        n_classes = 4
        self.weights = torch.FloatTensor( n_classes )
        for i in range(n_classes):
            self.weights[i] = len(df)/(n_classes * (df['Category'].astype(int) == i+1).sum())

        df = pd.read_csv(validation_csv, delimiter=";")
        for _, row in df.iterrows():
            path = directory/"validation/covid"/row['Name']
            if not path.exists():
                raise FileNotFoundError(f"Cannot find directory {path}")
            paths.append(path)
            severity[path] = row['Category']

        datablock = DataBlock(
            blocks=(CTScanBlock(width=width, height=height, depth=depth), CategoryBlock),
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
        model_name:str = "r3d_18",
        pretrained:bool = True,
        penultimate:int = 512,
        dropout:float = 0.5,
        max_pool:bool = True,
        fine_tune:bool = False,
    ) -> nn.Module:
        """
        Creates a deep learning model for the Cov3d to use.

        Returns:
           nn.Module: The created model.
        """ 
        get_model = getattr(video, model_name)
        self.fine_tune = fine_tune and pretrained
        model = get_model(pretrained=pretrained)
        update_first_layer(model)
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

        if max_pool:
            model.avgpool = torch.nn.AdaptiveMaxPool3d( (1,1,1) )

        model.fc = nn.Sequential(
            nn.Linear(in_features=model.fc.in_features, out_features=penultimate, bias=True),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(in_features=penultimate, out_features=4, bias=False),
        )

        return model

    def loss_func(self):
        return nn.CrossEntropyLoss(label_smoothing=0.1, weight=self.weights)

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


class Cov3dSlice(VisionApp):
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
            blocks=(CTSliceBlock, CategoryBlock),
            splitter=GrandparentSplitter(train_name='train', valid_name='validation'),
            get_y=get_y,
        )

        dataloaders = DataLoaders.from_dblock(
            datablock, 
            source=paths,
            bs=batch_size,
        )
        dataloaders.c = 2
        return dataloaders    

    # def model(
    #     self,
    #     initial_features:int = fa.Param(default=64, tune=True, tune_min=16, tune_max=256, help="The number of features in the initial CNN layer."),
    # ) -> nn.Module:
    #     """
    #     Creates a deep learning model for the Cov3d to use.

    #     Returns:
    #        nn.Module: The created model.
    #     """ 
    #     return ResNet3d(
    #         initial_features=initial_features,
    #     )

    # def loss_func(self):
    #     return nn.BCEWithLogitsLoss()

    def metrics(self):
        average="macro"
        return [
            accuracy,
            F1Score(average=average),
        ]

    def monitor(self):
        return "f1_score"

    def inference_dataloader(self, learner, **kwargs):
        self.inference_images = get_image_files(Path("../validation/non-covid/"))
        # self.inference_images = list({x.parent for x in self.inference_images})
        dataloader = learner.dls.test_dl(self.inference_images)
        self.categories = learner.dls.vocab
        return dataloader

    def output_results(
        self,
        results,
        output_csv: Path = fa.Param(default=None, help="A path to output the results as a CSV."),
        **kwargs,
    ):
        results_df = pd.DataFrame(results[0].numpy(), columns=self.categories)
        results_df["image"] = self.inference_images
        results_df["scan"] = [image.parent.name for image in self.inference_images]
        predictions = torch.argmax(results[0], dim=1)
        results_df['prediction'] = [self.categories[p] for p in predictions]

        if not output_csv:
            raise Exception("No output file given.")

        console.print(f"Writing results for {len(results_df)} sequences to: {output_csv}")
        results_df.to_csv(output_csv)


class Cov3dSliceSeverity(Cov3dSlice):
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

        severity = dict()

        df = pd.read_csv(training_csv, delimiter=";")
        for _, row in df.iterrows():
            path = directory/"train/covid"/row['Name']
            if not path.exists():
                raise FileNotFoundError(f"Cannot find directory {path}")
            paths.append(path)
            severity[path] = row['Category']
        
        df = pd.read_csv(validation_csv, delimiter=";")
        for _, row in df.iterrows():
            path = directory/"validation/covid"/row['Name']
            if not path.exists():
                raise FileNotFoundError(f"Cannot find directory {path}")
            paths.append(path)
            severity[path] = row['Category']

        datablock = DataBlock(
            blocks=(CTSliceBlock, CategoryBlock),
            splitter=GrandparentSplitter(train_name='train', valid_name='validation'),
            get_y=DictionaryGetter(severity),
        )

        dataloaders = DataLoaders.from_dblock(
            datablock, 
            source=paths,
            bs=batch_size,
        )
        dataloaders.c = 4

        return dataloaders    

    def inference_dataloader(self, learner, **kwargs):
        df = pd.read_csv("../val_partition_covid_categories.csv", sep=";")
        self.inference_images = []
        for index, row in df.iterrows():
            self.inference_images += list(get_image_files(Path("../validation/covid/")/row["Name"]))
        # self.inference_images = list({x.parent for x in self.inference_images})
        dataloader = learner.dls.test_dl(self.inference_images)
        self.categories = learner.dls.vocab
        return dataloader


class Cov3dCombined(VisionApp):
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

        subdirs = ["train/covid", "train/non-covid", "validation/covid", "validation/non-covid"]
        for s in subdirs:
            subdir = directory/s
            if not subdir.exists():
                raise FileNotFoundError(f"Cannot find directory '{subdir}'.")
            subdir_paths = [path for path in subdir.iterdir() if path.name.startswith("ct_scan")]
            if len(subdir_paths) == 0:
                raise FileNotFoundError(f"Cannot file directories with prefix 'ct_scan' in {subdir}")
            for scan_dir in subdir_paths:
                paths += list(get_image_files(scan_dir))

        severity = dict()

        def read_severity_csv(csv:Path, dir:str):
            df = pd.read_csv(csv, delimiter=";")
            for _, row in df.iterrows():
                path = directory/f"{dir}/covid"/row['Name']
                if not path.exists():
                    raise FileNotFoundError(f"Cannot find directory {path}")
                severity[path] = row['Category']
        
        read_severity_csv(training_csv, dir="train")
        read_severity_csv(validation_csv, dir="validation")

        datablock = DataBlock(
            blocks=(CTSliceBlock, TransformBlock),
            splitter=FuncSplitter(is_validation),
            get_y=Cov3dCombinedGetter(severity),
        )

        dataloaders = DataLoaders.from_dblock(
            datablock, 
            source=paths,
            bs=batch_size,
        )
        
        dataloaders.c = 2
        return dataloaders    

    # def model(
    #     self,
    #     initial_features:int = fa.Param(default=64, tune=True, tune_min=16, tune_max=256, help="The number of features in the initial CNN layer."),
    # ) -> nn.Module:
    #     """
    #     Creates a deep learning model for the Cov3d to use.

    #     Returns:
    #        nn.Module: The created model.
    #     """ 
    #     return ResNet3d(
    #         initial_features=initial_features,
    #         num_classes=4,
    #     )

    def loss_func(self):
        return Cov3dLoss()

    def metrics(self):
        return [
            SeverityF1(),
            PresenceF1(),
            SeverityAccuracy(),
            PresenceAccuracy(),
        ]

    def monitor(self):
        return "presence_f1"

    def inference_dataloader(self, learner, **kwargs):
        self.inference_images = list(get_image_files(Path("../validation/covid/"))) + list(get_image_files(Path("../validation/non-covid/")))
        # self.inference_images = list({x.parent for x in self.inference_images})
        dataloader = learner.dls.test_dl(self.inference_images)
        self.categories = ["presence", "severity"]
        return dataloader

    def output_results(
        self,
        results,
        output_csv: Path = fa.Param(default=None, help="A path to output the results as a CSV."),
        **kwargs,
    ):
        results_df = pd.DataFrame(results[0].numpy(), columns=self.categories)
        results_df["image"] = self.inference_images
        results_df["scan"] = [image.parent.name for image in self.inference_images]
        # predictions = torch.argmax(results[0], dim=1)
        # results_df['prediction'] = [self.categories[p] for p in predictions]

        if not output_csv:
            raise Exception("No output file given.")

        console.print(f"Writing results for {len(results_df)} sequences to: {output_csv}")
        results_df.to_csv(output_csv)


class Covideo(fa.FastApp):
    """
    A deep learning model to detect the presence and severity of COVID19 in patients from CT-scans.
    """
    def dataloaders(
        self,
        directory:Path = fa.Param(help="The data directory."),
        batch_size:int = fa.Param(default=4, help="The batch size."),
        training_csv:Path = fa.Param(help="The path to the training CSV file with severity information."),
        validation_csv:Path = fa.Param(help="The path to the validation CSV file with severity information."),
        width:int = fa.Param(default=128, help="The width to convert the images to."),
        height:int = fa.Param(default=None, help="The height to convert the images to. If None, then it is the same as the width."),
        depth:int = fa.Param(default=128, help="The depth of the 3d volume to interpolate to."),
        normalize:bool = fa.Param(False, help="Whether or not to normalize the pixel data by the mean and std of the dataset."),
        severity_factor:float = 0.5,
        flip:bool = False,
        distortion:bool = True,
    ) -> DataLoaders:
        """
        Creates a FastAI DataLoaders object which Cov3d uses in training and prediction.

        Returns:
            DataLoaders: The DataLoaders object.
        """
        directory = Path(directory).resolve()
        paths = []
        self.severity_factor = severity_factor

        subdirs = ["train/covid", "train/non-covid", "validation/covid", "validation/non-covid"]

        severity = dict()

        def read_severity_csv(csv:Path, dir:str):
            df = pd.read_csv(csv, delimiter=";")
            for _, row in df.iterrows():
                path = directory/f"{dir}/covid"/row['Name']
                if not path.exists():
                    raise FileNotFoundError(f"Cannot find directory {path}")
                severity[path] = row['Category']
        
        read_severity_csv(training_csv, dir="train")
        read_severity_csv(validation_csv, dir="validation")
        
        for s in subdirs:
            subdir = directory/s
            if not subdir.exists():
                raise FileNotFoundError(f"Cannot find directory '{subdir}'.")
            subdir_paths = [path for path in subdir.iterdir() if path.name.startswith("ct_scan")]
            if len(subdir_paths) == 0:
                raise FileNotFoundError(f"Cannot file directories with prefix 'ct_scan' in {subdir}")
            
            if self.severity_factor >= 1.0:
                subdir_paths = [p for p in subdir_paths if p in severity]
            
            paths += subdir_paths

            if s == "train/covid":
                self.train_covid_count = len(subdir_paths)
            elif s == "train/non-covid":
                self.train_non_covid_count = len(subdir_paths)


        batch_tfms = []
        if normalize:
            batch_tfms.append(Normalize())

        self.width = width
        self.height = height or width
        self.depth = depth

        item_tfms = []
        if flip:
            item_tfms.append(Flip)

        datablock = DataBlock(
            blocks=(CTScanBlock(width=width, height=height, depth=depth, distortion=distortion), TransformBlock),
            splitter=FuncSplitter(is_validation),
            get_y=Cov3dCombinedGetter(severity),
            batch_tfms=batch_tfms,
            item_tfms=item_tfms,
        )

        dataloaders = DataLoaders.from_dblock(
            datablock, 
            source=paths,
            bs=batch_size,
        )
        
        dataloaders.c = 2
        return dataloaders    

    def model(
        self,
        model_name:str = "r3d_18",
        pretrained:bool = True,
        penultimate:int = 512,
        dropout:float = 0.5,
        max_pool:bool = True,
        severity_regression:bool = False,
        final_bias:bool = False,
        fine_tune:bool = False,
        flatten:bool = False,
        even_stride:bool = False,
        positional_encoding:bool=False,
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
        out_features = 1
        if self.severity_factor > 0.0:
            if severity_regression:
                out_features += 1
            else:
                out_features += 4

        self.fine_tune = fine_tune and pretrained

        if model_name in ("r3d_18", "mc3_18", "r2plus1d_18"):
            get_model = getattr(video, model_name)
            model = get_model(pretrained=pretrained)
            update_first_layer(model, in_channels, pretrained=pretrained)

            if even_stride:
                first_layer = next(model.stem.children())
                first_layer.stride = (2,2,2)

            if max_pool:
                model.avgpool = torch.nn.AdaptiveMaxPool3d( (1,1,1) )

            if self.fine_tune:
                for param in model.parameters():
                    param.requires_grad = False

            blocks = (self.depth//8)*(self.width//16)*(self.height//16)
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
                    nn.Conv3d( in_channels=512, out_channels=4, kernel_size=1, bias=True),
                    nn.ReLU(),
                    nn.Flatten(1),
                    nn.Linear(in_features=4*blocks, out_features=out_features, bias=final_bias)
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
                        nn.Linear(in_features=model.fc.in_features, out_features=penultimate, bias=True),
                        nn.Dropout(dropout),
                        nn.ReLU(),
                        nn.Linear(in_features=penultimate, out_features=out_features, bias=final_bias),
                    )
                else:
                    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=out_features, bias=final_bias)
        else:
            model = torch.hub.load("facebookresearch/pytorchvideo", model=model_name, pretrained=pretrained)
            update_first_layer(model, in_channels, pretrained=pretrained)
            if self.fine_tune:
                for param in model.parameters():
                    param.requires_grad = False
            model.blocks[-1] = create_res_basic_head(in_features=2048, out_features=out_features, pool_kernel_size=(4,4,4))

        if self.positional_encoding:
            model = nn.Sequential(
                PositionalEncoding3D(),
                model,
            )

        return model

    def loss_func(
        self,
        presence_smoothing:float = 0.1,
        severity_smoothing:float = 0.1,
        neighbour_smoothing:bool = False,
        mse:bool = False,
    ):
        pos_weight = self.train_non_covid_count/self.train_covid_count
        return Cov3dLoss(
            pos_weight=torch.as_tensor([pos_weight]).cuda(), # hack - this should be to the device of the other tensors
            severity_factor=self.severity_factor,
            severity_regression=self.severity_regression,
            presence_smoothing=presence_smoothing,
            severity_smoothing=severity_smoothing,
            neighbour_smoothing=neighbour_smoothing,
            mse=mse,
        ) 

    def metrics(self):
        metrics = [
            PresenceF1(),
            PresenceAccuracy(),
        ]
        if self.severity_factor > 0.0:
            metrics += [
                SeverityF1(),
                SeverityAccuracy(),
            ]

        return metrics

    def monitor(self):
        return "presence_f1"

    def fit(
        self,
        learner,
        callbacks,
        epochs: int = fa.Param(default=20, help="The number of epochs."),
        freeze_epochs: int = fa.Param(
            default=3,
            help="The number of epochs to train when the learner is frozen and the last layer is trained by itself. Only if `fine_tune` is set on the app.",
        ),
        learning_rate: float = fa.Param(
            default=1e-4, help="The base learning rate (when fine tuning) or the max learning rate otherwise."
        ),
        **kwargs,
    ):
        if self.fine_tune:
            learner.fine_tune(
                0, freeze_epochs=freeze_epochs, base_lr=learning_rate, cbs=callbacks, **kwargs
            )  # hack
            for param in learner.model.parameters():
                param.requires_grad = True
            return learner.fine_tune(
                epochs, freeze_epochs=0, base_lr=learning_rate, cbs=callbacks, **kwargs
            )  # hack

        return learner.fit_one_cycle(epochs, lr_max=learning_rate, cbs=callbacks, **kwargs)

    def learner_kwargs(
        self,
        output_dir: Path = fa.Param("./outputs", help="The location of the output directory."),
        weight_decay: float = fa.Param(None, help="The amount of weight decay. If None then it uses the default amount of weight decay in fastai."),
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

    # def inference_dataloader(self, learner, **kwargs):
    #     self.inference_images = get_image_files(Path("../validation/non-covid/"))
    #     # self.inference_images = list({x.parent for x in self.inference_images})
    #     dataloader = learner.dls.test_dl(self.inference_images)
    #     self.categories = ["presence", "severity"]
    #     return dataloader

    # def output_results(
    #     self,
    #     results,
    #     output_csv: Path = fa.Param(default=None, help="A path to output the results as a CSV."),
    #     **kwargs,
    # ):
    #     results_df = pd.DataFrame(results[0].numpy(), columns=self.categories)
    #     results_df["image"] = self.inference_images
    #     results_df["scan"] = [image.parent.name for image in self.inference_images]
    #     predictions = torch.argmax(results[0], dim=1)
    #     results_df['prediction'] = [self.categories[p] for p in predictions]

    #     if not output_csv:
    #         raise Exception("No output file given.")

    #     console.print(f"Writing results for {len(results_df)} sequences to: {output_csv}")
    #     results_df.to_csv(output_csv)


class CovideoSeverity(Covideo):
    def monitor(self):
        return "severity_f1"


