import re
from pathlib import Path
from torch import nn
from typing import List
from fastai.data.core import DataLoaders
from fastai.data.transforms import GrandparentSplitter, get_image_files, FuncSplitter
from fastai.data.core import DataLoaders
from fastai.data.block import DataBlock, CategoryBlock, TransformBlock
from fastai.metrics import accuracy, Precision, Recall, F1Score
import torch
import pandas as pd
from fastapp.util import call_func
from fastai.learner import load_learner
from fastapp.vision import VisionApp
import fastapp as fa
from rich.console import Console
console = Console()
from fastapp.metrics import logit_f1, logit_accuracy
from pytorchvideo.models.head import create_res_basic_head
from fastcore.transform import Pipeline
from fastai.callback.preds import MCDropoutCallback

from torchvision.models import video

from .transforms import CTScanBlock, BoolBlock, CTSliceBlock, ReadCTScanTricubic, Normalize, Flip
from .models import ResNet3d, update_first_layer, PositionalEncoding3D
from .loss import Cov3dLoss
from .metrics import SeverityF1, PresenceF1, SeverityAccuracy, PresenceAccuracy, severity_probability_to_category



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
            
            # if self.severity_factor >= 1.0:
            #     subdir_paths = [p for p in subdir_paths if p in severity]
            
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
        cov3d_trained:Path=None,
        severity_everything:bool=False,
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
        out_features = 1
        if self.severity_factor > 0.0:
            if severity_everything:
                out_features += 5
            elif severity_regression:
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

        if cov3d_trained is not None:
            state_dict = torch.load(cov3d_trained)
            model.load_state_dict(state_dict)
            return model

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
            severity_everything=self.severity_everything,
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

    def __call__(
        self, 
        gpu: bool = fa.Param(True, help="Whether or not to use a GPU for processing if available."), 
        mc_samples:int = fa.Param(0, help="The number of Monte Carlo samples of the results to get."),
        mc_dropout:bool = fa.Param(False, help="Whether or not to use Monte Carlo dropout if doing MC sampling."),
        **kwargs
    ):
        # Open the exported learner from a pickle file
        path = call_func(self.pretrained_local_path, **kwargs)
        learner = load_learner(path, cpu=not gpu)

        # Create a dataloader for inference
        dataloader = call_func(self.inference_dataloader, learner, **kwargs)
        results_samples = []

        if mc_samples:
            callbacks = [MCDropoutCallback()] if mc_dropout else []

            for index in range(mc_samples):
                print(f"Monte Carlo sample run {index}")
                results, _ = learner.get_preds(dl=dataloader, reorder=False, with_decoded=False, act=self.activation(), cbs=callbacks)
                results_samples += [results]

        else:
            dataloader.after_item = Pipeline()
            results, _ = learner.get_preds(dl=dataloader, reorder=False, with_decoded=False, act=self.activation())
            results_samples.append(results)

            dataloader.after_item = Pipeline(Flip(always=True))
            results, _ = learner.get_preds(dl=dataloader, reorder=False, with_decoded=False, act=self.activation())
            results_samples.append(results)

        results = torch.stack(results_samples, dim=1)

        # Output results
        return call_func(self.output_results, results, **kwargs)

    def inference_dataloader(
        self, 
        learner, 
        scan:List[Path] = fa.Param(None, help="A directory of a CT scan."),
        scan_dir:List[Path] = fa.Param(None, help="A directory with CT scans in subdirectories. Subdirectories must start with ct_scan or test_ct_scan."), 
        **kwargs
    ):
        self.scans = []
        if isinstance(scan, (str, Path)):
            scan = [Path(scan)]
        if isinstance(scan_dir, (str, Path)):
            scan_dir = [Path(scan_dir)]

        for s in scan:
            self.scans.append(Path(s))
            
        for sdir in scan_dir:
            sdir = Path(sdir)
            self.scans += [path for path in sdir.iterdir() if path.name.startswith("ct_scan") or path.name.startswith("test_ct_scan")]

        dataloader = learner.dls.test_dl(self.scans)

        return dataloader

    def output_results(
        self,
        results,
        output_csv: Path = fa.Param(default=None, help="A path to output the results as a CSV."),
        output_mc: Path = fa.Param(default=None, help="A path to output all MC inference runs as a PyTorch tensor."),
        covid_txt: Path = fa.Param(default=None, help="A path to output the names of the predicted COVID positive scans."),
        noncovid_txt: Path = fa.Param(default=None, help="A path to output the names of the predicted COVID negative scans."),
        mild_txt: Path = fa.Param(default=None, help="A path to output the names of the predicted mild COVID scans."),
        moderate_txt: Path = fa.Param(default=None, help="A path to output the names of the predicted moderate COVID scans."),
        severe_txt: Path = fa.Param(default=None, help="A path to output the names of the predicted severe COVID scans."),
        critical_txt: Path = fa.Param(default=None, help="A path to output the names of the predicted critical COVID scans."),
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
            result_average = result.mean(dim=0)
            positive = result_average[0] >= 0.0
            probability = torch.sigmoid( result_average[0] )
            mc_samples_total = result.shape[0]
            mc_samples_positive = (result[:,0] >= 0.0).sum()/mc_samples_total

            severity_categories = ["mild", "moderate", "severe", "critical", "unknown"]
            if result.shape[-1] >= 5:
                softmax = torch.softmax(result[:,1:], dim=-1)
                average = softmax.mean(dim=0)
                severity_probabilities = average[0:4]/average[0:4].sum(dim=-1, keepdim=True)
                severity_id = torch.argmax(severity_probabilities, dim=-1).item()
                sample_severity_ids = torch.argmax(result[:,1:5], dim=1)
            elif result.shape[-1] > 1:
                prediction_probabilities = torch.sigmoid(result_average[1])
                severity_id = severity_probability_to_category(prediction_probabilities) - 1
                sample_prediction_probabilities = torch.sigmoid(result[:,1])
                sample_severity_ids = severity_probability_to_category(sample_prediction_probabilities) - 1
            else:
                severity_id = 4
                sample_severity_ids = torch.as_tensor([severity_id]*len(result))
                severity_probabilities = torch.zeros(4)

            severity = severity_categories[severity_id]
            mild_samples = (sample_severity_ids == 0).sum()/mc_samples_total
            moderate_samples = (sample_severity_ids == 1).sum()/mc_samples_total
            severe_samples = (sample_severity_ids == 2).sum()/mc_samples_total
            critical_samples = (sample_severity_ids == 3).sum()/mc_samples_total

            results_df_data.append([
                path.name,
                positive.item(),
                probability.item(),
                mc_samples_positive.item(),
                severity,
                mild_samples.item(),
                moderate_samples.item(),
                severe_samples.item(),
                critical_samples.item(),
                severity_probabilities[0].item(),
                severity_probabilities[1].item(),
                severity_probabilities[2].item(),
                severity_probabilities[3].item(),
                mc_samples_total,
                path,
            ])

        results_df = pd.DataFrame(results_df_data, columns=columns)

        if output_csv:
            console.print(f"Writing results for {len(results_df)} sequences to: {output_csv}")
            results_df.to_csv(output_csv, index=False)

        def get_digits(string):
            m = re.search(r"\d+", string)
            if m:
                return int(m.group(0))
            return -1

        def write_scans_txt(filename, mask):
            if filename:
                scans = results_df[ mask ][ "name" ].tolist()
                scans = sorted(scans, key=get_digits)
                print(f"writing to {filename}")
                with open(filename, 'w') as f:
                    f.write("\n".join(scans) + "\n")

        write_scans_txt(covid_txt, results_df['COVID19 positive'] == True)
        write_scans_txt(noncovid_txt, results_df['COVID19 positive'] == False)
        write_scans_txt(mild_txt, results_df['severity'] == "mild")
        write_scans_txt(moderate_txt, results_df['severity'] == "moderate")
        write_scans_txt(severe_txt, results_df['severity'] == "severe")
        write_scans_txt(critical_txt, results_df['severity'] == "critical")

        print(results_df)
        print(f"COVID19 Positive: {results_df['COVID19 positive'].sum()}")
        print(f"COVID19 Negative: {len(results_df) - results_df['COVID19 positive'].sum()}")

        return results_df


class Cov3dSeverity(Cov3d):
    def monitor(self):
        return "severity_f1"


