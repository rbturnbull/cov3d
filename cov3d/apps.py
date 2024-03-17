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
import pickle
from torchapp.util import call_func
from fastai.learner import load_learner
import torchapp as ta
from fastcore.foundation import L
from rich.console import Console
from speedict import Rdict, Options, DBCompressionType, AccessType


def vector_db_options():
    options = Options(raw_mode=True)
    options.set_compression_type(DBCompressionType.none())
    # options.set_cache_index_and_filter_blocks(True)
    options.set_optimize_filters_for_hits(True)
    options.optimize_for_point_lookup(1024)
    options.set_max_open_files(500)
    return options

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
    FlipPath,
    Lung1,
    Lung2,
    AdjustBrightness,
    AdjustContrast,
    VectorBlock,
    Clip,
)
from .models import ResNet3d, update_first_layer, PositionalEncoding3D, adapt_stoic_model
from .loss import FocalLoss, WeightedCrossEntropyLoss, WeightedFocusLoss
from .metrics import (
    PresenceF1,
    PresenceAccuracy,
    CovidF1,
    NonCovidF1,
)


def get_y(scan_path: Path):
    parent_name = scan_path.parent.name
    if parent_name == "covid":
        return True
    if parent_name == "non_covid":
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
        self.depth = 64 # will be overridden by dataloader
        self.width = 128 # will be overridden by dataloader
        self.height = 128 # will be overridden by dataloader

    def dataloaders(
        self,
        directory: Path = ta.Param(help="The data directory."),
        batch_size: int = ta.Param(default=4, help="The batch size."),
        csv: Path = ta.Param(
            None,
            help="The path to a file which contains the cross-validation splits."
        ),
        split: int = ta.Param(
            0,
            help="The cross-validation split to use. The default (i.e. 0) is the original validation set."
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
        # distortion: bool = True,
        autocrop:bool = True,
        max_scans:int = 0,
        individual_lung:bool = False,
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
        # if splits_csv is None:
        #     splits_csv = Path("cross-validation-2024.csv")
        #     if not splits_csv.exists():
        #         splits_csv = None

        assert csv is not None

        splits_df = pd.read_csv(csv)
        paths = [directory/path for path in splits_df['path']]
        validation_dict = {path:split == s for path, s in zip(paths, splits_df['split'])}
        has_covid_dict = {path:int(has_covid) for path, has_covid in zip(paths, splits_df['has_covid'])}
        if 'weight' in splits_df:
            weights_dict = {path:float(weight) for path, weight in zip(paths, splits_df['weight'])}
        else:
            weights_dict = {path:1.0 for path in paths}

        if max_scans:
            random.seed(42)
            random.shuffle(paths)
            paths = paths[:max_scans]

        if individual_lung:
            paths = [Lung1(path) for path in paths] + [Lung2(path) for path in paths]
            for path in paths:
                regular_path = Path(path)
                validation_dict[path] = validation_dict[regular_path]
                has_covid_dict[path] = has_covid_dict[regular_path]
                weights_dict[path] = weights_dict[regular_path]
        elif flip:
            # do each scan twice per epoch, one which is flipped
            flipped_paths = []
            for path in paths:
                flipped = FlipPath(path)
                validation_dict[flipped] = validation_dict[path]
                has_covid_dict[flipped] = has_covid_dict[path]
                weights_dict[flipped] = weights_dict[path]
                flipped_paths.append(flipped)

            paths += flipped_paths
        
        splitter = DictionarySplitter(validation_dict)

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

        datablock = DataBlock(
            blocks=(
                CTScanBlock(
                    width=width, 
                    height=height, 
                    depth=depth,
                    autocrop=autocrop,
                ),
                TransformBlock,
                TransformBlock,
            ),
            splitter=splitter,
            getters=[
                None,
                DictionaryGetter(has_covid_dict),
                DictionaryGetter(weights_dict),
            ],
            batch_tfms=batch_tfms,
            item_tfms=item_tfms,
            n_inp=1,
        )

        dataloaders = DataLoaders.from_dblock(
            datablock,
            source=paths,
            bs=batch_size,
        )

        dataloaders.c = 2 # self.categories_count # is this used??
        
        return dataloaders

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

        out_features = 2
        self.fine_tune = fine_tune and pretrained

        if Path(model_name).exists():
            model_path = Path(model_name)
            import dill
            pretrained_learner = load_learner(model_path, cpu=False, pickle_module=dill)
            model = pretrained_learner.model
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
        return [ExportLearnerCallback(monitor="presence_f1")]

    def loss_func(self, gamma:float=2.0):
        return WeightedCrossEntropyLoss()
        # return FocalLoss(gamma=gamma)

    def metrics(self):
        metrics = [
            PresenceF1(),
            PresenceAccuracy(),
            NonCovidF1(),
            CovidF1(),
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

    # def learner_kwargs(
    #     self,
    #     output_dir: Path = ta.Param(
    #         "./outputs", help="The location of the output directory."
    #     ),
    #     weight_decay: float = ta.Param(
    #         None,
    #         help="The amount of weight decay. If None then it uses the default amount of weight decay in fastai.",
    #     ),
    #     **kwargs,
    # ):
    #     output_dir = Path(output_dir)
    #     output_dir.mkdir(exist_ok=True, parents=True)

    #     return dict(
    #         loss_func=call_func(self.loss_func, **kwargs),
    #         metrics=call_func(self.metrics, **kwargs),
    #         path=output_dir,
    #         wd=weight_decay,
    #     )

    # def __call__(
    #     self,
    #     gpu: bool = ta.Param(
    #         True, help="Whether or not to use a GPU for processing if available."
    #     ),
    #     mc_samples: int = ta.Param(
    #         0, help="The number of Monte Carlo samples of the results to get."
    #     ),
    #     mc_dropout: bool = ta.Param(
    #         False,
    #         help="Whether or not to use Monte Carlo dropout if doing MC sampling.",
    #     ),
    #     **kwargs,
    # ):
    #     # Open the exported learner from a pickle file
    #     path = call_func(self.pretrained_local_path, **kwargs)
    #     try:
    #         learner = self.learner_obj = load_learner(path, cpu=not gpu)
    #     except Exception:
    #         import dill
    #         learner = self.learner_obj = load_learner(path, cpu=not gpu, pickle_module=dill)

    #     callbacks_to_remove = [
    #         "ExportLearnerCallback",
    #         "TorchAppWandbCallback",
    #         "SaveModelCallback",
    #     ]
    #     learner.cbs = L([callback for callback in learner.cbs if type(callback).__name__ not in callbacks_to_remove])

    #     # Create a dataloader for inference
    #     dataloader = call_func(self.inference_dataloader, learner, **kwargs)
    #     results_samples = []

    #     if mc_samples:
    #         callbacks = [MCDropoutCallback()] if mc_dropout else []

    #         for index in range(mc_samples):
    #             print(f"Monte Carlo sample run {index}")
    #             results, _ = learner.get_preds(
    #                 dl=dataloader,
    #                 reorder=False,
    #                 with_decoded=False,
    #                 act=self.activation(),
    #                 cbs=callbacks,
    #             )
    #             results_samples += [results]

    #     else:
    #         dataloader.after_item = Pipeline()
    #         results, _ = learner.get_preds(
    #             dl=dataloader, reorder=False, with_decoded=False, act=self.activation()
    #         )
    #         results_samples.append(results)

    #         dataloader.after_item = Pipeline(Flip(always=True))
    #         results, _ = learner.get_preds(
    #             dl=dataloader, reorder=False, with_decoded=False, act=self.activation()
    #         )
    #         results_samples.append(results)

    #     results = torch.stack(results_samples, dim=1)

    #     # Output results
    #     return call_func(self.output_results, results, **kwargs)

    def inference_dataloader(
        self,
        learner,
        scan: List[Path] = ta.Param(None, help="A directory of a CT scan."),
        csv:Path = ta.Param(None, help="A CSV with a list of scans in column 'path'."),
        scan_dir: List[Path] = ta.Param(
            None,
            help="A directory with CT scans in subdirectories. Subdirectories must start with 'ct_scan' or 'test_ct_scan'.",
        ),
        directory:Path=None,
        output_vectors:Path=ta.Param(None, help="The path to store vectors."),
        flip:bool=False,
        **kwargs,
    ):
        self.scans = []
        if isinstance(scan, (str, Path)):
            scan = [Path(scan)]
        if isinstance(scan_dir, (str, Path)):
            scan_dir = [Path(scan_dir)]

        for s in scan:
            self.scans.append(Path(s))

        if csv:
            splits_df = pd.read_csv(csv)
            self.scans = [Path(path) for path in splits_df['path']]
        
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
    
        self.output_vectors = output_vectors
        if output_vectors:
            self.output_vectors = Path(output_vectors)
            if hasattr(learner.model, "fc") and isinstance(learner.model.fc, nn.Sequential):
                learner.model.fc = list(learner.model.fc.children())[0]

        if flip:
            self.scans = [FlipPath(path) for path in self.scans]

        dataloader = learner.dls.test_dl(self.scans)

        return dataloader

    def output_results(
        self,
        results,
        output_csv: Path = ta.Param(
            default=None, help="A path to output the results as a CSV."
        ),
        covid_txt: Path = ta.Param(
            default=None,
            help="A path to output the names of the predicted COVID positive scans.",
        ),
        noncovid_txt: Path = ta.Param(
            default=None,
            help="A path to output the names of the predicted COVID negative scans.",
        ),
        **kwargs,
    ):
        results = results[0] # only take predictions
        results_df_data = []
        assert len(self.scans) == len(results)

        if self.output_vectors:
            self.output_vectors.parent.mkdir(exist_ok=True, parents=True)
            print(f'writing vector outputs to {self.output_vectors}')
            db = Rdict(path=str(self.output_vectors), options=vector_db_options(), access_type=AccessType.read_write())
            assert len(self.scans) == len(results)
            for path, result in zip(self.scans, results):
                assert len(result) == 512
                db[str(path).encode('utf-8')] = pickle.dumps(result)

            return db

        columns = [
            "name",
            "COVID19 positive",
            "probability",
            "path",
        ]
        
        for path, result in zip(self.scans, results):
            sample_probabilties = torch.softmax(result, dim=0)
            
            positive = sample_probabilties[0] < 0.5
            probability_positive = 1.0-sample_probabilties[0]

            results_df_data.append(
                [
                    path.name,
                    positive.item(),
                    probability_positive.item(),
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

        print(results_df)
        print(f"COVID19 Positive: {results_df['COVID19 positive'].sum()}")
        print(
            f"COVID19 Negative: {len(results_df) - results_df['COVID19 positive'].sum()}"
        )

        return results_df


class Cov3dEnsembler(Cov3d):    
    def dataloaders(
        self,
        directory: Path = ta.Param(help="The data directory."),
        vectors:List[Path] = ta.Param(help="The vector files from earlier models."),
        batch_size: int = ta.Param(default=4, help="The batch size."),
        csv: Path = ta.Param(
            None,
            help="The path to a file which contains the cross-validation splits."
        ),
        split: int = ta.Param(
            0,
            help="The cross-validation split to use. The default (i.e. 0) is the original validation set."
        ),
        max_scans:int = 0,
    ) -> DataLoaders:
        """
        Creates a FastAI DataLoaders object which Cov3d uses in training and prediction.

        Returns:
            DataLoaders: The DataLoaders object.
        """
        directory = Path(directory).resolve()
        paths = []

        assert csv is not None

        splits_df = pd.read_csv(csv)
        paths = [directory/path for path in splits_df['path']]
        validation_dict = {path:split == s for path, s in zip(paths, splits_df['split'])}
        has_covid_dict = {path:int(has_covid) for path, has_covid in zip(paths, splits_df['has_covid'])}
        if 'weight' in splits_df:
            weights_dict = {path:float(weight) for path, weight in zip(paths, splits_df['weight'])}
        else:
            weights_dict = {path:1.0 for path in paths}

        if max_scans:
            random.seed(42)
            random.shuffle(paths)
            paths = paths[:max_scans]

        splitter = DictionarySplitter(validation_dict)

        self.vector_dbs = [
            Rdict(path=str(path), options=vector_db_options(), access_type=AccessType.read_only())
            for path in vectors
        ]

        batch_tfms = []

        datablock = DataBlock(
            blocks=(
                VectorBlock(
                    vector_dbs=self.vector_dbs, 
                ),
                TransformBlock,
                TransformBlock,
            ),
            splitter=splitter,
            getters=[
                None,
                DictionaryGetter(has_covid_dict),
                DictionaryGetter(weights_dict),
            ],
            batch_tfms=batch_tfms,
            n_inp=1,
        )

        dataloaders = DataLoaders.from_dblock(
            datablock,
            source=paths,
            bs=batch_size,
        )

        dataloaders.c = 2
        
        return dataloaders

    def model(self, penultimate_features:int=512, dropout:float=0.5):

        input_features = 512 * len(self.vector_dbs)
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features=input_features, out_features=penultimate_features),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(in_features=penultimate_features, out_features=2),
        )

    def loss_func(self, gamma:float=0.0):
        return WeightedFocusLoss(gamma=gamma)
        # return FocalLoss(gamma=gamma)
