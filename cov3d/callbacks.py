from fastai.callback.tracker import SaveModelCallback
import dill

class ExportLearnerCallback(SaveModelCallback):
    "A `TrackerCallback` that saves the model's best during training and loads it at the end."
    def __init__(self, 
        filename:str = "",
        monitor:str = "valid_loss",
        **kwargs
    ):
        self.filename = filename or f"{monitor}.best"
        super().__init__(monitor=monitor, fname=self.filename, **kwargs)

    def after_fit(self, **kwargs):
        "Load the best model."
        super().after_fit(**kwargs)
        suffix=".pkl"
        self.learn.export(fname=self.filename+suffix, pickle_module=dill)
        