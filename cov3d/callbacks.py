from fastai.callback.tracker import TrackerCallback
import dill

class ExportLearnerCallback(TrackerCallback):
    "A `TrackerCallback` that saves the model's best during training and loads it at the end."
    order = TrackerCallback.order+1
    def __init__(self, 
        monitor='valid_loss', # value (usually loss or metric) being monitored.
        comp=None, # numpy comparison operator; np.less if monitor is loss, np.greater if monitor is metric.
        min_delta=0., # minimum delta between the last monitor value and the best monitor value.
        every_epoch=False, # if true, save model after every epoch; else save only when model is better than existing best.
        at_end=False, # if true, save model when training ends; else load best model if there is only one saved model.
        reset_on_fit=True, # before model fitting, reset value being monitored to -infinity (if monitor is metric) or +infinity (if monitor is loss).
        filename:str = "",
    ):
        super().__init__(monitor=monitor, comp=comp, min_delta=min_delta, reset_on_fit=reset_on_fit)
        assert not (every_epoch and at_end), "every_epoch and at_end cannot both be set to True"
        # keep track of file path for loggers
        self.last_saved_path = None
        self.every_epoch = every_epoch
        self.at_end = at_end
        self.filename = filename or f"{monitor}.best"

    def _save(self, filename="", suffix=".pkl"): 
        filename = filename or self.filename
        self.learn.export(fname=self.filename+suffix, pickle_module=dill)

    def after_epoch(self):
        "Compare the value monitored to its best score and save if best."
        if self.every_epoch:
            if (self.epoch%self.every_epoch) == 0: 
                self._save(f"{self.filename}-{self.epoch}")
        else: #every improvement
            super().after_epoch()
            if self.new_best:
                print(f'Better model found at epoch {self.epoch} with {self.monitor} value: {self.best}.')
                self._save()

    def after_fit(self, **kwargs):
        "Load the best model."
        if self.at_end: 
            self._save()
        elif not self.every_epoch: 
            pass
            # self.learn.load(f'{self.fname}', with_opt=self.with_opt) # reload
