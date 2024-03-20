from sklearn.metrics import f1_score
from fastai.metrics import AccumMetric
import torch





def get_presence_binary(predictions, target, *args):
    predictions_binary = torch.argmax(predictions, dim=1)
    target_binary = target > 0
    return predictions_binary, target_binary


def presence_accuracy(predictions, target, *args):
    """
    Gives the accuracy of detecting the presence of COVID.
    """
    predictions_binary, target_binary = get_presence_binary(predictions, target)
    return (predictions_binary == target_binary).float().mean()


def presence_f1(predictions, target, *args, category=None):
    """
    Gives the f1 score of detecting the presence of COVID.
    """
    average="macro" if category is None else None

    predictions_binary, target_binary = get_presence_binary(predictions, target)
    score = f1_score(target_binary.cpu(), predictions_binary.cpu(), average=average)

    return score if category is None else score[category]


def presence_f1_noncovid(predictions, target, *args):
    """
    Gives the f1 score of detecting the presence of COVID.
    """
    return presence_f1(predictions, target, category=0)


def presence_f1_covid(predictions, target, *args):
    """
    Gives the f1 score of detecting the presence of COVID.
    """
    return presence_f1(predictions, target, category=1)


class AccumMetricExtraArgs(AccumMetric):
    # def __call__(self, preds, targs, *args):
    #     breakpoint()
    #     return super().__call__(self, preds, targs)

    def accum_values(self, preds, targs, learn=None):
        if isinstance(targs, tuple):
            targs = targs[0]
        super().accum_values(preds, targs, learn=learn)

def PresenceF1():
    return AccumMetricExtraArgs(presence_f1, flatten=False)


def PresenceAccuracy():
    return AccumMetricExtraArgs(presence_accuracy, flatten=False)


def NonCovidF1():
    return AccumMetricExtraArgs(presence_f1_noncovid, flatten=False)


def CovidF1():
    return AccumMetricExtraArgs(presence_f1_covid, flatten=False)


