from sklearn.metrics import f1_score
from fastai.metrics import AccumMetric
import torch


def SeverityF1():
    return AccumMetric(severity_f1, flatten=False)


def PresenceF1():
    return AccumMetric(presence_f1, flatten=False)


def SeverityAccuracy():
    return AccumMetric(severity_accuracy, flatten=False)


def PresenceAccuracy():
    return AccumMetric(presence_accuracy, flatten=False)


def NonCovidF1():
    return AccumMetric(presence_f1_noncovid, flatten=False)


def CovidF1():
    return AccumMetric(presence_f1_covid, flatten=False)


def MildF1():
    return AccumMetric(mild_f1, flatten=False)


def ModerateF1():
    return AccumMetric(moderate_f1, flatten=False)


def SevereF1():
    return AccumMetric(severe_f1, flatten=False)


def CriticalF1():
    return AccumMetric(critical_f1, flatten=False)


def get_presence_binary(predictions, target):
    predictions_binary = predictions[:,1:].sum(dim=1) > predictions[:,0]
    target_binary = target > 0
    return predictions_binary, target_binary


def presence_accuracy(predictions, target):
    """
    Gives the accuracy of detecting the presence of COVID.
    """
    predictions_binary, target_binary = get_presence_binary(predictions, target)
    return (predictions_binary == target_binary).float().mean()


def presence_f1(predictions, target, category=None):
    """
    Gives the f1 score of detecting the presence of COVID.
    """
    average="macro" if category is None else None

    predictions_binary, target_binary = get_presence_binary(predictions, target)
    score = f1_score(target_binary.cpu(), predictions_binary.cpu(), average=average)

    return score if category is None else score[category]


def presence_f1_noncovid(predictions, target):
    """
    Gives the f1 score of detecting the presence of COVID.
    """
    return presence_f1(predictions, target, category=0)


def presence_f1_covid(predictions, target):
    """
    Gives the f1 score of detecting the presence of COVID.
    """
    return presence_f1(predictions, target, category=1)


def get_severtity_categories(predictions, target):
    mask = (1 <= target) & (target <= 4)
    prediction = predictions[mask]
    prediction_categories = torch.argmax(prediction[:,1:], dim=-1) + 1
    target_categories = target[mask]

    return prediction_categories, target_categories


def severity_accuracy(predictions, target):
    """
    Gives the accuracy of detecting the severity of COVID.
    """
    prediction_categories, target_categories = get_severtity_categories(
        predictions, target
    )
    return (prediction_categories == target_categories).float().mean()


def severity_f1(predictions, target, category=None):
    """
    Gives the f1 score of detecting the severity of COVID.
    """
    average="macro" if category is None else None
    prediction_categories, target_categories = get_severtity_categories(
        predictions, target
    )
    score = f1_score(
        target_categories.cpu(), prediction_categories.cpu(), average=average
    )


    return score if category is None else score[category-1]


def mild_f1(predictions, target):
    """
    Gives the f1 score of detecting the severity of COVID with the case is mild.
    """
    return severity_f1(predictions, target, category=1)


def moderate_f1(predictions, target):
    """
    Gives the f1 score of detecting the severity of COVID with the case is moderate.
    """
    return severity_f1(predictions, target, category=2)


def severe_f1(predictions, target):
    """
    Gives the f1 score of detecting the severity of COVID with the case is severe.
    """
    return severity_f1(predictions, target, category=3)


def critical_f1(predictions, target):
    """
    Gives the f1 score of detecting the severity of COVID with the case is critical.
    """
    return severity_f1(predictions, target, category=4)
