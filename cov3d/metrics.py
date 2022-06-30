from sklearn.metrics import f1_score
from fastai.metrics import AccumMetric
import torch


def severity_probability_to_category(tensor):
    return torch.clamp((4.0*tensor + 1.0).int(),max=4, min=1)


def get_severtity_categories(predictions, target):
    severity_present = target[:,1] > 0
    target_categories = target[severity_present,1]

    if predictions.shape[-1] > 2: # If there are more elements in the output, then assume cross-entroy loss was used and get the argmax
        prediction_categories = torch.argmax(predictions[severity_present,1:], dim=1)
        target_categories = target_categories - 1
    else:
        prediction_probabilities = torch.sigmoid(predictions[severity_present,1])
        prediction_categories = severity_probability_to_category(prediction_probabilities)
    
    return prediction_categories, target_categories


def get_presence_binary(predictions, target):
    predictions_binary = predictions[:,0] > 0.0
    target_binary = target[:,0] > 0.5
    return predictions_binary, target_binary


def presence_accuracy(predictions, target):
    """
    Gives the accuracy of detecting the presence of COVID.
    """
    predictions_binary, target_binary = get_presence_binary(predictions, target)
    return (predictions_binary == target_binary).float().mean()


def severity_accuracy(predictions, target):
    """
    Gives the accuracy of detecting the severity of COVID.
    """
    prediction_categories, target_categories = get_severtity_categories(predictions, target)
    return (prediction_categories == target_categories).float().mean()


def presence_f1(predictions, target):
    """
    Gives the f1 score of detecting the presence of COVID.
    """
    predictions_binary, target_binary = get_presence_binary(predictions, target)
    return f1_score(target_binary.cpu(), predictions_binary.cpu(), average="macro")


def severity_f1(predictions, target):
    """
    Gives the f1 score of detecting the severity of COVID.
    """
    prediction_categories, target_categories = get_severtity_categories(predictions, target)
    return f1_score(target_categories.cpu(), prediction_categories.cpu(), average="macro")


def SeverityF1():
    return AccumMetric(severity_f1, flatten=False)


def PresenceF1():
    return AccumMetric(presence_f1, flatten=False)


def SeverityAccuracy():
    return AccumMetric(severity_accuracy, flatten=False)


def PresenceAccuracy():
    return AccumMetric(presence_accuracy, flatten=False)