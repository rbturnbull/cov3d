from torch import nn
from torch import Tensor
import torch.nn.functional as F

class Cov3dLoss(nn.Module):
    def __init__(self, severity_factor:float=0.5, smoothing:float=0.1, severity_regression:bool=False, severity_smoothing:float=0.1, pos_weight=None, **kwargs):
        super().__init__(**kwargs)
        self.severity_factor = severity_factor
        self.smoothing = smoothing
        self.pos_weight = pos_weight
        self.severity_regression = severity_regression
        self.severity_smoothing = severity_smoothing

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        presence_labels = target[:,:1]
        smoothed_labels = presence_labels * (1.0-self.smoothing) + self.smoothing
        presence_loss = F.binary_cross_entropy_with_logits(input[:,:1], smoothed_labels, pos_weight=self.pos_weight)
        if self.severity_factor <= 0.0:
            return presence_loss

        if self.severity_regression:
            severity_present = target[:,1] > 0
            severity_probability_labels = target[:,1:] * 0.25 - 0.125

            severity_loss = F.binary_cross_entropy_with_logits(
                input[severity_present,1:], 
                severity_probability_labels[severity_present]
            ) if severity_present.sum() > 0 else 0.0
        else:
            severity_loss = F.cross_entropy(
                input[:,1:], 
                target[:,1]-1,  # The minus one is because the labels are 0–4 and we want to ignore the zero class
                ignore_index=-1, 
                label_smoothing=self.severity_smoothing,
            )

        # return severity_loss
        return presence_loss*(1.0-self.severity_factor) + self.severity_factor*severity_loss
