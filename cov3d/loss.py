from torch import nn
from torch import Tensor
import torch.nn.functional as F

class Cov3dLoss(nn.Module):
    def __init__(self, severity_factor:float=1.0, smoothing:float=0.1, **kwargs):
        super().__init__(**kwargs)
        self.severity_factor = severity_factor
        self.smoothing = smoothing

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        presence_labels = target[:,:1]
        smoothed_labels = presence_labels * (1.0-self.smoothing) + self.smoothing
        presence_loss = F.binary_cross_entropy_with_logits(input[:,:1], smoothed_labels)
        severity_present = target[:,1] > 0
        severity_probability_labels = target[:,1:] * 0.25 - 0.125

        severity_loss = F.binary_cross_entropy_with_logits(
            input[severity_present,1:], 
            severity_probability_labels[severity_present]
        ) if severity_present.sum() > 0 else 0.0

        return presence_loss + self.severity_factor*severity_loss