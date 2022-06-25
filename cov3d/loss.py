from torch import nn
from torch import Tensor
import torch
import torch.nn.functional as F

class Cov3dLoss(nn.Module):
    def __init__(
        self, 
        severity_factor:float=0.5, 
        smoothing:float=0.1, 
        severity_regression:bool=False, 
        severity_smoothing:float=0.1, 
        pos_weight=None, 
        neighbour_smoothing:bool=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.severity_factor = severity_factor
        self.smoothing = smoothing
        self.pos_weight = pos_weight
        self.severity_regression = severity_regression
        self.severity_smoothing = severity_smoothing
        self.neighbour_smoothing = neighbour_smoothing
        if self.neighbour_smoothing:
            self.neighbour_smoothing_weights = torch.as_tensor([[[0.05,0.9,0.05]]]).half().cuda() # hack

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        presence_labels = target[:,:1]
        smoothed_labels = presence_labels * (1.0-self.smoothing) + self.smoothing
        presence_loss = F.binary_cross_entropy_with_logits(input[:,:1], smoothed_labels, pos_weight=self.pos_weight)
        if self.severity_factor <= 0.0:
            return presence_loss

        severity_present = target[:,1] > 0
        if severity_present.sum() > 0:
            if self.severity_regression:
                severity_probability_labels = target[:,1:] * 0.25 - 0.125

                severity_loss = F.binary_cross_entropy_with_logits(
                    input[severity_present,1:], 
                    severity_probability_labels[severity_present]
                )
            else:
                severity_target = target[severity_present,1]-1 # The minus one is because the labels are 0–4 and we want to ignore the zero class
                if self.neighbour_smoothing:
                    severity_target = F.one_hot(severity_target, num_classes=4).half() # hack
                    severity_target = F.pad(severity_target, (1,1), mode="replicate")
                    severity_target = torch.unsqueeze(severity_target, dim=1)
                    severity_target = F.conv1d(severity_target, self.neighbour_smoothing_weights)
                    severity_target = torch.squeeze(severity_target, dim=1)

                    severity_loss = (-severity_target*F.log_softmax(input[severity_present,1:], dim=-1)).sum(dim=-1).mean()
                else:
                    severity_loss = F.cross_entropy(
                        input[severity_present,1:], 
                        severity_target,  
                        # label_smoothing=self.severity_smoothing,
                    )
        else:
            severity_loss = 0.0

        # return severity_loss
        return presence_loss*(1.0-self.severity_factor) + self.severity_factor*severity_loss
