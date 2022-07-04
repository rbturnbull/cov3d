from torch import nn
from torch import Tensor
import torch
import torch.nn.functional as F

from fastai.torch_core import default_device

class Cov3dLoss(nn.Module):
    def __init__(
        self, 
        severity_factor:float=0.5, 
        presence_smoothing:float=0.1, 
        severity_regression:bool=False, 
        severity_smoothing:float=0.1, 
        pos_weight=None, 
        neighbour_smoothing:bool=False,
        mse:bool=False,
        severity_everything:bool = False,
        divide_sum_weights:bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.device = default_device()
        self.severity_factor = severity_factor
        self.presence_smoothing = presence_smoothing
        self.pos_weight = pos_weight
        self.severity_regression = severity_regression
        self.mse = mse
        self.severity_smoothing = severity_smoothing
        self.neighbour_smoothing = neighbour_smoothing
        self.divide_sum_weights = divide_sum_weights
        if self.neighbour_smoothing:
            self.neighbour_smoothing_weights = torch.as_tensor([[[
                self.severity_smoothing*0.5,
                1.0-self.severity_smoothing,
                self.severity_smoothing*0.5,
            ]]]).half().to(self.device) # hack

        self.severity_everything = severity_everything
        if self.severity_everything:
            big = 1.0-self.severity_smoothing
            little = self.severity_smoothing*0.5

            # temporary hack. Numbers should be read in from the input files
            total = 2292
            total_by_groups = total / 6
            n_mild = 85
            n_moderate = 62
            n_severe = 85
            n_critical = 26
            n_negative = 1110
            n_positive = total - n_negative - n_critical - n_severe - n_moderate - n_mild

            self.mild_weight = total_by_groups/n_mild
            self.moderate_weight = total_by_groups/n_moderate
            self.severe_weight = total_by_groups/n_severe
            self.critical_weight = total_by_groups/n_critical
            self.negative_weight = total_by_groups/n_negative
            self.positive_weight = total_by_groups/n_positive
            # self.every_weights = torch.as_tensor([self.mild_weight, self.moderate_weight, self.severe_weight, self.critical_weight, self.negative_weight]).to(self.device)
            self.target_mild = torch.as_tensor([big, little, 0.0, 0.0, little]).to(self.device) * self.mild_weight
            self.target_moderate = torch.as_tensor([little, big, little, 0.0, 0.0]).to(self.device) * self.moderate_weight
            self.target_severe = torch.as_tensor([0.0, little, big, little, 0.0]).to(self.device) * self.severe_weight
            self.target_critical = torch.as_tensor([0.0, 0.0, self.severity_smoothing, big, 0.0]).to(self.device) * self.critical_weight
            # self.target_positive = torch.as_tensor([0.25, 0.25, 0.25, 0.25, 0.0]).to(self.device) * self.positive_weight
            self.target_negative = torch.as_tensor([self.severity_smoothing, 0.0, 0.0, 0.0, big]).to(self.device) * self.negative_weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        presence_labels = target[:,:1]
        smoothed_labels = presence_labels * (1.0-self.presence_smoothing*2.0) + self.presence_smoothing
        presence_loss = F.binary_cross_entropy_with_logits(input[:,:1], smoothed_labels, pos_weight=self.pos_weight)
        if self.severity_factor <= 0.0:
            return presence_loss

        if self.severity_everything:
            weights = 0.0
            positive_cases = (target[:,1] == 0) & (target[:,0] == 1)
            severity_loss = 0.0
            if torch.all(positive_cases):
                # severity_predictions = input[positive_cases,1:]
                severity_probabilities = torch.softmax(input[positive_cases,1:], dim=-1)
                severity_loss += - torch.log(severity_probabilities[:,:-1].sum(dim=-1)).sum()

                # severity_predictions_sum = torch.cat([severity_predictions[:,0:4].sum(dim=-1, keepdim=True), severity_predictions[:,4:]], dim=1)

                weights += positive_cases.sum() * self.positive_weight
                # severity_loss += -F.log_softmax(severity_predictions_sum, dim=-1)[:,0].sum()
            else:
                severity_predictions = input[~positive_cases,1:]
                severity_target = torch.zeros_like(severity_predictions)
                non_positive_targets = target[~positive_cases]

                mild_cases = non_positive_targets[:,1] == 1
                moderate_cases = non_positive_targets[:,1] == 2
                severe_cases = non_positive_targets[:,1] == 3
                critical_cases = non_positive_targets[:,1] == 4
                negative_cases = (non_positive_targets[:,1] == 0) & (non_positive_targets[:,0] == 0)

                if torch.any(mild_cases):
                    severity_target[ mild_cases, : ] = self.target_mild.repeat( mild_cases.sum(), 1)
                    weights += mild_cases.sum() * self.mild_weight
                if torch.any(moderate_cases):
                    severity_target[ moderate_cases, : ] = self.target_moderate.repeat( moderate_cases.sum(), 1)
                    weights += moderate_cases.sum() * self.moderate_weight
                if torch.any(severe_cases):
                    severity_target[ severe_cases, : ] = self.target_severe.repeat( severe_cases.sum(), 1)
                    weights += severe_cases.sum() * self.severe_weight
                if torch.any(critical_cases):
                    severity_target[ critical_cases, : ] = self.target_critical.repeat( critical_cases.sum(), 1)
                    weights += critical_cases.sum() * self.critical_weight
                if torch.any(negative_cases):
                    severity_target[ negative_cases, : ] = self.target_negative.repeat( negative_cases.sum(), 1)
                    weights += negative_cases.sum() * self.negative_weight

                severity_loss += (-severity_target*F.log_softmax(severity_predictions, dim=-1)).sum(dim=-1).sum()
            
            if self.divide_sum_weights:
                severity_loss /= weights
            else:
                severity_loss /= input.shape[0]
        else:
            severity_present = target[:,1] > 0
            if severity_present.sum() > 0:
                if self.severity_regression:
                    severity_probability_labels = target[:,1:] * 0.25 - 0.125

                    if self.mse:
                        severity_target_logits = torch.logit(severity_probability_labels)

                        severity_loss = F.mse_loss(
                            input[severity_present,1:], 
                            severity_target_logits,
                        )
                    else:
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
