import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, x1, x2, label):
        euclidean_distance = F.pairwise_distance(x1, x2)
        loss = (
            label * euclidean_distance.pow(2) 
            + (1 - label) * torch.clamp(self.margin - euclidean_distance, min=0).pow(2)
            )
        return loss.mean()