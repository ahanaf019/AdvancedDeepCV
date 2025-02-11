import torch
import torch.nn as nn
import torchvision
from torchvision import transforms


class CNNModel(nn.Module):
    def __init__(self, num_classes, hidd_dim=512, pretrained=True):
        super().__init__()
        self.cnn = torchvision.models.resnet50(torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None)
        self.cnn.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(2048, hidd_dim),
            nn.BatchNorm1d(hidd_dim),
            nn.ReLU(True),
            nn.Linear(hidd_dim, num_classes)
        )
    
    def forward(self, x):
        return self.cnn(x)