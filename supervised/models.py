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


class SiameseNetwork(nn.Module):
    def __init__(self, embed_dim, pretrained=True):
        super().__init__()
        self.cnn = torchvision.models.resnet50(torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None)
        self.cnn.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, embed_dim)
        )
    
    def forward(self, x1, x2):
        out1 = self.cnn(x1)
        out2 = self.cnn(x2)
        return out1, out2