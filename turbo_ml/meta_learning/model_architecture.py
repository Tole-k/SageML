import torch
import torch.nn as nn


class ModelArchitecture(nn.Module):
    def __init__(self, num_features: int, num_classes: int):
        super(ModelArchitecture, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Sigmoid(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Sigmoid(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )
        self.fc1 = nn.Linear(num_features, 256)
        self.fc4 = nn.Linear(64, num_classes)

    def forward(self, x):
        return self.nn(x)
