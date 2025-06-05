# src/python/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple3DCNN(nn.Module):
    def __init__(self, num_classes):
        super(Simple3DCNN, self).__init__()
        # Input: (batch, 3, 16, 224, 224)
        self.conv1 = nn.Conv3d(3, 8, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.conv2 = nn.Conv3d(8, 16, kernel_size=(3, 3, 3), padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        # After pool2: shape (batch, 16, 8, 56, 56)
        self.fc1 = nn.Linear(16 * 8 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)   # → (batch, 8, 16, 112, 112)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)   # → (batch, 16, 8, 56, 56)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
