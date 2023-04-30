from Module import cifar10
import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from transformer import Transformer
from ctypes import *
import client
from seal import *

class cifar10(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sequential = torch.nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.Conv2d(64, 64, 5, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 5, padding=1),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 5, padding=1),
            nn.Conv2d(128, 256, 5, padding=1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 5, padding=1),
            nn.MaxPool2d(2, 2),
            nn.AvgPool2d(2, 2),
            nn.AvgPool2d(8, 8),
            nn.Dropout(0.5),
            nn.Dropout(0.1),
            nn.Linear(256, 10)
        )

    def forward(self, tens):
        tens = self.sequential(tens)
        return tens
    
if __name__ == "__main__":
    print(torch.zeros(1).float())