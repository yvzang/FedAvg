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
    '''sealer = Seal()
    module = cifar10()
    module1 = cifar10()
    trans = Transformer()

    params_list = trans.para_to_list(module.state_dict(), module)
    params_list1 = trans.para_to_list(module1.state_dict(), module1)
    encrypted_params = sealer.encrypt(params_list)
    encrypted_params1 = sealer.encrypt(params_list1)


    mut_encrypted = sealer.mutiple(encrypted_params, encrypted_params1)
    decrypted_params = sealer.decrypt(mut_encrypted, params_list.__len__())
    '''

    m = cifar10()
    input = torch.randn(20, 3, 32, 32)
    out = m(input)
    print(out.shape)
