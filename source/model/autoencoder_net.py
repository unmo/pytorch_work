# -*-coding: utf-8 -*-

import numpy as np
import torch
from sklearn.datasets import fetch_openml
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


class Encoder(torch.nn.Module):
    def __init__(self, data_size, layer_num):
        super().__init__()
        pass

    def forward(self, x):
        return x


class Decoder(torch.nn.Module):
    def __init__(self, data_size, layer_num):
        super().__init__()
        pass

    def forward(self, x):
        return x


class StackedAutoEncoder(torch.nn.Module):
    def __init__(self, data_size, layer_num):
        super().__init__()
        self.enc = Encoder(data_size, layer_num)
        self.dec = Decoder(data_size, layer_num)

    def forward(self, x):
        middle = self.enc(x)
        x = self.dec(middle)

        return x


if __name__ == "__main__":
    pass
