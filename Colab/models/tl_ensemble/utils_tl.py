import sys
from concurrent.futures import ThreadPoolExecutor
import os

import numpy as np
import pandas as pd
import sklearn
from tqdm.auto import tqdm
import scipy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import torch.nn.functional as F  # useful stateless functions

from torch.utils.data import Subset
from torchvision import datasets, transforms
import torchvision.models as models

def params_info(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.\n")
    return 0

def data_transform():
    return transforms.Compose([
    #transforms.Resize((255, 255)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
        )
    ])
