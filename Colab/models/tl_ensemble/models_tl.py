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

def build_model(tl_model, fine_tune=True, num_classes=167):
    if tl_model == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif tl_model == 'resnet34':
        model = models.resnet34(pretrained=True)
    elif tl_model == 'resnet50':
        model = models.resnet50(pretrained=True)

    if fine_tune:
        for params in model.parameters():
            params.requires_grad = True
    else:
        for params in model.parameters():
            params.requires_grad = False
            
    model.fc = nn.Linear(512, num_classes)
    return model
