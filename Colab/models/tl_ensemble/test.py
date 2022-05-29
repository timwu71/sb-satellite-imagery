import sys
from concurrent.futures import ThreadPoolExecutor
import os
import argparse

import math
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


preds = torch.rand(65)
y = torch.rand(65)

temp = (preds - y) < 0.5
print(temp, temp.sum().item())