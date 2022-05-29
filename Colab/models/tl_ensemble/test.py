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

batch_size = 32

train_X = torch.rand(65, 8, 255, 255)
train_Y = torch.rand(65)

num_batches = int(math.ceil(train_Y.size(dim=0) / batch_size))
print(num_batches)
for t in range(num_batches):
    x = train_X[t*batch_size:(t+1)*batch_size]
    y = train_Y[t*batch_size:(t+1)*batch_size]
    print("t: ", t, "x, y sizes:", x.size(), y.size())