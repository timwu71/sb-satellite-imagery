import sys
from concurrent.futures import ThreadPoolExecutor
import os
import argparse

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
# Note: when testing on your own may have to change this to reset your python path
import get_partial_data

def load_data(data):
# parameter data is 'all', 'partial', or 'random'. 
# 'all' and 'partial' loads the respective .npz stored files, 'random' randomly generates new partial data.
    print('loading data...')

    if data == 'all':
        # load all data
        train_data = np.load('/home/timwu0/231nproj/data_clean/train.npz', allow_pickle=True)
        train_X, train_Y = train_data['train_X'], train_data['train_Y']
        print("train_X: ", train_X.shape)
        print("train_Y: ", train_Y.shape)

        val_data = np.load('/home/timwu0/231nproj/data_clean/val.npz', allow_pickle=True)
        val_X, val_Y = val_data['val_X'], val_data['val_Y']
        print("val_X: ", val_X.shape)
        print("val_Y: ", val_Y.shape)

        trainval_X, trainval_Y = np.concatenate((train_data['train_X'], val_data['val_X']), axis=0), np.concatenate((train_data['train_Y'], val_data['val_Y']), axis=0)
        print("trainval_X: ", trainval_X.shape)
        print("trainval_Y: ", trainval_Y.shape)

        test_data = np.load('/home/timwu0/231nproj/data_clean/test.npz', allow_pickle=True)
        test_X, test_Y = test_data['test_X'], test_data['test_X'], 
        print("test_X: ", test_X.shape)
        print("test_Y: ", test_Y.shape)
    elif data == 'partial':
        train_data = np.load('/home/timwu0/231nproj/data_clean/train_partial.npz', allow_pickle=True)
        train_X, train_Y = train_data['train_X'], train_data['train_Y']
        print("train_X: ", train_X.shape)
        print("train_Y: ", train_Y.shape)

        val_data = np.load('/home/timwu0/231nproj/data_clean/val_partial.npz', allow_pickle=True)
        val_X, val_Y = val_data['val_X'], val_data['val_Y']
        print("val_X: ", val_X.shape)
        print("val_Y: ", val_Y.shape)

        trainval_X, trainval_Y = np.concatenate((train_data['train_X'], val_data['val_X']), axis=0), np.concatenate((train_data['train_Y'], val_data['val_Y']), axis=0)
        print("trainval_X: ", trainval_X.shape)
        print("trainval_Y: ", trainval_Y.shape)

        test_data = np.load('/home/timwu0/231nproj/data_clean/test_partial.npz', allow_pickle=True)
        test_X, test_Y = test_data['test_X'], test_data['test_X'], 
        print("test_X: ", test_X.shape)
        print("test_Y: ", test_Y.shape)
    else: 
        print('generating random data...')
        label = "n_under5_mort" 
        train_X, train_Y = get_partial_data.get_data_split(label, 'train', 0.0005)
        print("train_X: ", train_X.shape)
        print("train_Y: ", train_Y.shape)

        val_X, val_Y = get_partial_data.get_data_split(label, 'val', 0.0005)
        print("val_X: ", val_X.shape)
        print("val_Y: ", val_Y.shape)

        test_X, test_Y = get_partial_data.get_data_split(label, 'test', 0.0005)
        print("test_X: ", test_X.shape)
        print("test_Y: ", test_Y.shape)
    return torch.from_numpy(train_X), torch.from_numpy(train_Y), torch.from_numpy(val_X), torch.from_numpy(val_Y), torch.from_numpy(test_X), torch.from_numpy(test_Y)
