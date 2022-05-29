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
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import torch.nn.functional as F  # useful stateless functions
import scipy

from PIL import Image
# Note: when testing on your own may have to change this to reset your python path
from utils_tl import *


label = "n_under5_mort"
dataset_root_dir = "/home/timwu0/231nproj/data/"

SPLITS = {
    'train': [
        'AL', 'BD', 'CD', 'CM', 'GH', 'GU', 'HN', 'IA', 'ID', 'JO', 'KE', 'KM',
        'LB', 'LS', 'MA', 'MB', 'MD', 'MM', 'MW', 'MZ', 'NG', 'NI', 'PE', 'PH',
        'SN', 'TG', 'TJ', 'UG', 'ZM', 'ZW'],
    'val': [
        'BF', 'BJ', 'BO', 'CO', 'DR', 'GA', 'GN', 'GY', 'HT', 'NM', 'SL', 'TD',
        'TZ'],
    'test': [
        'AM', 'AO', 'BU', 'CI', 'EG', 'ET', 'KH', 'KY', 'ML', 'NP', 'PK', 'RW',
        'SZ']
}
SPLITS['trainval'] = SPLITS['train'] + SPLITS['val']

#partial splits

SPLITS['train_partial'] = SPLITS['train'][:5]
SPLITS['val_partial'] = SPLITS['val'][:5]
SPLITS['test_partial'] = SPLITS['test'][:5]
SPLITS['trainval_partial'] = SPLITS['train_partial'] + SPLITS['val_partial']

class SustainBenchDataset(Dataset):
    def __init__(self, annotations_file, img_dir, file_ext, split, category, bands=None, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.split = split
        self.bands = bands
        self.category = category
        self.img_labels['survey'] = self.img_labels['DHSID_EA'].str[:10]
        self.img_labels['cc'] = self.img_labels['DHSID_EA'].str[:2]
        # Set up dataframe to have accurate path names
        self.img_labels['survey'] = self.img_labels['DHSID_EA'].str[:10]
        self.img_labels['cc'] = self.img_labels['DHSID_EA'].str[:2]
        self.img_labels['path'] = img_dir + self.img_labels['survey'] + '/' + self.img_labels['DHSID_EA'] + file_ext
        # Only include necessary countries' data with non NaN values
        self.df_split = self.img_labels[self.img_labels['cc'].isin(SPLITS[split]) & self.img_labels[category].notna()].copy()
        path_years = self.df_split[['DHSID_EA', 'path', 'year']].apply(tuple, axis=1)
        self.df_split.set_index('DHSID_EA', verify_integrity=True, inplace=True, drop=False) #drop=False to keep column from disappearing
        print()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df_split)

    def __getitem__(self, idx):
        df_row = self.df_split.iloc[idx]
        image = np.load(df_row['path'])['x']  # with all 8 channels 

        # Reduce to 3 bands/channels at a time if needed
        if self.bands is not None:
          image = np.load(df_row['path'])['x'][self.bands, :, :]
        
        label = df_row[self.category]
        image = Image.fromarray(image)
        # Apply transforms if needed
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

def get_dataloaders():


    train_dataset = SustainBenchDataset(
        annotations_file='/home/timwu0/231nproj/data/dhs_final_labels.csv',
        img_dir='/home/timwu0/231nproj/data/',
        category = 'n_under5_mort',
        file_ext = '.npz',
        split = 'train_partial',
        bands = [2, 1, 0],
        transform=data_transform()
    )

    val_dataset = SustainBenchDataset(
        annotations_file='/home/timwu0/231nproj/data/dhs_final_labels.csv',
        img_dir='/home/timwu0/231nproj/data/',
        category = 'n_under5_mort',
        file_ext = '.npz',
        split = 'val_partial',  #TODO: CHANGE THIS TO VAL
        bands = [2, 1, 0],
        transform=data_transform()
    )

    test_dataset = SustainBenchDataset(
        annotations_file='/home/timwu0/231nproj/data/dhs_final_labels.csv',
        img_dir='/home/timwu0/231nproj/data/',
        category = 'n_under5_mort',
        file_ext = '.npz',
        split = 'test_partial',  #TODO: CHANGE THIS TO TEST
        bands = [2, 1, 0],
        transform=data_transform()
    )

    loader_train = DataLoader(train_dataset, batch_size=64, num_workers=64)

    loader_val = DataLoader(val_dataset, batch_size=64, num_workers=64)

    loader_test = DataLoader(test_dataset, batch_size=64, num_workers=64)

    return loader_train, loader_val, loader_test