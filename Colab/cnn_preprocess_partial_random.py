import sys
from concurrent.futures import ThreadPoolExecutor
import os
import magic

import numpy as np
import pandas as pd
import sklearn
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import torch.nn.functional as F  # useful stateless functions

import get_partial_data


# TODO: copy dhs_final_labels.csv to VM, change os path below accordingly
df = pd.read_csv('/home/timwu0/231nproj/data/dhs_final_labels.csv')
df['survey'] = df['DHSID_EA'].str[:10]
df['cc'] = df['DHSID_EA'].str[:2]

# TODO: run modified version of get_public_datasets.py, change data_dir below to match VM path
data_dir = '/home/timwu0/231nproj/data/'
df['path'] = data_dir + df['survey'] + '/' + df['DHSID_EA'] + '.npz'
# df['path'] = dataset_root_dir + '/dhs_npzs/' + df['survey'] + '/' + df['DHSID_EA'] + '.npz'

path_years = df[['DHSID_EA', 'path', 'year']].apply(tuple, axis=1)
df.set_index('DHSID_EA', verify_integrity=True, inplace=True, drop=False) #had to add drop=False to keep column from disappearing  -- R
print(df['path'].iloc[0])
df.info()
  


label = "n_under5_mort"

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
SPLITS['val_partial'] = SPLITS['train'][:2]
SPLITS['test_partial'] = SPLITS['train'][:2]
SPLITS['trainval_partial'] = SPLITS['train_partial'] + SPLITS['val_partial']



train_X, train_Y = get_partial_data.get_data_split(label, 'train_partial', 1)
print("train_X: ", train_X.shape)
print("train_Y: ", train_Y.shape)
print('Saving data in folder /home/timwu0/231nproj/data_clean')
np.savez_compressed('/home/timwu0/231nproj/data_clean/train_partial', train_X=train_X, train_Y=train_Y)

val_X, val_Y = get_partial_data.get_data_split(label, 'val_partial', 1)
print("val_X: ", val_X.shape)
print("val_Y: ", val_Y.shape)
print('Saving data in folder /home/timwu0/231nproj/data_clean')
np.savez_compressed('/home/timwu0/231nproj/data_clean/val_partial', val_X=val_X, val_Y=val_Y)

test_X, test_Y = get_partial_data.get_data_split(label, 'test_partial', 1)
print("test_X: ", test_X.shape)
print("test_Y: ", test_Y.shape)
print('Saving data in folder /home/timwu0/231nproj/data_clean')
np.savez_compressed('/home/timwu0/231nproj/data_clean/test_partial', test_X=test_X, test_Y=test_Y)