# Preprocessing Data for EC2 Instance

# This code is used to preprocess the data on the EC2 Instance.
# You should only have to run it **ONCE**, the cleaned data should be saved to this folder on the instance:
# [insert folder].

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

# dhs_final_labels.csv file in 231nproj
df = pd.read_csv('../../../../231nproj/dhs_final_labels.csv')
df['survey'] = df['DHSID_EA'].str[:10]
df['cc'] = df['DHSID_EA'].str[:2]

data_dir = '../../../../231nproj/data/'
df['path'] = data_dir + df['survey'] + '/' + df['DHSID_EA'] + '.npz'

path_years = df[['DHSID_EA', 'path', 'year']].apply(tuple, axis=1)
df.set_index('DHSID_EA', verify_integrity=True, inplace=True, drop=False) #had to add drop=False to keep column from disappearing  -- R
print(df['path'].iloc[0])
df.info()

def paths_to_X(paths):  # -> (N, C, H, W) model input X
  '''
    Args
    - paths: array (N, 1)
      - path: str, path to npz file containing single entry 'x'
        representing a (C, H, W) image

    Returns: X, input matrix (N, C, H, W)
    '''
  N = len(paths)  # should be 117644
  C, H, W = 8, 255, 255

  imgs = []
  for n in range(N):
    npz_path = paths[n][0]
    imgs.append(np.load(npz_path)['x'])  # shape (C, H, W)
    if n % 2000  == 0:
        print('On example', n)

  return np.stack(imgs, axis=0)

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

SPLITS['train_partial'] = SPLITS['train'][:5] #used to be:5
SPLITS['val_partial'] = SPLITS['val'][:2] #used to be :2
SPLITS['test_partial'] = SPLITS['test'][:2]	#used to be :2
SPLITS['trainval_partial'] = SPLITS['train_partial'] + SPLITS['val_partial']


def get_data_split(label, split):
    train_dhsids = df.index[df['cc'].isin(SPLITS[split]) & df[label].notna()]

    train_X_paths = df.loc[train_dhsids, 'path'].values.reshape(-1, 1)
    train_X = paths_to_X(train_X_paths)
    train_Y = df.loc[train_dhsids, label].values

    # knn.fit(train_X, train_Y)
    # preds = knn.predict(test_X)
    return train_X, train_Y

data_dir = '../../../../231nproj/'
train_X, train_Y = get_data_split(label, 'train_partial')
print("train_X: ", train_X.shape)
print("train_Y: ", train_Y.shape)
print('Saving data in folder' + str(data_dir) + 'data_clean/train_partial')
np.savez_compressed(str(data_dir) + 'data_clean/train_partial', train_X=train_X, train_Y=train_Y)

val_X, val_Y = get_data_split(label, 'val_partial')
print("val_X: ", val_X.shape)
print("val_Y: ", val_Y.shape)
print('Saving data in folder' + str(data_dir) + 'data_clean/val_partial')
np.savez_compressed(str(data_dir) + 'data_clean/val_partial', val_X=val_X, val_Y=val_Y)

test_X, test_Y = get_data_split(label, 'test_partial')
print("test_X: ", test_X.shape)
print("test_Y: ", test_Y.shape)
print('Saving data in folder' + str(data_dir) + 'data_clean/test_partial')
np.savez_compressed(str(data_dir) + 'data_clean/test_partial', test_X=test_X, test_Y=test_Y)


# train_X, train_Y = get_data_split(label, 'train')
# print("train_X: ", train_X.shape)
# print("train_Y: ", train_Y.shape)
# print('Saving data in folder' + str(data_dir) + '/data_clean/train')
# np.savez_compressed(str(data_dir) + 'data_clean/train', train_X=train_X, train_Y=train_Y)

# val_X, val_Y = get_data_split(label, 'val')
# print("val_X: ", val_X.shape)
# print("val_Y: ", val_Y.shape)
# print('Saving data in folder' + str(data_dir) + '/data_clean/val')
# np.savez_compressed(str(data_dir) + 'data_clean/val', val_X=val_X, val_Y=val_Y)

# test_X, test_Y = get_data_split(label, 'test')
# print("test_X: ", test_X.shape)
# print("test_Y: ", test_Y.shape)
# print('Saving data in folder' + str(data_dir) + '/data_clean/test')
# np.savez_compressed(str(data_dir) + 'data_clean/test', test_X=test_X, test_Y=test_Y)
