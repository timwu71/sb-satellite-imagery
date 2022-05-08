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


def get_data_split(df, label, split, frac):
    train_dhsids = df.index[df['cc'].isin(SPLITS[split]) & df[label].notna()]
    
    if frac != 1:
      train_dhsids = train_dhsids.sample(frac=frac)

    train_X_paths = df.loc[train_dhsids, 'path'].values.reshape(-1, 1)
    train_X = paths_to_X(train_X_paths)
    train_Y = df.loc[train_dhsids, label].values
    
    # knn.fit(train_X, train_Y)
    # preds = knn.predict(test_X)
    return train_X, train_Y
