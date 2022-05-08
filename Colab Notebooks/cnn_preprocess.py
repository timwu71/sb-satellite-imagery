import sys
from concurrent.futures import ThreadPoolExecutor
import os
import magic

import numpy as np
import pandas as pd
import sklearn
from tqdm.auto import tqdm


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
  X = np.zeros((N, C, H, W))

  imgs = []
  for n in range(N):
    npz_path = paths[n][0]
    imgs.append(np.load(npz_path)['x'])  # shape (C, H, W)
    if n % 1000 == 0:
        print('On example', n)
  np.concatenate(imgs, axis=0)
  return X


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


def get_train_and_test(label, trainsplit='train', testsplit='test'):
    train_dhsids = df.index[df['cc'].isin(SPLITS[trainsplit]) & df[label].notna()]
    test_dhsids = df.index[df['cc'].isin(SPLITS[testsplit]) & df[label].notna()]

    train_X_paths = df.loc[train_dhsids, 'path'].values.reshape(-1, 1)
    train_X = paths_to_X(train_X_paths)
    train_Y = df.loc[train_dhsids, label].values
    test_X_paths = df.loc[test_dhsids, 'path'].values.reshape(-1, 1)
    test_X = paths_to_X(test_X_paths)
    test_Y = df.loc[test_dhsids, label].values

    # knn.fit(train_X, train_Y)
    # preds = knn.predict(test_X)
    return train_X, train_Y, test_X, test_Y


train_X, train_Y, val_X, val_Y = get_train_and_test(label, 'train', 'val') 
trainval_X, trainval_Y, test_X, test_Y = get_train_and_test(label, 'trainval', 'test')
print("train_X: ", train_X.shape)
print("train_Y: ", train_Y.shape)
print("trainval_X: ", trainval_X.shape)
print("trainval_Y: ", trainval_Y.shape)
print("val_X: ", val_X.shape)
print("val_Y: ", val_Y.shape)
print("test_X: ", test_X.shape)
print("test_Y: ", test_Y.shape)

print('Saving data in /home/timwu0/231nproj/clean_data.npz')
np.savez_compressed('/home/timwu0/231nproj/clean_data', train_X=train_X, 
                                                        train_Y=train_Y, 
                                                        val_X=val_X, 
                                                        val_Y=val_Y, 
                                                        trainval_X=trainval_X,
                                                        trainval_Y=trainval_Y, 
                                                        test_X=test_X, 
                                                        test_Y=test_Y)