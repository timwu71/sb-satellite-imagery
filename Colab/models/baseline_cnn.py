import sys
from concurrent.futures import ThreadPoolExecutor
import os
import magic

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

        train_data = np.load('/home/timwu0/231nproj/data_clean/test.npz', allow_pickle=True)
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
        train_X, train_Y = get_partial_data.get_data_split(label, 'train', 0.02)
        print("train_X: ", train_X.shape)
        print("train_Y: ", train_Y.shape)

        val_X, val_Y = get_partial_data.get_data_split(label, 'val', 0.02)
        print("val_X: ", val_X.shape)
        print("val_Y: ", val_Y.shape)

        test_X, test_Y = get_partial_data.get_data_split(label, 'test', 0.02)
        print("test_X: ", test_X.shape)
        print("test_Y: ", test_Y.shape)
    return torch.from_numpy(train_X), torch.from_numpy(train_Y), torch.from_numpy(val_X), torch.from_numpy(val_Y), torch.from_numpy(test_X), torch.from_numpy(test_Y)



train_X, train_Y, val_X, val_Y, test_X, test_Y = load_data(data='random')
print('finished loading data.')


USE_GPU = True
dtype = torch.float32 

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Constant to control how frequently we print train loss.
print_every = 100
print('using device:', device)



# TODO: find replacement for loader - make function make minibatches

def check_accuracy_part34(X, Y, model, val_or_test):
    if val_or_test == "val":
        print('Checking accuracy on validation set')
    elif val_or_test == "test":
        print('Checking accuracy on test set')

    batch_size = 64
    num_batches = Y.shape[0] // batch_size   
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    
    all_preds = []
    with torch.no_grad():
        model = model.to(device=device)  # move the model parameters to CPU/GPU
        for t in range(num_batches):
          x = X[t*batch_size:(t+1)*batch_size, :, :, :]
          y = Y[t*batch_size:(t+1)*batch_size]
          x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
          y = y.to(device=device, dtype=torch.long)
          scores = model(x).cpu().numpy()
          #print(scores.shape)
          preds = np.argmax(scores, axis=1)
          num_correct += (preds == y.cpu().numpy()).sum()
          num_samples += preds.shape[0]
          
          # for r^2
          print('preds:', preds[:10])
          all_preds.append(preds)
        all_preds = np.concatenate(all_preds, axis=0)
        print(all_preds.shape, Y.cpu().numpy().shape)
        r2, _ = scipy.stats.pearsonr(all_preds, Y.cpu().numpy()[:all_preds.shape[0]])
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc), ' and an r^2 value of', r2)
    return acc, r2


def train_part34(model, optimizer, val_or_test, epochs=1):
    """
    Train a model using the PyTorch Module API.
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: Nothing, but prints model accuracies during training.
    """
    
    batch_size = 64
    if val_or_test == "val":
      X = train_X
      Y = train_Y
    elif val_or_test == "test":
      X = trainval_X
      Y = trainval_Y
    num_batches = Y.shape[0] // batch_size

    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
      # TODO: find replacement for loader to make minibatches x, y from X, Y
        for t in range(num_batches):
          x = X[t*num_batches:(t+1)*num_batches, :, :, :]
          y = Y[t*num_batches:(t+1)*num_batches]
          model.train()  # put model to training mode
          x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
          y = y.to(device=device, dtype=torch.long)
        
          scores = model(x)
          y_one_hots = torch.zeros_like(scores)
          y_one_hots[np.arange(y.size(dim=0)),y] = 1
          # print('scores:', scores, 'y:' , y_one_hots)
          loss = F.cross_entropy(scores, y_one_hots)

          # Zero out all of the gradients for the variables which the optimizer
          # will update.
          optimizer.zero_grad()

          # This is the backwards pass: compute the gradient of the loss with
          # respect to each  parameter of the model.
          loss.backward()

          # Actually update the parameters of the model using the gradients
          # computed by the backwards pass.
          optimizer.step()

          if t % print_every == 0:
              print('Iteration %d, loss = %.4f' % (t, loss.item()))
              check_accuracy_part34(X, Y, model, "val")
              print()


def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

# We need to wrap `flatten` function in a module in order to stack it
# in nn.Sequential
class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)


best_val = 0

model = None
optimizer = None

channel_0 = 8
channel_1 = 32
channel_2 = 16
channel_3 = 16
hidden_layer_size = 32
learning_rate = 1e-3

model = nn.Sequential(
    nn.Conv2d(channel_0, channel_1, (3, 3), padding="same"),
    nn.ReLU(),
    nn.MaxPool2d((2, 2), stride=2),  # changes H, W from 32 to 16
    nn.BatchNorm2d(num_features = channel_1),
    nn.Conv2d(channel_1, channel_2, (3, 3), padding="same"),
    nn.ReLU(),
    nn.MaxPool2d((2, 2), stride=2),  # changes H, W from 16 to 8
    nn.BatchNorm2d(num_features = channel_2),
    nn.Conv2d(channel_2, channel_3, (3, 3), padding="same"),
    nn.ReLU(),
    nn.MaxPool2d((2, 2), stride=2),  # changes H, W from 8 to 4
    nn.BatchNorm2d(num_features = channel_3),
    Flatten(),
    nn.Linear(15376, hidden_layer_size),
    nn.ReLU(),
    nn.Linear(hidden_layer_size, 167),
)

optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)


train_part34(model, optimizer, epochs=5, val_or_test="val")
val_acc, r2 = check_accuracy_part34(val_X, val_Y, model, "val")
if r2 > best_val:
  best_model = model


best_model = model
check_accuracy_part34(test_X, test_Y, best_model, "test")
