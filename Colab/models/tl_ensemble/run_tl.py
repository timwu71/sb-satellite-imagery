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

from models_tl import *
from utils_tl import *
from data_tl import *


train_X, train_Y, val_X, val_Y, test_X, test_Y = load_data(data='random')
print('finished loading data.')

USE_GPU = True
dtype = torch.float32 

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('using device:', device)

# Constant to control how frequently we print train loss.
print_every = 200
tl_model = 'resnet18'
transform = data_transform()

# Hyperparameters
lr = 1e-3
num_batches = 32
epochs = 10


# Resnet build inspired by https://debuggercafe.com/satellite-image-classification-using-pytorch-resnet34/

model = build_model(tl_model = tl_model, fine_tune=False, num_classes=167).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()
params_info(model)

def train_epoch(model, optimizer, criterion, num_batches):
    model.train()
    print('Training...')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for t in range(num_batches):
        x = train_X[t*num_batches:(t+1)*num_batches, :, :, :]
        y = train_Y[t*num_batches:(t+1)*num_batches]
        counter += 1
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        # forward pass
        outputs = model(x)
        # calculate the loss
        loss = criterion(outputs, y)
        train_running_loss += loss.item()
        # calculate the accuracy
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == y).sum().item()
        # backrpop
        loss.backward()
        optimizer.step()
    
    # loss and accuracy for the complete epoch
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / y.size(dim=0))
    return epoch_loss, epoch_acc

def val_epoch(model, criterion, num_batches):
    model.eval()
    print('Validating...')
    val_running_loss = 0.0
    val_running_correct = 0
    counter = 0
    with torch.no_grad():
        for t in range(num_batches):
            x = train_X[t*num_batches:(t+1)*num_batches, :, :, :]
            y = train_Y[t*num_batches:(t+1)*num_batches]
            counter += 1
            x = x.to(device)
            y = y.to(device)    
            # forward pass
            outputs = model(x)
            # calculate the loss
            loss = criterion(outputs, x)
            val_running_loss += loss.item()
            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            val_running_correct += (preds == y).sum().item()      
    
    # loss and accuracy for the complete epoch
    epoch_loss = val_running_loss / counter
    epoch_acc = 100. * (val_running_correct / y.size(dim=0))
    return epoch_loss, epoch_acc


# lists to keep track of losses and accuracies
train_loss, valid_loss = [], []
train_acc, valid_acc = [], []
# start the training
for epoch in range(epochs):
    print(f"[INFO]: Epoch {epoch+1} of {epochs}")
    train_epoch_loss, train_epoch_acc = train_epoch(model, optimizer, criterion, num_batches=num_batches)
    valid_epoch_loss, valid_epoch_acc = val_epoch(model,  criterion, num_batches=num_batches)
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    train_acc.append(train_epoch_acc)
    valid_acc.append(valid_epoch_acc)
    print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
    print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
    print('-'*50)






