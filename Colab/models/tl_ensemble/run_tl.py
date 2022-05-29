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

from models_tl import *
from utils_tl import *
from dataloader_tl import *


USE_GPU = True
dtype = torch.float32 

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('using device:', device)

# some constants
num_classes = 167
expectation_helper = torch.unsqueeze(torch.arange(num_classes), dim=0)
print_every = 500
tl_model = 'resnet18'
transform = data_transform()
num_workers = 84

# Hyperparameters
lr = 1e-3
batch_size = 64
epochs = 1


# Resnet build inspired by https://debuggercafe.com/satellite-image-classification-using-pytorch-resnet34/

model = build_model(tl_model = tl_model, fine_tune=False, num_classes=num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.L1Loss()
params_info(model)

print('Fetching Dataloaders...')
loader_train, loader_val, loader_test = get_dataloaders(batch_size, num_workers, partial=False)
_, loader_val_partial, _ = get_dataloaders(batch_size, num_workers, partial=True)


def train_epoch(model, optimizer, criterion, loader=loader_train):
    model.train()
    print('Training...')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    samples = 0
    for (x, y) in tqdm(loader, bar_format='{l_bar}{bar:40}{r_bar}{bar:-40b}'):
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.float32)
        optimizer.zero_grad()
        # forward pass
        outputs = model(x)
        # calculate the loss
        # One hots in case we want them
        #y_one_hots = torch.zeros_like(outputs)
        #y_one_hots[np.arange(y.size(dim=0)),y] = 1

        # calculate expectation
        preds = ((outputs * expectation_helper.to(device)).sum(dim=1)/outputs.sum(dim=1)).type(torch.float32)
        
        loss = criterion(preds, y)
        
        counter += 1
        samples += y.size(0)
        train_running_loss += loss.item()
        # calculate the accuracy
        train_running_correct += (torch.abs(preds - y) < 0.5).sum().item()
        # backprop
        loss.backward()
        optimizer.step()

        #if counter % print_every == 0:
        #    mid_epoch_train_loss = train_running_loss / samples
        #    mid_epoch_train_acc = 100. * (train_running_correct / samples)
        #    mid_epoch_val_loss, mid_epoch_val_acc = val_epoch(model, criterion, loader=loader_val_partial)
        #    print(f"Training loss: {mid_epoch_train_loss:.3f}, training acc: {mid_epoch_train_acc:.3f} %")
        #    print(f"Validation loss: {mid_epoch_val_loss:.3f}, validation acc: {mid_epoch_val_acc:.3f} %")
    
    # loss and accuracy for the complete epoch
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / samples)
    return epoch_loss, epoch_acc

def val_epoch(model, criterion, loader=loader_val):
    model.eval()
    print('Validating...')
    val_running_loss = 0.0
    val_running_correct = 0
    samples = 0
    counter = 0
    with torch.no_grad():
        for (x, y) in tqdm(loader, bar_format='{l_bar}{bar:40}{r_bar}{bar:-40b}'):
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)

            # forward pass
            outputs = model(x)
            # calculate the loss
            preds = (outputs * expectation_helper.to(device)).sum(dim=1)/outputs.sum(dim=1)
            loss = criterion(preds, y)
            
            counter += 1
            samples += y.size(0)
            val_running_loss += loss.item()
            # calculate the accuracy
            val_running_correct += (torch.abs(preds - y) < 0.5).sum().item()  
            
            #if samples > 16000:
            #    print('preds: ', preds, 'y: ', y, 'preds - y: ', preds - y)
    
    # loss and accuracy for the complete epoch
    epoch_loss = val_running_loss / counter
    epoch_acc = 100. * (val_running_correct / samples)
    return epoch_loss, epoch_acc


# lists to keep track of losses and accuracies
train_loss, valid_loss = [], []
train_acc, valid_acc = [], []
# start the training
for epoch in range(epochs):
    print(f"[INFO]: Epoch {epoch+1} of {epochs}")
    train_epoch_loss, train_epoch_acc = train_epoch(model, optimizer, criterion, loader=loader_train)
    valid_epoch_loss, valid_epoch_acc = val_epoch(model,  criterion, loader=loader_val)
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    train_acc.append(train_epoch_acc)
    valid_acc.append(valid_epoch_acc)
    print(f"Epoch {epoch+1} finished. Final epoch results:")
    print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}%")
    print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}%")
    print('-'*75)

#print("all train losses: ", train_loss)
#print("all train accuracies: ", train_acc)
#print("all val losses: ", valid_loss)
#print("all val accuracies: ", valid_acc)



