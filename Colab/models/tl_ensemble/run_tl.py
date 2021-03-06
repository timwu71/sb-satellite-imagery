import sys
from concurrent.futures import ThreadPoolExecutor
import os
import argparse
import csv

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
TUNING = False

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('using device:', device)

# some constants
num_classes = 167
expectation_helper = torch.unsqueeze(torch.arange(num_classes), dim=0)
print_every = 500
transform = data_transform()
num_workers = 84
#BANDS = [[2, 1, 0], [3, 6, 0], [6, 3, 2]]
#BANDS = [[6, 3, 2], [4, 3, 0]]
bands = [2, 1, 0]

# Hyperparameters
tl_model = 'resnet18'
#lrs = [5e-4, 1e-3, 2e-3, 4e-3]
lr = 1e-3
batch_size = 64
# l2 regularization
weight_decay = 1e-3
epochs = 30
frozen_layers = [6]

# Resnet build inspired by https://debuggercafe.com/satellite-image-classification-using-pytorch-resnet34/


def train_epoch(model, optimizer, criterion, loader):
    model.train()
    print('Training...')
    all_preds = []
    all_y = []
    epoch_loss = 0
    counter = 0
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
        # backprop
        loss.backward()
        optimizer.step()

        all_y.append(y.cpu().numpy())
        preds_numpy = preds.detach().cpu().numpy()
        all_preds.append(preds_numpy)
        epoch_loss += loss.item()
        counter += 1
    # loss, r2, accuracy for the complete epoch
    epoch_loss /= counter
    all_preds = np.concatenate(all_preds, axis=0)
    all_y = np.concatenate(all_y, axis=0)
    r2, _ = scipy.stats.pearsonr(all_preds, all_y)
    r2 = r2 ** 2
    epoch_acc = 100. * (np.absolute(all_preds - all_y) < 0.5).sum().item() / all_y.shape[0]
    
    return epoch_loss, r2, epoch_acc

def val_epoch(model, criterion, loader):
    model.eval()
    print('Validating...')
    all_preds = []
    all_y = []
    epoch_loss = 0
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

            all_y.append(y.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            epoch_loss += loss.item()
            counter += 1
    # loss, r2, accuracy for the complete epoch
    epoch_loss /= counter
    all_preds = np.concatenate(all_preds, axis=0)
    all_y = np.concatenate(all_y, axis=0)
    r2, _ = scipy.stats.pearsonr(all_preds, all_y)
    r2 = r2 ** 2
    epoch_acc = 100. * (np.absolute(all_preds - all_y) < 0.5).sum().item() / all_y.shape[0]
    return epoch_loss, r2, epoch_acc

def run_model(lr, weight_decay, tl_model, bands, frozen_layer):
# lists to keep track of losses and accuracies
    print('Fetching Dataloaders...')
    loader_train, loader_val, _ = get_dataloaders(batch_size, num_workers, partial=False, bands=bands)
    train_loss, valid_loss = [], []
    train_r2, valid_r2 = [], []
    train_acc, valid_acc = [], []
    model = build_model(tl_model = tl_model, fine_tune=False, num_classes=num_classes).to(device)
    ct = 0
    for child in model.children():
        ct += 1
        if ct <= frozen_layer:
            for param in child.parameters():
                param.requires_grad = False
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.L1Loss()
    params_info(model)

    best_model = None
    best_r2 = 0
    # start the training
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_r2, train_epoch_acc = train_epoch(model, optimizer, criterion, loader_train)
        valid_epoch_loss, valid_epoch_r2, valid_epoch_acc = val_epoch(model,  criterion, loader_val)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_r2.append(train_epoch_r2)
        valid_r2.append(valid_epoch_r2)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f"Epoch {epoch+1} finished. Final epoch results:")
        print(f"Training loss: {train_epoch_loss:.4f}, training r^2: {train_epoch_r2:.4f} training acc: {train_epoch_acc:.4f}%")
        print(f"Validation loss: {valid_epoch_loss:.4f}, validation r^2: {valid_epoch_r2:.4f} validation acc: {valid_epoch_acc:.4f}%")
        print('-'*75)
        if valid_epoch_r2 > best_r2:
            best_r2 = valid_epoch_r2
            best_model = model
            torch.save(best_model, '/home/timwu0/231nproj/sb-satellite-imagery/saved_data/best_model.pt')
    performance = max(valid_r2)
    print(f"Finished training with {frozen_layer:.1f} layers frozen. Best val r^2: {performance:.4f}")
    return performance, model, train_loss, train_r2, train_acc, valid_loss, valid_r2, valid_acc


if TUNING:
# HYPERPARAMETER TUNING

    best_model = None
    best_r2 = 0
    best_stats = []
    best_frozen_layer = 0
    #best_tl_model = None
    print("Starting hyperparameter tuning...")
    for frozen_layer in frozen_layers:
        r2, model, train_loss, train_r2, train_acc, valid_loss, valid_r2, valid_acc = run_model(lr, weight_decay, tl_model, bands, frozen_layer)
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_stats = [train_loss, train_r2, train_acc, valid_loss, valid_r2, valid_acc]
            best_frozen_layer = frozen_layer
            #best_tl_model = tl_model
    print(f"Best score when freezing {frozen_layer:.1f} layers. Achieved val r^2 of: {best_r2:.4f}")

    print("Saving best model...")

    #PATH = '/home/timwu0/231nproj/sb-satellite-imagery/saved_models.pt'

    #torch.save(best_model, '/home/timwu0/231nproj/sb-satellite-imagery/saved_data/best_model.pt')

    Details = ['Training Loss', 'Training R^2', 'Training Accuracy', 'Validation Loss', 'Validation R^2', 'Validation Accuracy']  
    with open('/home/timwu0/231nproj/sb-satellite-imagery/saved_data/best_stats.csv', 'w') as f: 
        write = csv.writer(f) 
        write.writerow(Details) 
        write.writerows(best_stats) 
else:
    # Test time!
    print('Testing on test set...')
    _, _, loader_test = get_dataloaders(batch_size, num_workers, partial=False, bands=bands)
    model = torch.load('/home/timwu0/231nproj/sb-satellite-imagery/saved_data/best_model.pt')
    criterion = nn.L1Loss()

    test_loss, test_r2, test_acc = val_epoch(model, criterion, loader_test)
    print(f"Test loss: {test_loss:.4f}, test r^2: {test_r2:.4f}, test acc: {test_acc:.4f}%")




#print("Best model saved in ", PATH)

#print("all train losses: ", train_loss)
#print("all train accuracies: ", train_acc)
#print("all val losses: ", valid_loss)
#print("all val accuracies: ", valid_acc)



