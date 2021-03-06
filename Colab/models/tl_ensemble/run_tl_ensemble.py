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
from torch.autograd import Variable


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
transform = data_transform()
num_workers = 84
BANDS = [[2, 1, 0], [6, 3, 2], [4, 3, 0]]

# Hyperparameters
tl_model = 'resnet18'
#lrs = [5e-4, 1e-3, 2e-3, 4e-3]
lr = 1e-3
batch_size = 64
# l2 regularization
weight_decay = 1e-3
epochs = 5


# Resnet build inspired by https://debuggercafe.com/satellite-image-classification-using-pytorch-resnet34/


def train_epoch(models, optimizer, criterion, loader, ensemble_weights):
    for model in models:
        model.train()
    print('Training...')
    all_preds = []
    all_y = []
    epoch_loss = 0
    counter = 0
    w1, w2, w3 = ensemble_weights[0], ensemble_weights[1], ensemble_weights[2]
    for (x, y) in tqdm(loader, bar_format='{l_bar}{bar:40}{r_bar}{bar:-40b}'):
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.float32)
        optimizer.zero_grad()
        # forward pass
        outputs = []
        for model in models:
            outputs.append(model(x))
        # calculate the loss
        # One hots in case we want them
        #y_one_hots = torch.zeros_like(outputs)
        #y_one_hots[np.arange(y.size(dim=0)),y] = 1
        preds = []
        # calculate expectation
        for output in outputs:
            preds.append(((outputs * expectation_helper.to(device)).sum(dim=1)/outputs.sum(dim=1)).type(torch.float32))    
        pred = w1 * preds[0] + w2 * preds[1] + w3 * preds[2]
        loss = criterion(pred, y)
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
    
    ensemble_weights = w1, w2, w3
    return epoch_loss, r2, epoch_acc, ensemble_weights

def val_epoch(model, criterion, loader, ensemble_weights):
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

def run_model(lr, weight_decay, tl_model, all_bands):
# lists to keep track of losses and accuracies
    train_loss, valid_loss = [], []
    train_r2, valid_r2 = [], []
    train_acc, valid_acc = [], []
    models = []
    print('Fetching Dataloaders...')
    for bands in all_bands:
        loader_train, loader_val, loader_test = get_dataloaders(batch_size, num_workers, partial=False, bands=bands)
        model = build_model(tl_model = tl_model, fine_tune=False, num_classes=num_classes).to(device)
        params_info(model)
        models.append(model)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.L1Loss()

    w1 = Variable(torch.randn(1).type(dtype=torch.float32), requires_grad=True)

    
    # start the training
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_r2, train_epoch_acc = train_epoch(models, optimizer, criterion, loader_train)
        valid_epoch_loss, valid_epoch_r2, valid_epoch_acc = val_epoch(models,  criterion, loader_val)
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
    performance = max(valid_r2)
    print(f"Finished training with bands {bands}. Best val r^2: {performance:.4f}")
    return performance, model

# HYPERPARAMETER TUNING

best_model = None
best_r2 = 0
#best_tl_model = None
print("Starting hyperparameter tuning...")
r2, model = run_model(lr, weight_decay, tl_model, BANDS)
best_r2 = r2
best_model = model
        #best_tl_model = tl_model
# print(f"Achieved val r^2 of: {best_r2:.4f}")

print("Saving best model...")

PATH = '/home/timwu0/231nproj/sb-satellite-imagery/saved_models.pt'

torch.save(best_model, '/home/timwu0/231nproj/sb-satellite-imagery/saved_models.pt')

print("Best model saved in ", PATH)

#print("all train losses: ", train_loss)
#print("all train accuracies: ", train_acc)
#print("all val losses: ", valid_loss)
#print("all val accuracies: ", valid_acc)



