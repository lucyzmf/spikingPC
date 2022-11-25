# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import torch.utils.data as data
import sys
import os
import shutil

from torch.nn.parameter import Parameter
from torch.nn import init
from torch.autograd import Variable
import torchvision.transforms as T
import torchvision
import math
import numpy as np

import matplotlib.pyplot as plt
import IPython.display as ipd

from tqdm import tqdm

from network import *
from utils import *

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# set seed
torch.manual_seed(999)

# %%
# TODO data preprocessing 
###############################################################
# IMPORT DATASET 
###############################################################

transform = T.Compose(
    [T.ToTensor(),
     T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 128

traindata = torchvision.datasets.MNIST(root='./data', train=True,
                                       download=True, transform=transform)

testdata = torchvision.datasets.MNIST(root='./data', train=False,
                                      download=True, transform=transform)

# select classes you want to include in your subset
classes = [0, 1, 2, 3, 4]


def get_idx(data, classes=classes):
    """get index of selected classes

    Args:
        data (dataset): pytorch dataset object
        classes (list): list containing int of classes

    Returns:
        idx: list of indices for classes 
    """
    idx = torch.zeros(len(data))
    for i in classes:
        idx += data.targets == i
    return idx


# get indices that correspond to one of the selected classes
train_indices = get_idx(traindata)
test_indices = get_idx(testdata)

print(len(test_indices))
# %%
# get new data 
seq_length = 10
trainDataSeq = image_to_sequence(traindata.data, seq_length)
testDataSeq = image_to_sequence(testdata.data, seq_length)

print(testDataSeq.shape)

# %%
# subset data given classes 
trainDataSeq = trainDataSeq[train_indices]
testDataSeq = testDataSeq[test_indices]

trainTarget = traindata.targets[train_indices]
testTarget = testdata.targets[test_indices]

print(testDataSeq.shape)
print(testTarget.shape)
# %%
# creat tensor datasets for data loader 
train_dataset = data.TensorDataset(trainDataSeq, trainTarget)
test_dataset = data.TensorDataset(testDataSeq, testTarget)

# %%
# data loading 
train_loader = data.DataLoader(train_dataset, batch_size=batch_size,
                               shuffle=True, num_workers=2)
test_loader = data.DataLoader(test_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=2)

# check data loading correctness

# %%
# set input and t param
IN_dim = 28*28
T = seq_length 

# %%
###############################################################################################
##########################          Test function             ###############################
###############################################################################################
# test function
def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    # for data, target in test_loader:
    for i ,(data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        data = data.mean(1).view(-1, IN_dim)
        with torch.no_grad():
            model.eval()
            hidden = model.init_hidden(data.size(0))

            outputs, hidden= model(data, hidden, seq_length) 
           
            output = outputs[-1]
            # output = torch.stack(outputs[-10:]).mean(dim=0)
            
            test_loss += F.nll_loss(output, target, reduction='sum').data.item()
            pred = output.data.max(1, keepdim=True)[1]
        
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        torch.cuda.empty_cache()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
           test_loss, correct, len(test_loader.dataset),
           100. * correct / len(test_loader.dataset)))
    return test_loss, 100. * correct / len(test_loader.dataset)

###############################################################################################
##########################          Train function             ###############################
###############################################################################################

# train function for one epoch
def train(train_loader, n_classes, model, named_params):
    global steps
    global estimate_class_distribution

    train_loss = 0
    total_clf_loss = 0
    total_regularizaton_loss = 0
    total_oracle_loss = 0
    model.train()


    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # data = data.mean(1).view(-1, 32*32)
       
        B = target.size()[0]

        for p in range(K):

            if p==0:
                h = model.init_hidden(data.size(0))
            elif p%omega==0:
                h = tuple(v.detach() for v in h)

            
            o, h,hs = model.network.forward(data, h )

            prob_out = F.softmax(h[-1], dim=1)
            output = F.log_softmax(h[-1], dim=1) 

    
            if p%omega==0 and p>0: 
                optimizer.zero_grad()
                
                # classification loss
                clf_loss = (p+1)/(K)*F.nll_loss(output, target)
                # clf_loss = snr*F.cross_entropy(output, target,reduction='none')
                # clf_loss = torch.mean(clf_loss)

                # energy loss 
                    
                regularizer = get_regularizer_named_params( named_params, _lambda=1.0 )      
                loss = clf_loss  + regularizer 

                loss.backward()

                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    
                    
                optimizer.step()
                post_optimizer_updates( named_params)
            
                train_loss += loss.item()
                total_clf_loss += clf_loss.item()
                total_regularizaton_loss += regularizer #.item()
        
        if batch_idx > 0 and batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tlr: {:.6f}\tLoss: {:.6f}\tOracle: \
                {:.6f}\tClf: {:.6f}\tReg: {:.6f}\tFr: {:.6f}'.format(
                   epoch, batch_idx * batch_size, len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), lr, train_loss / log_interval, 
                   total_oracle_loss / log_interval, 
                   total_clf_loss / log_interval, total_regularizaton_loss / log_interval, model.network.fr/T/log_interval))
            # print(model.network.fr)
            train_loss = 0
            total_clf_loss = 0
            total_regularizaton_loss = 0
            total_oracle_loss = 0
        model.network.fr = 0

# %%
###############################################################
# DEFINE NETWORK
###############################################################
model = one_layer_SeqModel(IN_dim, 784, 10, is_rec=True, is_LTC=False)
model.to(device)
print(model)


# training parameters
K  = T # sequence length
omega = int(T/K)
clip = 1.
log_interval = 100
lr = 1e-3
epoch = 30
n_classes = len(classes)

# %%
# define new loss and optimiser 
total_params = count_parameters(model)

# define optimiser
optimizer = optim.Adamax(model.parameters(), lr=lr, weight_decay=0.0001)
# reduce the learning after 20 epochs by a factor of 10
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)  

# %%
# TODO define network 

# %%
# TODO train and test function 
# %%
# TODO
