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
# TODO define new loss and optimiser 
# %%
# TODO define network 

# %%
# TODO train and test function 
# %%
# TODO 