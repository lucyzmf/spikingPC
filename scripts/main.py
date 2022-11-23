# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
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


# %%
# get new data 
seq_length = 10
trainDataSeq = image_to_sequence(traindata.data, seq_length)
testDataSeq = image_to_sequence(testdata.data, seq_length)

print(testDataSeq.shape)
# %%
# TODO define new loss and optimiser 
# %%
# TODO define network 

# %%
# TODO train and test function 
# %%
# TODO 