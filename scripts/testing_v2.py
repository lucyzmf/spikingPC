# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio

from torch.nn.parameter import Parameter
from torch.nn import init
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision
import numpy as np
import wandb
from datetime import date
import os

import matplotlib.pyplot as plt
import seaborn as sns
import IPython.display as ipd

from tqdm import tqdm

from network_class import *
from utils import *
from FTTP import *

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# set seed
torch.manual_seed(999)

# %%
###############################################################
# IMPORT DATASET 
###############################################################

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))])

batch_size = 100

traindata = torchvision.datasets.MNIST(root='./data', train=True,
                                       download=True, transform=transform)

testdata = torchvision.datasets.MNIST(root='./data', train=False,
                                      download=True, transform=transform)

# data loading 
train_loader = torch.utils.data.DataLoader(traindata, batch_size=batch_size,
                                           shuffle=False, num_workers=2)
test_loader = torch.utils.data.DataLoader(testdata, batch_size=batch_size,
                                          shuffle=False, num_workers=2)

# %%
# set input and t param
###############################################################
# DEFINE NETWORK
###############################################################
# training parameters
T = 20
K = T  # K is num updates per sequence
omega = int(T / K)  # update frequency
clip = 1.
log_interval = 100
lr = 1e-3
epoch = 10
n_classes = 10
num_readout = 10
adap_neuron = True
onetoone = True

# %%
IN_dim = 784
hidden_dim = [256, [10 * num_readout, 784]]
T = 20  # sequence length, reading from the same image T times

# define network
model = SnnNetwork(IN_dim, hidden_dim, n_classes, is_adapt=adap_neuron, one_to_one=onetoone)
model.to(device)
print(model)

# define new loss and optimiser 
total_params = count_parameters(model)
print('total param count %i' % total_params)
# %%

exp_dir = '/home/lucy/spikingPC/results/Jan-20-2023/withener_onetoone/'
saved_dict = model_result_dict_load(exp_dir + 'onelayer_rec_best.pth.tar')

model.load_state_dict(saved_dict['state_dict'])
# %%

fig, axs = plt.subplots(1, 10, figsize=(35, 3))
for i in range(10):
    sns.heatmap(model.rout2rin.weight[:, 10 * i:(i+1)*10].detach().cpu().numpy().mean(axis=1).reshape(28, 28), ax=axs[i])

plt.show()
# %%
