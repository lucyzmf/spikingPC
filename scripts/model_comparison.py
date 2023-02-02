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
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import IPython.display as ipd

from tqdm import tqdm

from network_class import *
from utils import *
from FTTP import *

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

batch_size = 200

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
hidden_dim = [10 * num_readout, 784]
T = 20  # sequence length, reading from the same image T times

# define network
model_lowener = SnnNetwork(IN_dim, hidden_dim, n_classes, is_adapt=adap_neuron, one_to_one=onetoone)
model_lowener.to(device)
print(model_lowener)

# define network
model_baseline = SnnNetwork(IN_dim, hidden_dim, n_classes, is_adapt=adap_neuron, one_to_one=onetoone)
model_baseline.to(device)

# define new loss and optimiser
total_params = count_parameters(model_baseline)
print('total param count %i' % total_params)

# %%
# load different models
exp_dir_lowener = '/home/lucy/spikingPC/results/Feb-01-2023/curr18_withenerx2_outmemconstantdecay/'
saved_dict = model_result_dict_load(exp_dir_lowener + 'onelayer_rec_best.pth.tar')

model_lowener.load_state_dict(saved_dict['state_dict'])

exp_dir_baseline = '/home/lucy/spikingPC/results/Feb-01-2023/curr18_withener_outmemconstantdecay/'
saved_dict = model_result_dict_load(exp_dir_baseline + 'onelayer_rec_best.pth.tar')

model_baseline.load_state_dict(saved_dict['state_dict'])

# %%
# get params and put into dict
param_names = []
param_dict = {}
for name, param in model_baseline.named_parameters():
    if param.requires_grad:
        param_names.append(name)
        param_dict[name] = param.detach().cpu().numpy()

print(param_names)

# %%
# compare strength of inhibitory weights from p to r per class
inhibition_strength_per_class = {'class': np.concatenate((np.arange(10), np.arange(10))),
                                 'inhibition': [], 'model type': []}
for i in range(10 * 2):
    if i < 10:
        model = model_baseline
        model_type = 'baseline'
    else:
        model = model_lowener
        model_type = 'low energy'
    w = model.rout2rin.weight[:, num_readout * (i % 10):((i % 10) + 1) * num_readout].detach()
    inhibition_strength_per_class['inhibition'].append(((w < 0) * w).sum().cpu().item())
    inhibition_strength_per_class['model type'].append(model_type)

inhibition_strength_df = pd.DataFrame.from_dict(inhibition_strength_per_class)

fig = plt.figure()
sns.barplot(inhibition_strength_df, x='class', y='inhibition', hue='model type')
plt.show()

# %%
