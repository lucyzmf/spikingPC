# %%

# this script investigates when models trained with bp vs fptt without poisson and dp=0.3, 
# how robust they are in switching to possison encoding and higher drop out rates 

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

dp = np.arange(0.3, 0.9, 0.1)

# fptt_model_acc = []
# bp_model_acc = []

# for i in range(len(dp)):
    # define network
model_fptt = SnnNetwork(IN_dim, hidden_dim, n_classes, is_adapt=adap_neuron, one_to_one=onetoone, dp_rate=0.3)
model_fptt.to(device)

# define network
model_bp = SnnNetwork(IN_dim, hidden_dim, n_classes, is_adapt=adap_neuron, one_to_one=onetoone, dp_rate=0.3)
model_bp.to(device)

# load different models
exp_dir_fptt = '/home/lucy/spikingPC/results/Feb-01-2023/curr18_withener_outmemconstantdecay/'
saved_dict1 = model_result_dict_load(exp_dir_fptt + 'onelayer_rec_best.pth.tar')

model_fptt.load_state_dict(saved_dict1['state_dict'])

exp_dir_bp = '/home/lucy/spikingPC/results/Feb-06-2023/curr18_ener_outmemconstantdecay_bp_20/'
saved_dict2 = model_result_dict_load(exp_dir_bp + 'onelayer_rec_best.pth.tar')

model_bp.load_state_dict(saved_dict2['state_dict'])

    # # # get test acc
    # _, _, _, test_acc_fptt = get_all_analysis_data(model_fptt, test_loader, device, IN_dim, T)
    # _, _, _, test_acc_bp = get_all_analysis_data(model_bp, test_loader, device, IN_dim, T)

    # fptt_model_acc.append(test_acc_fptt)
    # bp_model_acc.append(test_acc_bp)

# %%
# fig = plt.figure()
# plt.plot(dp, fptt_model_acc_poi, label='fptt_poisson_test')
# plt.plot(dp, bp_model_acc_poi, label='bp_poisson_test')
# plt.plot(dp, fptt_model_acc, label='fptt_no_poisson_test')
# plt.plot(dp, bp_model_acc, label='bp_no_poisson_test')
# plt.xlabel('test dp rate')
# plt.ylabel('test acc')
# plt.legend()
# plt.show()

# %%
plt.hist([model_fptt.r_in_rec.rec_w.weight.detach().cpu().numpy().flatten(), \
    model_bp.r_in_rec.rec_w.weight.detach().cpu().numpy().flatten()], label=['fptt', 'bp'])
plt.legend()
plt.title('r rec w')
plt.show()

# %%
plt.hist([model_fptt.r_out_rec.rec_w.weight.detach().cpu().numpy().flatten(), \
    model_bp.r_out_rec.rec_w.weight.detach().cpu().numpy().flatten()], label=['fptt', 'bp'])
plt.legend()
plt.title('p rec w')
plt.show()

# %%
