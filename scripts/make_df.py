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
from scipy.ndimage import median_filter

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

# load data
testdata = torchvision.datasets.MNIST(root='./data', train=False,
                                      download=True, transform=transform)

images = torch.stack([img for img, _ in testdata])
targets = testdata.targets

# get all images as tensors
n_per_class = 100
n_classes = 10
# per n_per_class contain index to that class of images
normal_set_idx = torch.zeros(n_per_class * n_classes)
change_set_idx = torch.zeros(n_per_class * n_classes)
for i in range(n_classes):
    indices1 = (testdata.targets == i).nonzero(as_tuple=True)[0]
    normal_set_idx[i * n_per_class: (i + 1) * n_per_class] = indices1[: n_per_class]

    indices2 = (testdata.targets != i).nonzero(as_tuple=True)[0]
    change_set_idx[i * n_per_class: (i + 1) * n_per_class] = indices2[: n_per_class]


# %%
###############################################################
# inference call
###############################################################
# 2. inference call
# 	if normal,
# 		run inference for T time steps
# 	if change
# 		run inference for 1/4T original stimulus + 3/4T new stim
# 	log hiddens and pred
def get_pred_per_t(log_softmaxs):
    t = len(log_softmaxs)
    preds = []
    for i in range(t):
        pred = log_softmaxs[i].data.max(1, keepdim=True)[1]
        preds.append(pred)
    return torch.transpose(torch.stack(preds), 0, 1)


def inference(model, data_indices: list, is_change: bool, T: int, p: float):
    """
    inference call that given data index and condition and model, returns hiddens and preds for one batch of data
    :param T: total time steps
    :param p: proportion of first stimulus out of T in change condition
    :param model:
    :param data_indices: [normal idx, change idx]
    :param is_change: whether its normal or change condition
    :return: torch tensor of preds per t (T*num samples), targets, hiddens
    """
    if not is_change:
        data, target = images[data_indices[0]].to(device), targets[data_indices[0]]
        data = data.view(-1, model.in_dim)

        with torch.no_grad():
            model.eval()

            h_i = model.init_hidden(data.size(0))

            log_sm, h = model.inference(data, h_i, T)

            preds = get_pred_per_t(log_sm)  # should be shape n * T

        target_byt = targets[data_indices[0]].repeat(1, T)  # shape n*T

    else:
        data1, target1 = images[data_indices[0]].to(device), targets[data_indices[0]]
        data1 = data1.view(-1, model.in_dim)

        data2, target2 = images[data_indices[1]].to(device), targets[data_indices[1]]
        data2 = data2.view(-1, model.in_dim)

        h = []
        preds = []

        with torch.no_grad():
            model.eval()

            h_i = model.init_hidden(data1.size(0))

            log_sm1, h1 = model.inference(data1, h_i, int(T * p))
            preds1 = get_pred_per_t(log_sm1)
            h += h1
            preds += preds1

            log_sm2, h2 = model.inference(data2, h1[-1], (T - int(T * p)))
            preds2 = get_pred_per_t(log_sm2)
            h += h2
            preds += preds2

        preds = torch.stack(preds)

        target_byt1 = targets[data_indices[0]].repeat(1, int(T * p))
        target_byt2 = targets[data_indices[0]].repeat(1, (T - int(T * p)))
        target_byt = torch.stack(target_byt1, target_byt2, dim=1)  # shape n*T

    return preds, target_byt, h


# %%
###############################################################
# make dfs
# #############################################################
# acc df
def make_acc_df(preds_, targets_, model_type: str, condition: str):
    """
    make df with per t model pred and target
    :param preds_:
    :param targets_:
    :param model_type: for labeling
    :param condition: for labeling
    :return:
    """
    T = preds_.size(1)  # get seq len
    n_sample = preds_.size(0)

    df = pd.DataFrame({
        'pred': preds_.flatten(),
        'target': targets_.flatten(),
        't': torch.arange(T).repeat(n_sample, 1).flatten(),
        'model_type': [model_type] * (n_sample * T),
        'condition': [condition] * (n_sample * T)
    })

    return df


def get_error(h, layer, n_samples, b_size, T):
    a = get_states(h, 2 + 4 * layer, hidden_dim[layer], b_size, num_samples=n_samples, T=T)
    s = get_states(h, 0 + 4 * layer, hidden_dim[layer], b_size, num_samples=n_samples, T=T)

    return (a - s) ** 2


# epsilon df

def make_epsilon_df(hidden: list, targets, model_type: str, condition: str, first_stim_p, hidden_dim):
    df_all = pd.DataFrame(columns=['neuron', 'layer', 't', 'epsilon', 'filtered epsi', 'de/e',
                                   'model type', 'condition', 'target'])

    n = targets.size(0)
    T = targets.size(1)

    # iterate over layers
    for layer in range(len(hidden_dim)):
        epsilon = get_error(hidden, layer, n, n, T)  # n * T * neurons
        filtered_epsilon = median_filter(epsilon, size=(1, 3, 1), cval=0, mode='constant')

        baseline_e = filtered_epsilon[:, :int(T * first_stim_p), :].mean(axis=1)  # n*neuron
        baseline_e = np.tile(baseline_e, (1, T, 1))  # n*T*neuron

        deltae_e = (filtered_epsilon / baseline_e) / baseline_e

        df_layer = pd.DataFrame({
            'neuron': np.tile(np.arange(hidden_dim[layer]), (n, T, 1)).flatten(),
            'layer': np.full(n * T * hidden_dim[layer], layer),
            't': np.tile(np.arange(T), (n, 1, hidden_dim[layer])).flatten(),
            'epsilon': epsilon.flatten(),
            'filtered epsi': filtered_epsilon.flatten(),
            'de/e': deltae_e.flatten(),
            'model type': [model_type] * (n * T * hidden_dim[layer]),
            'condition': [condition] * (n * T * hidden_dim[layer]),
            'target': np.tile(targets.numpy(), (1, T, hidden_dim[layer])).flatten()
        })

        df_all = pd.concat([df_all, df_layer])

    return df_all

    # %%


# set input and t param
###############################################################
# DEFINE NETWORK
###############################################################
n_classes = 10
num_readout = 10
adap_neuron = True
onetoone = True

# %%
IN_dim = 784
hidden_dim = [784, 512, 512]
T = 200  # sequence length, reading from the same image time_steps times

dp = 0.4
is_rec = False
# fptt_model_acc = []
# bp_model_acc = []

# for i in range(len(dp)):
# define network
model_wE = SnnNetwork2Layer(IN_dim, hidden_dim, n_classes, is_adapt=adap_neuron, one_to_one=onetoone, dp_rate=0.4,
                            is_rec=is_rec)
model_wE.to(device)

# define network
model_woE = SnnNetwork2Layer(IN_dim, hidden_dim, n_classes, is_adapt=adap_neuron, one_to_one=onetoone, dp_rate=0.4,
                             is_rec=is_rec)
model_woE.to(device)

# load different models
exp_dir_wE = '/home/lucy/spikingPC/results/Mar-28-2023/fptt_ener0.05_taux2_scaledinput05_dt0.5_exptau05/'
saved_dict1 = model_result_dict_load(exp_dir_wE + 'onelayer_rec_best.pth.tar')

model_wE.load_state_dict(saved_dict1['state_dict'])

exp_dir_woE = '/home/lucy/spikingPC/results/Mar-28-2023/fptt_ener0_taux2_scaledinput05_dt0.5_exptau05/'
saved_dict2 = model_result_dict_load(exp_dir_woE + 'onelayer_rec_best.pth.tar')

model_woE.load_state_dict(saved_dict2['state_dict'])

# %%
# function that makes pandas df
conditions = ['normal', 'change']
ischange = [False, True]
models = [model_wE, model_woE]
model_type = ['E', 'w/o E']
T = 150
p = 1 / 3

acc_all = pd.DataFrame()
epsi_all = pd.DataFrame()

for i in range(len(conditions)):
    for j in range(len(models)):
        df_model_acc = pd.DataFrame()
        df_model_epsi = pd.DataFrame()

        for k in range(n_classes):
            preds, targets, hidden = inference(models[j], [normal_set_idx, change_set_idx], ischange[i], T, p)
            df_acc = make_acc_df(preds, targets, model_type[j], conditions[i])
            df_epsi = make_epsilon_df(hidden, targets, model_type[j], conditions[i], first_stim_p=p,
                                      hidden_dim=hidden_dim)

            if iter == 0:
                df_model_acc = df_acc
                df_model_epsi = df_epsi
            else:
                df_model_acc = pd.concat([df_model_acc, df_acc])
                df_model_epsi = pd.concat([df_model_epsi, df_epsi])

        if i == 0 and j == 0:
            acc_all = df_model_acc
            epsi_all = df_model_epsi
        else:
            acc_all = pd.concat([acc_all, df_model_acc])
            epsi_all = pd.concat([epsi_all, df_model_epsi])

acc_all.head()
epsi_all.head()
