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

from scipy.ndimage import median_filter, gaussian_filter

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
"""
create a subset of images for testing, n per class
"""

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))])

batch_size = 200

testdata = torchvision.datasets.MNIST(root='./data', train=False,
                                      download=True, transform=transform)

images = torch.stack([img for img, _ in testdata]).squeeze()
targets = testdata.targets

# get all images as tensors
n_per_class = 60
n_classes = 10

# per n_per_class contain index to that class of images
normal_set_idx = torch.zeros(n_per_class * n_classes).long()
change_set_idx = torch.zeros(n_per_class * n_classes).long()
for i in range(n_classes):
    indices1 = (testdata.targets == i).nonzero(as_tuple=True)[0]
    normal_set_idx[i * n_per_class: (i + 1) * n_per_class] = indices1[: n_per_class]

    indices2 = (testdata.targets != i).nonzero(as_tuple=True)[0]
    change_set_idx[i * n_per_class: (i + 1) * n_per_class] = indices2[: n_per_class]

# data loading
test_subset = torch.utils.data.Subset(testdata, normal_set_idx)
test_loader = torch.utils.data.DataLoader(test_subset, batch_size=batch_size,
                                          shuffle=False, num_workers=2)

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
hidden_dim = [600, 500, 500]
T = 100  # sequence length, reading from the same image time_steps times

dp = 0.4
is_rec = False
# fptt_model_acc = []
# bp_model_acc = []

# for i in range(len(dp)):
# define network
model_wE = SnnNetwork2Layer(IN_dim, hidden_dim, n_classes, is_adapt=adap_neuron, one_to_one=onetoone, dp_rate=dp,
                            is_rec=is_rec)
model_wE.to(device)

# define network
model_woE = SnnNetwork2Layer(IN_dim, hidden_dim, n_classes, is_adapt=adap_neuron, one_to_one=onetoone, dp_rate=dp,
                             is_rec=is_rec)
model_woE.to(device)

# load different models
exp_dir_wE = '/home/lucy/spikingPC/results/Apr-17-2023/fptt_ener0.05_taux2_dt0.5_exptau05_absloss_bias025/'
saved_dict1 = model_result_dict_load(exp_dir_wE + 'onelayer_rec_best.pth.tar')

model_wE.load_state_dict(saved_dict1['state_dict'])

exp_dir_woE = '/home/lucy/spikingPC/results/Apr-17-2023/fptt_ener0.0_taux2_dt0.5_exptau05_absloss_bias025/'
saved_dict2 = model_result_dict_load(exp_dir_woE + 'onelayer_rec_best.pth.tar')

model_woE.load_state_dict(saved_dict2['state_dict'])


# %%
def get_a_s_e(hidden, layer, batch_size, n_samples, T):
    a = get_states(hidden, 2 + layer * 4, hidden_dim[layer], batch_size, num_samples=n_samples, T=T)
    s = get_states(hidden, 0 + layer * 4, hidden_dim[layer], batch_size, num_samples=n_samples, T=T)
    e = np.abs(a-s) 
    return s, a, e

# %%
test_loader_all = torch.utils.data.DataLoader(testdata, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

# %%
# for error
error_E = [np.zeros((len(testdata), T, hidden_dim[i])) for i in range(len(hidden_dim))]
error_nE = [np.zeros((len(testdata), T, hidden_dim[i])) for i in range(len(hidden_dim))]

# for spk rate
spike_E = [np.zeros((len(testdata), T)) for i in range(len(hidden_dim))]
spike_nE = [np.zeros((len(testdata), T)) for i in range(len(hidden_dim))]

# iterate over test dataset
for i, (data, target) in enumerate(test_loader_all):
    data, target = data.to(device), target.to(device)
    data = data.view(-1, model_wE.in_dim)

    with torch.no_grad():
        model_wE.eval()
        model_woE.eval()


        hidden = model_wE.init_hidden(data.size(0))

        _, h_e = model_wE.inference(data, hidden, T)
        _, h_ne = model_woE.inference(data, hidden, T)

        for layer in range(len(hidden_dim)):
            _, _, error_e = get_a_s_e([h_e], layer, batch_size=batch_size, n_samples=batch_size, T=T)
            _, _, error_ne= get_a_s_e([h_ne], layer, batch_size=batch_size, n_samples=batch_size, T=T)

            error_E[layer][i*batch_size:(i+1)*batch_size] = error_e
            error_nE[layer][i*batch_size:(i+1)*batch_size] = error_ne

            spks_e = get_states([h_e], 1+layer*4, hidden_dim[layer], batch_size, T, batch_size)
            spike_E[layer][i*batch_size:(i+1)*batch_size] = spks_e.mean(axis=-1)

            spks_ne = get_states([h_ne], 1+layer*4, hidden_dim[layer], batch_size, T, batch_size)
            spike_nE[layer][i*batch_size:(i+1)*batch_size] = spks_ne.mean(axis=-1)

print(error_E[0].shape)
print(spike_E[0].shape)
# %%
##############################################################
# spk rate by layer comparison 
##############################################################
df_spk = pd.DataFrame(columns=['model', 'layer', 'mean spk', 't'])
for i in range(len(hidden_dim)):
    df = pd.DataFrame(np.vstack((spike_E[i].T, spike_nE[i].T)))
    df['model'] = ['E'] * T + ['nE'] * T 
    df['layer'] = [i] * T * 2
    df['t'] = (np.concatenate((range(T), range(T))))
    df = pd.melt(df, id_vars=['model', 't', 'layer'], value_name='mean spk', value_vars=range(len(testdata)), var_name='img idx')
    
    df_spk = pd.concat([df_spk, df])

df_spk.head()
# %%

# increase size of font
sns.set(font_scale=1.5)
#set sns theme to be white and no grid lines
sns.set_style("whitegrid", {'axes.grid' : False})

fig, ax1 = plt.subplots(figsize=(6, 5))
ax2 = ax1.twiny()

sns.lineplot(df_spk[df_spk['model']=='E'], x='t', hue='layer', y='mean spk', 
            palette=sns.color_palette('Blues', n_colors=len(hidden_dim)), ax=ax1)
ax1.legend(title='E layers', bbox_to_anchor=(1.05, 0.9), loc=2, borderaxespad=0., frameon=False)

sns.lineplot(df_spk[df_spk['model']=='nE'], x='t', y='mean spk', hue='layer',
                palette=sns.color_palette('YlOrBr', n_colors=len(hidden_dim)), ax=ax2)
ax2.xaxis.set_ticks_position('bottom')
ax2.xaxis.set_label_position('bottom')
ax2.legend(title='nE layers', bbox_to_anchor=(1.05, 0.1), loc=3, borderaxespad=0., frameon=False)

sns.despine()
# move legend to outside axes center right
plt.title('mean spk rate per layer')
plt.show()

# %%
# mean spk rate per layer 
sns.barplot(df_spk, x='layer', y='mean spk', hue='model', 
            palette=[(0.1271049596309112, 0.4401845444059977, 0.7074971164936563), 
                     (0.9949711649365629, 0.5974778931180315, 0.15949250288350636)], 
            capsize=0.1)
sns.despine()
plt.legend(title='model', bbox_to_anchor=(1.05, 0.5), loc=2, borderaxespad=0., frameon=False)
plt.show()

# %%
##############################################################
# mean abs(a-s) per neuron comparison
##############################################################

mean_error_E = [np.mean(error_E[i], axis=0) for i in range(len(hidden_dim))]
mean_error_nE = [np.mean(error_nE[i], axis=0) for i in range(len(hidden_dim))]
print(mean_error_E[1].shape)
# %%
# create df with columns model, layer, t, mean error
df_all = pd.DataFrame(columns=['model', 'layer', 't', 'mean error'])
for l in range(len(hidden_dim)):
    df = pd.DataFrame(np.vstack((mean_error_E[l], mean_error_nE[l])))
    df['model'] = ['E'] * T + ['nE'] * T 
    df['layer'] = [l] * T * 2
    df['t'] = (np.concatenate((range(T), range(T))))
    df = pd.melt(df, id_vars=['model', 't', 'layer'], value_name='mean error', value_vars=range(500), var_name='neuron idx')
    
    df_all = pd.concat([df_all, df])

df_all.head()
# sum episolon need to be normalised by mean spk rate 
# %%
# use sns seaborn colorblind palette
fig, ax1 = plt.subplots()
ax2 = ax1.twiny()

sns.lineplot(df_all[df_all['model']=='E'], x='t', y='mean error', hue='layer', 
             palette=sns.color_palette('Blues', n_colors=len(hidden_dim)), ax=ax1)
# change legend title and position to outside upper right
ax1.legend(title='E layers', bbox_to_anchor=(1.05, 0.9), loc=2, borderaxespad=0., frameon=False)

sns.lineplot(df_all[df_all['model']=='nE'], x='t', y='mean error', hue='layer',
                palette=sns.color_palette('YlOrBr', n_colors=len(hidden_dim)), ax=ax2)
ax2.xaxis.set_ticks_position('bottom')
ax2.xaxis.set_label_position('bottom')
# change legend title and position to outside lower right
ax2.legend(title='nE layers', bbox_to_anchor=(1.05, 0.1), loc=3, borderaxespad=0., frameon=False)

# remove top and right spines
sns.despine()

# modify legend labels so that there are more readable and no bounding box
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., frameon=False)

plt.show()
# %%
