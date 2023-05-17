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
from scipy import stats

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
# get params and put into dict
param_names_wE = []
param_dict_wE = {}
for name, param in model_wE.named_parameters():
    if param.requires_grad:
        param_names_wE.append(name)
        param_dict_wE[name] = param.detach().cpu().numpy()

print(param_names_wE)

param_names_woE = []
param_dict_woE = {}
for name, param in model_woE.named_parameters():
    if param.requires_grad:
        param_names_woE.append(name)
        param_dict_woE[name] = param.detach().cpu().numpy()

print(param_names_woE)

# %%
colors = [(0.1271049596309112, 0.4401845444059977, 0.7074971164936563), 
                     (0.9949711649365629, 0.5974778931180315, 0.15949250288350636)]

# %%
weights = [x for x in param_names_wE if ('weight' in x) and ('bias' not in x) and 
           ('fc_weights' not in x) and ('out2layer2' not in x)]

# plt.style.use('seaborn-v0_8-deep')

fig, axes = plt.subplots(1, len(weights), figsize=(len(weights) * 4, 4))
for i in range(len(weights)):
    axes[i].hist(param_dict_wE[weights[i]].flatten(), label='w E', histtype='step')
    axes[i].hist(param_dict_woE[weights[i]].flatten(), label='w/o E', histtype='step')
    axes[i].legend()
    axes[i].set_title(weights[i])

plt.tight_layout()
plt.show()

# %%
# compute mean and variance per set of weights 
w_mean_wE = [np.mean(param_dict_wE[x]) for x in weights]
w_mean_woE = [np.mean(param_dict_woE[x]) for x in weights]

w_var_wE = [np.var(param_dict_wE[x]) for x in weights]
w_var_woE = [np.var(param_dict_woE[x]) for x in weights]

# create df with columns model, mean, var, weight
df_wE = pd.DataFrame({'model': ['wE'] * len(weights), 'mean': w_mean_wE, 'log var': np.log(w_var_wE), 'weight': weights})
df_woE = pd.DataFrame({'model': ['woE'] * len(weights), 'mean': w_mean_woE, 'log var': np.log(w_var_woE), 'weight': weights})

df_both = pd.concat([df_wE, df_woE])

# plot mean and var
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
sns.barplot(x='weight', y='mean', hue='model', data=df_both, ax=axes[0])
sns.barplot(x='weight', y='log var', hue='model', data=df_both, ax=axes[1])
# rotate x labels
for ax in axes:
    for item in ax.get_xticklabels():
        item.set_rotation(90)
plt.tight_layout()
plt.show()

# %%
# check for all neurons in the network, what proportion of exci vs inhi weights they have for 
# ff and fb weights 

def get_ff_fb_ex_inhi(weights, num_neurons, ax=1):
    ex = np.zeros(num_neurons)
    inh = np.zeros(num_neurons)

    ex = np.mean(weights > 0, axis=ax)
    inh = np.mean(weights < 0, axis=ax)

    assert len(ex) == num_neurons

    return ex, inh

def get_exinhisplit_df(param_dict):
    l1_ff_ex, l1_ff_inh = get_ff_fb_ex_inhi(param_dict['input_fc.weight'], hidden_dim[0])
    l1_fb_ex, l1_fb_inh = get_ff_fb_ex_inhi(param_dict['layer2to1.weight'], hidden_dim[0])

    l2_ff_ex, l2_ff_inh = get_ff_fb_ex_inhi(param_dict['layer1to2.weight'], hidden_dim[1])
    l2_fb_ex, l2_fb_inh = get_ff_fb_ex_inhi(param_dict['layer3to2.weight'], hidden_dim[1])

    l3_ff_ex, l3_ff_inh = get_ff_fb_ex_inhi(param_dict['layer2to3.weight'], hidden_dim[2])
    l3_fb_ex, l3_fb_inh = get_ff_fb_ex_inhi(param_dict['out2layer3.weight'], hidden_dim[2])

    df_ = pd.DataFrame({'layer': ['1'] * hidden_dim[0] + ['2'] * hidden_dim[1] + ['3'] * hidden_dim[2],
                        'ff ex': np.concatenate([l1_ff_ex, l2_ff_ex, l3_ff_ex]),
                        'ff inh': np.concatenate([l1_ff_inh, l2_ff_inh, l3_ff_inh]),
                        'fb ex': np.concatenate([l1_fb_ex, l2_fb_ex, l3_fb_ex]),
                        'fb inh': np.concatenate([l1_fb_inh, l2_fb_inh, l3_fb_inh])})
    return df_

# %%
ex_inhi_split_wE = get_exinhisplit_df(param_dict_wE)
ex_inhi_split_woE = get_exinhisplit_df(param_dict_woE)

ex_inhi_split = pd.concat([ex_inhi_split_wE, ex_inhi_split_woE])
ex_inhi_split['model'] = ['wE'] * len(ex_inhi_split_wE) + ['woE'] * len(ex_inhi_split_woE)

ex_inhi_split.head()

# %%
# plot ex inh split
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
sns.barplot(x='layer', y='ff ex', hue='model', data=ex_inhi_split, ax=axes[0])
sns.barplot(x='layer', y='ff inh', hue='model', data=ex_inhi_split, ax=axes[1])

plt.tight_layout()
plt.show()

# %%
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
sns.barplot(x='layer', y='fb ex', hue='model', data=ex_inhi_split, ax=axes[0])
sns.barplot(x='layer', y='fb inh', hue='model', data=ex_inhi_split, ax=axes[1])

plt.tight_layout()
plt.show()

# %%
sns.scatterplot(x='ff ex', y='fb ex', hue='model', data=ex_inhi_split)
plt.show()

# %%
# compute the mean absoulte weight for each set of weights compared between models
df_weights = pd.DataFrame(columns=['Model', 'weight', 'abs values'])

# weights that connected spiking neurons 
spike_weights = [x for x in weights if ('out' not in x)]
print(spike_weights)

for i in range(len(spike_weights)):
    size = param_dict_wE[spike_weights[i]].size
    d1 = pd.DataFrame({'Model': ['Energy'] * size, 
                                    'weight': [spike_weights[i].replace('.weight', '')] * size,
                                    'abs values': np.abs(param_dict_wE[spike_weights[i]]).flatten()})
    d2 = pd.DataFrame({'Model': ['Control'] * size, 
                                    'weight': [spike_weights[i].replace('.weight', '')] * size, 
                                    'abs values': np.abs(param_dict_woE[spike_weights[i]]).flatten()})
    df_weights = pd.concat([df_weights, d1, d2])

df_weights.head()

# %%
# plot mean abs values
sns.barplot(x='weight', y='abs values', hue='Model', data=df_weights, palette=colors)
# rotate x labels
for item in plt.gca().get_xticklabels():
    item.set_rotation(90)
sns.despine()
plt.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# %%
taus = [x for x in param_names_wE if ('tau' in x)]
# move output_layer tau to the end
taus.append(taus.pop(-4))
print(taus)

layers = ['L1', 'L2', 'L3']
taus_label = [r'$\tau_{adp}$', r'$\tau_{s}$', r'$\tau_{a}$']

fig, axes = plt.subplots(3, 3, figsize=(9, 8), sharey=True)
for i in range(3):
    sns.kdeplot(param_dict_wE[taus[i]].flatten()*2, label='Energy', ax=axes[0, i], 
                 fill=True, alpha=0.5, color=colors[0])
    sns.kdeplot(param_dict_woE[taus[i]].flatten()*2, label='Control', ax=axes[0, i], 
                 fill=True, color=colors[1], alpha=0.5)
    axes[0, i].set_title(layers[0] + ' ' + taus_label[i])
    sns.despine()

    sns.kdeplot(param_dict_wE[taus[i+3]].flatten()*2, label='Energy', ax=axes[1, i], 
                 fill=True, color=colors[0], alpha=0.5)
    sns.kdeplot(param_dict_woE[taus[i+3]].flatten()*2, label='Control', ax=axes[1, i], 
                 fill=True, color=colors[1], alpha=0.5)
    axes[1, i].set_title(layers[1] + ' ' + taus_label[i])
    sns.despine()

    sns.kdeplot(param_dict_wE[taus[i+6]].flatten()*2, label='Energy', ax=axes[2, i], 
                 fill=True, color=colors[0], alpha=0.5)
    sns.kdeplot(param_dict_woE[taus[i+6]].flatten()*2, label='Control', ax=axes[2, i], 
                 fill=True, color=colors[1], alpha=0.5)
    axes[2, i].set_title(layers[2] + ' ' + taus_label[i])
    sns.despine()


plt.tight_layout()
# put legend to right of the figure
plt.legend(title='Model', bbox_to_anchor=(1.05, 2.3), loc='upper left', frameon=False)
plt.show()

# %%
# get mean curr by class to compare between groups
def get_a_s_e(hidden, layer, batch_size, n_samples, T):
    a = get_states(hidden, 2 + layer * 4, hidden_dim[layer], batch_size, num_samples=n_samples, T=T)
    s = get_states(hidden, 0 + layer * 4, hidden_dim[layer], batch_size, num_samples=n_samples, T=T)
    e = np.abs(a-s) 
    return s, a, e


def usi(expected_curr, unexpected_curr, layer_idx, T, ts=None):
    df = pd.DataFrame(np.vstack((expected_curr.mean(axis=0).T, unexpected_curr.mean(axis=0).T)),
                      columns=['t%i' % i for i in range(T)])
    df['neuron idx'] = np.concatenate((np.arange(hidden_dim[layer_idx]), np.arange(hidden_dim[layer_idx])))
    df['condition'] = ['normal seq'] * hidden_dim[layer_idx] + ['stim change seq'] * hidden_dim[layer_idx]

    # compute USI
    if ts is not None:
        df['mean a'] = df.loc[:, 't' + str(ts[0]):'t' + str(ts[1]-1)].mean(axis=1)
        df['var a'] = df.loc[:, 't' + str(ts[0]):'t' + str(ts[1]-1)].var(axis=1)
    else:
        df['mean a'] = df.loc[:, 't10': 't'+str(T-1)].mean(axis=1)
        df['var a'] = df.loc[:, 't10': 't'+str(T-1)].var(axis=1)

    # df_usi = pd.DataFrame({
    #     'neuron idx': np.arange(hidden_dim[layer_idx]),
    #     'usi': (df['mean a'][df['condition'] == 'normal seq'].to_numpy() - df['mean a'][
    #         df['condition'] == 'stim change seq'].to_numpy()) /
    #            np.sqrt(df['var a'][df['condition'] == 'normal seq'].to_numpy())
    # })
    df_usi = pd.DataFrame({
        'neuron idx': np.arange(hidden_dim[layer_idx]),
        'MSD': ((df[df['condition'] == 'normal seq'].loc[:, 't' + str(ts[0]):'t' + str(ts[1]-1)].to_numpy() - df[
            df['condition'] == 'stim change seq'].loc[:, 't' + str(ts[0]):'t' + str(ts[1]-1)].to_numpy())).mean(axis=1) 
    })

    return df, df_usi


def compute_delta(signal):
    """compute delta of signal of a single neuron from multiple samples 

    Args:
        signal (_type_): n * T

    """
    _, T = signal.shape
    delta = signal[:, 1:] - signal[:, :T - 1]
    return delta


def df_single_neuron(expected_curr, unexpected_curr, neuron_idx, delta=None, conditions = ['normal seq', 'stim change seq']):
    steps = T - 1
    if delta:
        # filtered_exp = median_filter(expected_curr[:, :, neuron_idx], (1, 3))
        # filtered_unexp = median_filter(unexpected_curr[:, :, neuron_idx], (1, 3))
        filtered_exp = gaussian_filter(expected_curr[:, :, neuron_idx], (0, 3))
        filtered_unexp = gaussian_filter(unexpected_curr[:, :, neuron_idx], (0, 3))

        delta_exp = compute_delta(filtered_exp)
        delta_unexp = compute_delta(filtered_unexp)
        df_ = pd.DataFrame(np.vstack((delta_exp, delta_unexp)),
                           columns=['t%i' % i for i in range(steps)])
    else:
        df_ = pd.DataFrame(np.vstack((expected_curr[:, :steps, neuron_idx], unexpected_curr[:, :steps, neuron_idx])),
                           columns=['t%i' % i for i in range(steps)])

    df_['condition'] = [conditions[0]] * len(expected_curr) + [conditions[1]] * len(expected_curr)
    df_ = pd.melt(df_, id_vars=['condition'], value_vars=['t%i' % i for i in range(steps)],
                  var_name='t', value_name='volt')
    return df_


# %%
# get analysis data 
added_value = None
batches = 3
hiddens_wE, preds_wE, images_all, _ = get_all_analysis_data(model_wE, test_loader, device, IN_dim, T, batch_no=batches,
                                                            occlusion_p=added_value)
hiddens_woE, preds_woE, _, _ = get_all_analysis_data(model_woE, test_loader, device, IN_dim, T, batch_no=batches,
                                                     occlusion_p=added_value)

n_samples = len(preds_wE)
target_all = testdata.targets.data[normal_set_idx]

# %%
###############################
# convergence speed and evolution of spk rate, epsilon over a normal sequence
###############################
# as the amount of activity is affected given the number of positive pixels in the image
# check whether whether spk rate is associated with pixel values 
spk_l2_E = get_states(hiddens_wE, 5, hidden_dim[1], batch_size, num_samples=n_samples, T=T)
spk_l2_nE = get_states(hiddens_woE, 5, hidden_dim[1], batch_size, num_samples=n_samples, T=T)


def get_value_byclass(spk_log):
    spk_byclass = np.zeros((10, T))
    for i in range(n_classes):
        spk_byclass[i, :] = spk_log[target_all == i].mean(axis=0).mean(axis=1)
    return spk_byclass


l2_spk_byclass = get_value_byclass(spk_l2_E)
for i in range(n_classes):
    plt.plot(l2_spk_byclass[i, 5:], label=str(i))
plt.legend()
plt.show()

# %%
# breakdown of exci vs inhi during inference averaged over T
def exci_inhi_breakdown(spk_log, weights):
    exci = []
    inhi = []

    for t in range(T): 
        topdown = spk_log[:, t, :] @ weights.T
        exci.append(((topdown>0)*topdown).mean())
        inhi.append(-((topdown<0)*topdown).mean())
    
    return exci, inhi 

exci_l2E, inhi_l2E = exci_inhi_breakdown(spk_l2_E, param_dict_wE['layer2to1.weight'])

plt.plot(np.arange(T), exci_l2E, label='exci topdown')
plt.plot(np.arange(T), inhi_l2E, label='inhi topdown')
plt.legend()
plt.show()


# %%
# spk mean at layer 1 
if added_value is not None:
    spk_layer1_wE = get_states(hiddens_wE, 1, hidden_dim[0], batch_size, num_samples=n_samples, T=T)
    spk_layer1_woE = get_states(hiddens_woE, 1, hidden_dim[0], batch_size, num_samples=n_samples, T=T)

    rand_sample = np.random.randint(0, n_samples, size=5)

    s_wE = spk_layer1_wE[rand_sample]
    s_woE = spk_layer1_woE[rand_sample]

    fig, axes = plt.subplots(2, 5, figsize=(17, 6))
    for i in range(5):
        axes[0, i].imshow(s_wE[i].mean(axis=0).reshape(28, 28), vmin=0, vmax=1)
        axes[0, i].set_title('w E mean l1 spk occlusion')

        axes[1, i].imshow(s_woE[i].mean(axis=0).reshape(28, 28), vmin=0, vmax=1)
        axes[1, i].set_title('w/o E mean l1 spk occlusion')
    plt.show()

    spk_layer2_wE = get_states(hiddens_wE, 5, hidden_dim[1], batch_size, num_samples=n_samples, T=T)
    spk_layer2_woE = get_states(hiddens_woE, 5, hidden_dim[1], batch_size, num_samples=n_samples, T=T)

    s_wE2 = spk_layer2_wE[rand_sample]
    s_woE2 = spk_layer2_woE[rand_sample]

    fig, axes = plt.subplots(2, 5, figsize=(17, 6))
    for i in range(5):
        axes[0, i].imshow(s_wE2[i].mean(axis=0).reshape(16, 16), vmin=0, vmax=1)
        axes[0, i].set_title('w E mean l2 spk occlusion')

        axes[1, i].imshow(s_woE2[i].mean(axis=0).reshape(16, 16), vmin=0, vmax=1)
        axes[1, i].set_title('w/o E mean l2 spk occlusion')
    plt.show()

    fig, axes = plt.subplots(2, 5, figsize=(17, 6))
    for i in range(5):
        max = np.max((param_dict_wE['layer2to1.weight'] @ s_wE2[i].mean(axis=0)))
        pos = axes[0, i].imshow((param_dict_wE['layer2to1.weight'] @ s_wE2[i].mean(axis=0)).reshape(28, 28), vmin=-max,
                                vmax=max, cmap='vlag')
        fig.colorbar(pos, ax=axes[0, i])
        axes[0, i].set_title('w E mean l2->l1 occlusion')

        max = np.max((param_dict_woE['layer2to1.weight'] @ s_woE2[i].mean(axis=0)))
        pos = axes[1, i].imshow((param_dict_woE['layer2to1.weight'] @ s_woE2[i].mean(axis=0)).reshape(28, 28),
                                vmin=-max, vmax=max, cmap='vlag')
        fig.colorbar(pos, ax=axes[1, i])
        axes[1, i].set_title('w/o E mean l2->l1 spk occlusion')
    plt.show()


# %%
#######################################################################################################################
###################################             STIM CHANGE EXPERIMENT              ###################################
#######################################################################################################################


soma_l1_wE, a_curr_l1_wE, error_l1_wE = get_a_s_e(hiddens_wE, 0, batch_size, n_samples, T)
soma_l1_woE, a_curr_l1_woE, error_l1_woE = get_a_s_e(hiddens_woE, 0, batch_size, n_samples, T)

soma_l2_wE, a_curr_l2_wE, error_l2_wE = get_a_s_e(hiddens_wE, 1, batch_size, n_samples, T)
soma_l2_woE, a_curr_l2_woE, error_l2_woE = get_a_s_e(hiddens_woE, 1, batch_size, n_samples, T)

soma_l3_wE, a_curr_l3_wE, error_l3_wE = get_a_s_e(hiddens_wE, 2, batch_size, n_samples, T)
soma_l3_woE, a_curr_l3_woE, error_l3_woE = get_a_s_e(hiddens_woE, 2, batch_size, n_samples, T)

# %%
###############################
# get enery signal for stimulus change sequence
###############################
first_stim_t = 50
size_lim = 200

# plot energy consumption in network with two consecutive images
continuous_seq_hiddens_E = []
continuous_seq_hiddens_nE = []

with torch.no_grad():
    model_wE.eval()
    model_woE.eval()

    hidden_i = model_wE.init_hidden(images[normal_set_idx[:size_lim]].view(-1, IN_dim).size(0))

    log_sm_E1, hidden1_E = model_wE.inference(images[normal_set_idx[:size_lim]].view(-1, IN_dim).to(device),
                                              hidden_i, first_stim_t)
    log_sm_nE1, hidden1_nE = model_woE.inference(images[normal_set_idx[:size_lim]].view(-1, IN_dim).to(device),
                                                 hidden_i, first_stim_t)
    continuous_seq_hiddens_E += hidden1_E
    continuous_seq_hiddens_nE += hidden1_nE

    # present second stimulus without reset
    # hidden1[-1] = model.init_hidden(images[sample_image_nos[0], :, :].view(-1, IN_dim).size(0))
    log_sm_E2, hidden2_E = model_wE.inference(images[change_set_idx[:size_lim]].view(-1, IN_dim).to(device),
                                              hidden1_E[-1], (T - first_stim_t))
    log_sm_nE2, hidden2_nE = model_woE.inference(images[change_set_idx[:size_lim]].view(-1, IN_dim).to(device),
                                                 hidden1_nE[-1], (T - first_stim_t))

    continuous_seq_hiddens_E += hidden2_E
    continuous_seq_hiddens_nE += hidden2_nE
torch.cuda.empty_cache()

# get data
conti_soma_l1_wE, conti_a_curr_l1_wE, conti_error_l1_wE = get_a_s_e([continuous_seq_hiddens_E], 0, size_lim, size_lim,
                                                                    T)
conti_soma_l1_woE, conti_a_curr_l1_woE, conti_error_l1_woE = get_a_s_e([continuous_seq_hiddens_nE], 0, size_lim,
                                                                       size_lim, T)

conti_soma_l2_wE, conti_a_curr_l2_wE, conti_error_l2_wE = get_a_s_e([continuous_seq_hiddens_E], 1, size_lim, size_lim,
                                                                    T)
conti_soma_l2_woE, conti_a_curr_l2_woE, conti_error_l2_woE = get_a_s_e([continuous_seq_hiddens_nE], 1, size_lim,
                                                                       size_lim, T)

conti_soma_l3_wE, conti_a_curr_l3_wE, conti_error_l3_wE = get_a_s_e([continuous_seq_hiddens_E], 2, size_lim, size_lim,
                                                                    T)
conti_soma_l3_woE, conti_a_curr_l3_woE, conti_error_l3_woE = get_a_s_e([continuous_seq_hiddens_nE], 2, size_lim,
                                                                       size_lim, T)

# %%
###############################
# usi analysis
###############################
df_l2_a_E, df_usi_l2_a_E = usi(a_curr_l2_wE, conti_a_curr_l2_wE, 1, T, [first_stim_t, T])
df_l2_s_E, df_usi_l2_s_E = usi(soma_l2_wE, conti_soma_l2_wE, 1, T, [first_stim_t, T])

df_l2_a_woE, df_usi_l2_a_woE = usi(a_curr_l2_woE, conti_a_curr_l2_woE, 1, T, [first_stim_t, T])
df_l2_s_woE, df_usi_l2_s_woE = usi(soma_l2_woE, conti_soma_l2_woE, 1, T, [first_stim_t, T])

# %%
df_usi_compare_a = pd.concat([df_usi_l2_s_E, df_usi_l2_s_woE])
df_usi_compare_a['model type'] = ['E'] * hidden_dim[1] + ['w/o E'] * hidden_dim[1]

sns.histplot(df_usi_compare_a, x='usi', hue='model type', element="step", stat="density")
plt.title('compare usi of change exp layer2 by model type (soma)')
plt.show()

# %%
high_usi_index = df_usi_l2_s_E.sort_values(by='usi')['neuron idx'][0]

# %%

df_single_s = df_single_neuron(soma_l2_wE[:size_lim], conti_soma_l2_wE[:size_lim], high_usi_index)
sns.lineplot(df_single_s, x='t', y='volt', hue='condition')
plt.title('high usi neuron soma voltage during seq')
plt.show()

# %%
low_usi_index = df_usi_l2_s_E.sort_values(by='usi')['neuron idx'][256]

df_single_s = df_single_neuron(soma_l2_wE[:size_lim], conti_soma_l2_wE[:size_lim], low_usi_index)
sns.lineplot(df_single_s, x='t', y='volt', hue='condition')
plt.title('low usi neuron soma voltage during seq')
plt.show()

# %%
#######################################################################################################################
###################################                 BU TD MISMATCH EXP              ###################################
#######################################################################################################################
match_dig = 1
# mismatch_dig = 4
mismatch_dig = np.delete(np.arange(0, 10), match_dig)

sample_size = 50
zeros = images[targets == match_dig][:sample_size].to(device)

no_inputs = torch.zeros((zeros.size(0), IN_dim)).to(device)

blank_t = 25
match_t = 50
end_seq_t = T - blank_t - match_t


def match_mismatch_ex(match_condition):
    h_E = []
    h_nE = []

    clamp_class = match_dig if match_condition else np.random.choice(mismatch_dig)
    print('clamp class: ', clamp_class)

    with torch.no_grad():
        model_wE.eval()
        model_woE.eval()

        hidden_i = model_wE.init_hidden(zeros.size(0))

        _, h1_E = model_wE.inference(no_inputs, hidden_i, blank_t)
        _, h1_nE = model_woE.inference(no_inputs, hidden_i, blank_t)
        h_E += h1_E
        h_nE += h1_nE

        _, h2_E = model_wE.clamped_generate(clamp_class, zeros.view(-1, IN_dim), h1_E[-1], match_t,
                                            clamp_value=1, batch=True)
        _, h2_nE = model_woE.clamped_generate(clamp_class, zeros.view(-1, IN_dim), h1_nE[-1], match_t,
                                              clamp_value=1, batch=True)
        h_E += h2_E
        h_nE += h2_nE

        _, h3_E = model_wE.inference(no_inputs, h2_E[-1], end_seq_t)
        _, h3_nE = model_woE.inference(no_inputs, h2_nE[-1], end_seq_t)
        h_E += h3_E
        h_nE += h3_nE

    return h_E, h_nE


# %%
h_match_E, h_match_nE = match_mismatch_ex(True)
h_mismatch_E, h_mismatch_nE = match_mismatch_ex(False)

# %%
# get data from match
match_soma_l1_wE, match_a_curr_l1_wE, match_error_l1_wE = get_a_s_e([h_match_E], 0, sample_size, sample_size, T)
match_soma_l1_woE, match_a_curr_l1_woE, match_error_l1_woE = get_a_s_e([h_match_nE], 0, sample_size, sample_size, T)

match_soma_l2_wE, match_a_curr_l2_wE, match_error_l2_wE = get_a_s_e([h_match_E], 1, sample_size, sample_size, T)
match_soma_l2_woE, match_a_curr_l2_woE, match_error_l2_woE = get_a_s_e([h_match_nE], 1, sample_size, sample_size, T)

match_soma_l3_wE, match_a_curr_l3_wE, match_error_l3_wE = get_a_s_e([h_match_E], 2, sample_size, sample_size, T)
match_soma_l3_woE, match_a_curr_l3_woE, match_error_l3_woE = get_a_s_e([h_match_nE], 2, sample_size, sample_size, T)

# %%
# get data from mismatch
mis_soma_l1_woE, mis_a_curr_l1_woE, mis_error_l1_woE = get_a_s_e([h_mismatch_nE], 0, sample_size, sample_size, T)
mis_soma_l1_wE, mis_a_curr_l1_wE, mis_error_l1_wE = get_a_s_e([h_mismatch_E], 0, sample_size, sample_size, T)

mis_soma_l2_wE, mis_a_curr_l2_wE, mis_error_l2_wE = get_a_s_e([h_mismatch_E], 1, sample_size, sample_size, T)
mis_soma_l2_woE, mis_a_curr_l2_woE, mis_error_l2_woE = get_a_s_e([h_mismatch_nE], 1, sample_size, sample_size, T)

mis_soma_l3_wE, mis_a_curr_l3_wE, mis_error_l3_wE = get_a_s_e([h_mismatch_E], 2, sample_size, sample_size, T)
mis_soma_l3_woE, mis_a_curr_l3_woE, mis_error_l3_woE = get_a_s_e([h_mismatch_nE], 2, sample_size, sample_size, T)

# %%
df_l2_a_matchexp_E, df_usi_l2_a_matchexp_E = usi(match_a_curr_l2_wE, mis_a_curr_l2_wE, 1, T, ts=[blank_t, blank_t+match_t])
df_l2_s_matchexp_E, df_usi_l2_s_matchexp_E = usi(match_soma_l2_wE, mis_soma_l2_wE, 1, T, ts=[blank_t, blank_t+match_t])

df_l2_a_matchexp_woE, df_usi_l2_a_matchexp_woE = usi(match_a_curr_l2_woE, mis_a_curr_l2_woE, 1, T, ts=[blank_t, blank_t+match_t])
df_l2_s_matchexp_woE, df_usi_l2_s_matchexp_woE = usi(match_soma_l2_woE, mis_soma_l2_woE, 1, T, ts=[blank_t, blank_t+match_t])

# %%
df_usi_compare_a = pd.concat([df_usi_l2_a_matchexp_E, df_usi_l2_a_matchexp_woE])
df_usi_compare_a['model type'] = ['E'] * hidden_dim[1] + ['w/o E'] * hidden_dim[1]

print(stats.ks_2samp(df_usi_l2_a_matchexp_E['MSD'], df_usi_l2_a_matchexp_woE['MSD']))

# increase size of font
sns.set(font_scale=1.5)
#set sns theme to be white and no grid lines
sns.set_style("whitegrid", {'axes.grid' : False})

colors = [(0.1271049596309112, 0.4401845444059977, 0.7074971164936563), 
                     (0.9949711649365629, 0.5974778931180315, 0.15949250288350636)]

fig = plt.figure(figsize=(7, 4))
sns.histplot(df_usi_compare_a, x='MSD', hue='model type', bins=10, stat='percent', palette=colors)
plt.title('L2 apical tuft')
# remove legend
plt.legend([],[], frameon=False)
sns.despine()
plt.show()

# %%
df_usi_compare_s = pd.concat([df_usi_l2_s_matchexp_E, df_usi_l2_s_matchexp_woE])
df_usi_compare_s['model type'] = ['E'] * hidden_dim[1] + ['w/o E'] * hidden_dim[1]

print(stats.ks_2samp(df_usi_l2_s_matchexp_E['MSD'], df_usi_l2_s_matchexp_woE['MSD']))

fig = plt.figure(figsize=(7, 4))
sns.histplot(df_usi_compare_s, x='MSD', hue='model type', bins=10, stat='percent', palette=colors)
plt.title('L2 soma')
sns.despine()
plt.show()

# %%
# compare spk rate in these conditions
spk_l2_E_match = get_states([h_match_E], 5, hidden_dim[1], sample_size, T, sample_size)
spk_l2_E_mis = get_states([h_mismatch_E], 5, hidden_dim[1], sample_size, T, sample_size)

spk_l2_nE_match = get_states([h_match_nE], 5, hidden_dim[1], sample_size, T, sample_size)
spk_l2_nE_mis = get_states([h_mismatch_nE], 5, hidden_dim[1], sample_size, T, sample_size)


def make_df_matchexp_spk(spk_match, spk_mismatch):
    n, t, _ = spk_match.shape
    df_ = pd.DataFrame(np.vstack((spk_match.mean(axis=-1), spk_mismatch.mean(axis=-1))),
                       columns=['t%i' % i for i in range(t)])
    df_['condition'] = ['match'] * n + ['mismatch'] * n
    df_ = pd.melt(df_, id_vars=['condition'], value_vars=['t%i' % i for i in range(t)],
                  var_name='t', value_name='spk rate')
    return df_


spk_mismatch_l2_E = make_df_matchexp_spk(spk_l2_E_match, spk_l2_E_mis)

sns.lineplot(spk_mismatch_l2_E, x='t', y='spk rate', hue='condition')
sns.despine()
plt.show()

# %%
# check distribution of delta spk rate between match and mismatch for both models during stim onset
delta_spk_E = spk_l2_E_match[:, 25:75, :].mean(axis=0).mean(axis=0) - spk_l2_E_mis[:, 25:75, :].mean(axis=0).mean(axis=0)
delta_spk_nE = spk_l2_nE_match[:, 25:75, :].mean(axis=0).mean(axis=0)- spk_l2_nE_mis[:, 25:75, :].mean(axis=0).mean(axis=0)

print(stats.ks_2samp(delta_spk_E, delta_spk_nE))

df = pd.DataFrame({
    r'$\delta R$': np.concatenate((delta_spk_E, delta_spk_nE)), 
    'model': ['E'] * len(delta_spk_E) + ['nE'] * len(delta_spk_nE)
})

sns.set(font_scale=1.5)
sns.set_style("whitegrid", {'axes.grid' : False})

fig = plt.figure(figsize=(7, 4))
sns.histplot(df, x=r'$\delta R$', hue='model', stat='percent', bins=10, palette=colors)
sns.despine()
plt.title('L2 ' + r'$\delta R$')
plt.legend(frameon=False)
plt.show()


# %%
# plot grid of example neurons responding similar or different to exp vs unexp stimuli 
# create list for [0, low, high] mean diff index
def get_indices(df_):
    """get index of high, 0, low mean diff neurons"""
    high_idx = df_.sort_values(by='mean diff')['neuron idx'].to_list()[-1]
    zero_idx = df_.sort_values(by='mean diff')['neuron idx'].to_list()[int(len(df_)/2)]
    low_idx = df_.sort_values(by='mean diff')['neuron idx'].to_list()[0]
    return [low_idx, zero_idx, high_idx]

apical_indices = get_indices(df_usi_l2_a_matchexp_E)
soma_indices = get_indices(df_usi_l2_s_matchexp_E)
# soma_indices = apical_indices

print(apical_indices)

# %%
from matplotlib.patches import Patch

two_blues = [(0.1271049596309112, 0.4401845444059977, 0.7074971164936563), 
                     (0.5356862745098039, 0.746082276047674, 0.8642522106881968)]

def plot_single_neuron_ax(idx, ax, match_curr, mismatch_curr):
    df = df_single_neuron(match_curr[:sample_size], mismatch_curr[:sample_size], idx, conditions=['match', 'mismatch'])
    sns.lineplot(df, x='t', y='volt', hue='condition', ax=ax, palette=two_blues)

    range = ax.get_ylim()

    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.fill_between(np.arange(T), range[0], range[1], where=np.logical_and(np.arange(T)>blank_t, (np.arange(T)<(blank_t+match_t))),
    alpha=0.25, facecolor='grey')

    ax.set_xticks([])
    ax.set_xlabel('')
    # ax.set_title('idx %i' %idx)
    ax.get_legend().remove()


sns.set(font_scale=1.)
sns.set_style("whitegrid", {'axes.grid' : False})

fig, axs = plt.subplots(2, 3, figsize=(8, 3), sharex=True)
for i in range(len(apical_indices)):
    plot_single_neuron_ax(apical_indices[i], axs[0][i], match_a_curr_l2_wE, mis_a_curr_l2_wE)
    axs[0, i].set_title('MSD %.3f' %df_usi_l2_a_matchexp_E[df_usi_l2_a_matchexp_E['neuron idx']==apical_indices[i]]['mean diff'].values[0], 
                        fontsize=10)

    plot_single_neuron_ax(soma_indices[i], axs[1][i], match_soma_l2_wE, mis_soma_l2_wE)
    axs[1, i].set_title('MSD %.3f' %df_usi_l2_s_matchexp_E[df_usi_l2_s_matchexp_E['neuron idx']==soma_indices[i]]['mean diff'].values[0], 
                        fontsize=10)
    # annotate at the bottom of the ax
    axs[1, i].annotate('t', xy=(0.5, -0.2), xycoords='axes fraction', ha='center', va='center')


# axs[0, 1].legend().set_visible(True)
# realign legend to have both labels in one line and position center top outside of plot
# axs[0, 1].legend(borderaxespad=0., ncols=2, loc='upper center', frameon=False)
handles, labels = axs[0, 2].get_legend_handles_labels()
handles.append(Patch(facecolor='grey', alpha=0.25))
labels.append("stim onset")
fig.legend(handles, labels, loc='upper center', frameon=False, ncol=3, bbox_to_anchor=(0.55, 1.08))

# remove uncessary y labels 
for i in range(3):
    axs[1, i].set_xlabel('')

axs[0, 1].set_ylabel('')
axs[1, 1].set_ylabel('')

axs[0, 2].set_ylabel('')
axs[1, 2].set_ylabel('')

# label columns
rows = ['L2-apical', 'L2-soma']

pad = 20

for ax, row in zip(axs[:, 0], rows):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad / 2, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')

plt.tight_layout()
plt.show()

# %%
# compare error 

def abs_error(expected_curr, unexpected_curr, layer_idx, T, ts=None):
    df = pd.DataFrame(np.vstack((expected_curr.mean(axis=0).T, unexpected_curr.mean(axis=0).T)),
                      columns=['t%i' % i for i in range(T)])
    df['neuron idx'] = np.concatenate((np.arange(hidden_dim[layer_idx]), np.arange(hidden_dim[layer_idx])))
    df['condition'] = ['match'] * hidden_dim[layer_idx] + ['mismatch'] * hidden_dim[layer_idx]
    df['mean abs error'] = np.concatenate((expected_curr.mean(axis=0).mean(axis=0), 
                                  unexpected_curr.mean(axis=0).mean(axis=0)))

    # compute mean abs error
    df_usi = pd.DataFrame({
        'neuron idx': np.arange(hidden_dim[layer_idx]),
        'mean abs diff': np.abs(((df[df['condition'] == 'match'].loc[:, 't' + str(ts[0]):'t' + str(ts[1]-1)].to_numpy() - df[
            df['condition'] == 'mismatch'].loc[:, 't' + str(ts[0]):'t' + str(ts[1]-1)].to_numpy()))).mean(axis=1) 
    })

    return df, df_usi

df_l2_e_matchexp_E, df_usi_l2_e_matchexp_E = abs_error(match_error_l2_wE, mis_error_l2_wE, 1, T, ts=[blank_t, blank_t+match_t])
df_l2_e_matchexp_woE, df_usi_l2_e_matchexp_woE = abs_error(match_error_l2_woE, mis_error_l2_woE, 1, T, ts=[blank_t, blank_t+match_t])

# %%
# plot distribution of errors for each model 
sns.histplot(df_l2_e_matchexp_E, x='mean abs error', hue='condition', bins=10, stat='percent', palette=colors)
plt.show()

# %%
df_l2_e_matchexp_E = df_l2_e_matchexp_E.melt(id_vars=['neuron idx', 'condition', 'mean abs error'],
                                                value_vars=['t%i' % i for i in range(T)],
                                                var_name='t', value_name='abs error')
sns.lineplot(df_l2_e_matchexp_E, x='t', y='abs error', hue='condition', palette=colors)
plt.show()

# %%
df_usi_compare_a = pd.concat([df_usi_l2_e_matchexp_E, df_usi_l2_e_matchexp_woE])
df_usi_compare_a['model type'] = ['E'] * hidden_dim[1] + ['w/o E'] * hidden_dim[1]

print(stats.ks_2samp(df_usi_l2_e_matchexp_E['mean abs diff'], df_usi_l2_e_matchexp_woE['mean abs diff']))

sns.histplot(df_usi_compare_a, x='mean abs diff', hue='model type', bins=11, palette=colors)
plt.title('compare usi of match mismatch layer2 by model type (error)')
sns.despine()
plt.show()

# %%
# compute the delta spk rate of layers for match and mismatch 
def delta_spk_rate(spk_match, spk_mismatch, ts):
    """compute delta spk rate of match and mismatch"""
    delta_spk = spk_match[:, ts[0]:ts[1], :].mean(axis=0).mean(axis=0) - spk_mismatch[:, ts[0]:ts[1], :].mean(axis=0).mean(axis=0)
    return delta_spk

spk_l1_E_match = get_states([h_match_E], 1, hidden_dim[0], sample_size, T, sample_size)
spk_l1_E_mis = get_states([h_mismatch_E], 1, hidden_dim[0], sample_size, T, sample_size)

spk_l1_nE_match = get_states([h_match_nE], 1, hidden_dim[0], sample_size, T, sample_size)
spk_l1_nE_mis = get_states([h_mismatch_nE], 1, hidden_dim[0], sample_size, T, sample_size)

spk_l3_E_match = get_states([h_match_E], 9, hidden_dim[2], sample_size, T, sample_size)
spk_l3_E_mis = get_states([h_mismatch_E], 9, hidden_dim[2], sample_size, T, sample_size)

spk_l3_nE_match = get_states([h_match_nE], 9, hidden_dim[2], sample_size, T, sample_size)
spk_l3_nE_mis = get_states([h_mismatch_nE], 9, hidden_dim[2], sample_size, T, sample_size)

delta_spk_l1_E = delta_spk_rate(spk_l1_E_match, spk_l1_E_mis, [blank_t, blank_t+match_t])
delta_spk_l1_nE = delta_spk_rate(spk_l1_nE_match, spk_l1_nE_mis, [blank_t, blank_t+match_t])

delta_spk_l2_E = delta_spk_rate(spk_l2_E_match, spk_l2_E_mis, [blank_t, blank_t+match_t])
delta_spk_l2_nE = delta_spk_rate(spk_l2_nE_match, spk_l2_nE_mis, [blank_t, blank_t+match_t])

delta_spk_l3_E = delta_spk_rate(spk_l3_E_match, spk_l3_E_mis, [blank_t, blank_t+match_t])
delta_spk_l3_nE = delta_spk_rate(spk_l3_nE_match, spk_l3_nE_mis, [blank_t, blank_t+match_t])

# %%
# create df with delta spk rate for each layer and model
df_delta_spk = pd.DataFrame({
    r'$\delta R$': np.concatenate((delta_spk_l1_E, delta_spk_l1_nE, delta_spk_l2_E, delta_spk_l2_nE, delta_spk_l3_E, delta_spk_l3_nE)),
    'Layer': ['1'] * 2 * hidden_dim[0] + ['2'] * 2 * hidden_dim[1] + ['3'] * 2 * hidden_dim[2],
    'Model': ['Energy'] * hidden_dim[0] + ['Control'] * hidden_dim[0] + \
        ['Energy'] * hidden_dim[1] + ['Control'] * hidden_dim[1] + \
        ['Energy'] * hidden_dim[2] + ['Control'] * hidden_dim[2]
})

sns.set(font_scale=1.5)
sns.set_style("whitegrid", {'axes.grid' : False})

fig = plt.figure(figsize=(7, 4))
sns.barplot(data=df_delta_spk, x='Layer', y=r'$\delta R$', hue='Model', palette=colors, alpha=0.5)
plt.title(r'$\delta R$ ' + 'match vs mismatch')
plt.legend(frameon=False)
sns.despine()
plt.show()


# %%


# %%
###############################
# see how adding pix value impact firing rate 
###############################
added_value = np.arange(-2, 2.1, 1)

test_loader_all = torch.utils.data.DataLoader(testdata, batch_size=batch_size, shuffle=False, num_workers=2)

def add_pixel_test(added_v, layer=0):
    spk_E = np.zeros((len(added_v), len(testdata), hidden_dim[layer]))
    spk_nE = np.zeros((len(added_v), len(testdata), hidden_dim[layer]))

    acc_E = np.zeros(len(added_v))
    acc_nE = np.zeros(len(added_v))

    in_E = np.zeros((len(added_v), len(testdata)))
    in_nE = np.zeros((len(added_v), len(testdata)))

    for step in range(len(added_v)):
        print(added_v[step])
        correct1 = 0
        correct2 = 0

        for i, (data, target) in enumerate(test_loader_all):
            data += added_v[step]

            data, target = data.to(device), target.to(device)
            data = data.view(-1, model_wE.in_dim)
 
            with torch.no_grad():
                model_wE.eval()
                model_woE.eval()

                in_E[step, i*batch_size:(i+1)*batch_size] = model_wE.input_fc(data).mean(dim=1).detach().cpu().numpy()
                in_nE[step, i*batch_size:(i+1)*batch_size] = model_woE.input_fc(data).mean(dim=1).detach().cpu().numpy()

                hidden = model_wE.init_hidden(data.size(0))

                o1, he = model_wE.inference(data, hidden, T)
                o2, hne = model_woE.inference(data, hidden, T)

                pred1 = o1[-1].data.max(1, keepdim=True)[1]
                pred2 = o2[-1].data.max(1, keepdim=True)[1]

                correct1 += pred1.eq(target.data.view_as(pred1)).sum()
                correct2 += pred2.eq(target.data.view_as(pred2)).sum()
            
            spk_e = get_states([he], 1 + layer * 4, hidden_dim[layer], batch_size, num_samples=batch_size, T=T)
            spk_ne = get_states([hne], 1 + layer * 4, hidden_dim[layer], batch_size, num_samples=batch_size, T=T)

            spk_E[step, i*batch_size:(i+1)*batch_size] = spk_e.mean(axis=1)
            spk_nE[step, i*batch_size:(i+1)*batch_size] = spk_ne.mean(axis=1)

        acc_E[step] = correct1.cpu().numpy() / len(testdata)
        acc_nE[step] = correct2.cpu().numpy() / len(testdata)

    print(spk_E.shape)

    return spk_E, spk_nE, acc_E, acc_nE, in_E, in_nE

layer = 0
spk_addpix_E, spk_addpix_nE, acc_E, acc_nE, input_E, input_nE = add_pixel_test(added_value, layer=layer)

print(spk_addpix_E.shape)
print(acc_E.shape)
print(input_E.shape)
# %%
# create dataframe containing colums for mean input, spike rate per neuron avg across all samples, neuron index 
def make_df_addpix(spk, input, added_value):
    """
    each value in spk is spk rate of a neuron for a sample
    input is avg current input into each neuron per sample"""
    df = pd.DataFrame(spk.mean(axis=1))
    df['added value'] = added_value
    df['avg input'] = input.mean(axis=1)
    df[r'$\delta$ current'] = df['avg input'] - input.mean(axis=1)[int(len(added_value)/2)]
    df = pd.melt(df, id_vars=['added value', 'avg input', r'$\delta$ current'], value_vars=range(500), var_name='neuron idx', value_name='spk rate')
    return df

# %%
df_E = make_df_addpix(spk_addpix_E, input_E, added_value)
df_nE = make_df_addpix(spk_addpix_nE, input_nE, added_value)

df_all_addedpix = pd.concat([df_E, df_nE])
df_all_addedpix['model'] = ['Energy'] * len(df_E) + ['Control'] * len(df_nE)

df_all_addedpix[:10]

# %%
# plot lineplot, x axis is avg input, y value is spk rate, hue is model 
sns.set(font_scale=1.5)
sns.set_style("whitegrid", {'axes.grid' : False})

colors = [(0.1271049596309112, 0.4401845444059977, 0.7074971164936563), 
                     (0.9949711649365629, 0.5974778931180315, 0.15949250288350636)]


# %%
# plot spk rate against normalised avg input for each model 
def normalise_input(df):
    """df of one model"""
    df['Norm avg current'] = (df['avg input'] - df['avg input'].min()) / (df['avg input'].max() - df['avg input'].min())
    return df

df_all_addedpix_norm_E = normalise_input(df_all_addedpix[df_all_addedpix['model']=='Energy'])
df_all_addedpix_norm_nE = normalise_input(df_all_addedpix[df_all_addedpix['model']=='Control'])

df_all_addedpix = pd.concat([df_all_addedpix_norm_E, df_all_addedpix_norm_nE])

fig = plt.figure(figsize=(6, 5))
sns.lineplot(data=df_all_addedpix, x='Norm avg current', y='spk rate', hue='model', palette=colors)
sns.despine()
plt.legend(frameon=False)
plt.title('L' + str(layer+1))
plt.show()

# %%
# plot spk rate against avg input for each model
fig = plt.figure(figsize=(6, 5))
sns.lineplot(data=df_all_addedpix, x='avg input', y='spk rate', hue='model', palette=colors)
sns.despine()
plt.legend(frameon=False)
plt.title('L' + str(layer+1))
plt.show()

# %%
# appendix plot
df = pd.DataFrame({
    'model': ['Energy'] * len(added_value) + ['Control'] * len(added_value),
    'acc': np.concatenate((acc_E, acc_nE)),
    'added value': np.round(np.concatenate((added_value, added_value)), decimals=1)
})

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
sns.barplot(df, x='added value', y='acc', hue='model', palette=colors, ax=axes[0])
axes[0].set_title('acc with values added to input')
# set legend to be invisible
axes[0].legend([],[], frameon=False)
axes[0].set_xticklabels([int(i) for i in added_value])
sns.despine()

sns.lineplot(data=df_all_addedpix, x='added value', 
            y=r'$\delta$ current', hue='model', palette=colors)
axes[1].set_title(r'$\delta$ current'+' into L' + str(layer+1))
axes[1].legend(frameon=False)
plt.tight_layout()
plt.show()

# %%
mean_rate_by_neuron = spk_addpix_E.mean(axis=1)
# check proportion of saturating neuron 
satur_p = [(mean_rate_by_neuron[i] == 1).mean() for i in range(len(added_value))]
plt.plot(added_value, satur_p)
plt.xlabel('added value')
plt.ylabel('p of saturated neurons')

# %%
corr = [np.corrcoef(added_value, mean_rate_by_neuron[:, i])[0, 1] for i in range(mean_rate_by_neuron.shape[1])]
plt.hist(corr)
plt.title('distribution of r between spk rate and added value')
plt.show()

# %%
colors_multi = ['b', 'r', 'g', 'purple', colors[1]]

fig = plt.figure(figsize=(7, 5))
random_sample = np.random.choice(mean_rate_by_neuron.shape[1], 20)
for i in range(len(added_value)):
    sns.barplot(x=np.arange(len(random_sample)), y=mean_rate_by_neuron[i, random_sample], label=str(added_value[i]), color=colors_multi[i])

plt.legend(title='added value', frameon=False, bbox_to_anchor=(1, 1))
plt.xlabel('randomly sampled neuron')
plt.ylabel('spk rate')
plt.title('spk rate per neuron at different input condition')
sns.despine()
plt.show()





# %%























# %%
###############################
# verify variance distribution
###############################
def variance(data):
    x_mean = data.mean()
    deviation = [(x - x_mean) ** 2 for x in data]
    return sum(deviation) / len(deviation)


var_E_l1_conti = [variance(conti_error_l1_E.mean(axis=0)[:, i]) for i in range(784)]
var_nE_l1_conti = [variance(conti_error_l1_nE.mean(axis=0)[:, i]) for i in range(784)]

sns.histplot(np.sqrt(var_E_l1_conti), element='step', label='w E', binwidth=0.1, kde=True)
sns.histplot(np.sqrt(var_nE_l1_conti), element='step', label='w/o E', binwidth=0.1, kde=True)
plt.title('std of epislon per layer1 neuron')
plt.legend()
plt.show()

# %%
var_E_l2_conti = [variance(conti_error_l2_E.mean(axis=0)[:, i]) for i in range(hidden_dim[1])]
var_nE_l2_conti = [variance(conti_error_l2_nE.mean(axis=0)[:, i]) for i in range(hidden_dim[1])]

sns.histplot(np.sqrt(var_E_l2_conti), element='step', label='w E', binwidth=0.1, kde=True)
sns.histplot(np.sqrt(var_nE_l2_conti), element='step', label='w/o E', binwidth=0.1, kde=True)
plt.title('std of epislon per layer2 neuron')
plt.legend()
plt.show()

# %%
##############################################################
# find index for low and high error variance neurons
##############################################################

###############################
# layer1
###############################
# use mdeian to select neurons
high_var_l1_E_map = var_E_l1_conti > np.quantile(var_E_l1_conti, 0.5)
plt.imshow(high_var_l1_E_map.reshape(28, 28))

# %%
high_var_l1_nE_map = var_nE_l1_conti > np.quantile(var_nE_l1_conti, 0.5)
plt.imshow(high_var_l1_nE_map.reshape(28, 28))


# %%
# model with E

def create_error_df(error_norm, error_conti, high_var_map, layer_index, class_filter):
    df = pd.DataFrame(np.vstack((error_norm[target_all == class_filter].mean(axis=0), error_conti.mean(axis=0))), \
                      columns=['neuron%i' % i for i in range(hidden_dim[layer_index])])
    df['t'] = np.concatenate((np.arange(T), np.arange(T)))
    df['condition'] = ['normal seq'] * T + ['stim change seq'] * T
    df = pd.melt(df, id_vars=['t', 'condition'], value_vars=['neuron%i' % i for i in range(hidden_dim[layer_index])],
                 var_name='neuron index', value_name='epsilon')
    df['high var'] = 0
    for i in range(hidden_dim[layer_index]):
        df['high var'][df['neuron index'] == 'neuron%i' % i] = high_var_map[i]

    return df


# %%
df_E_l1 = create_error_df(error_l1_wE, conti_error_l1_E, high_var_l1_E_map, 0, first_dig)
df_nE_l1 = create_error_df(error_l1_woE, conti_error_l1_nE, high_var_l1_nE_map, 0, first_dig)


# %%
# plot
def plot_error_sig(df_e, df_ne, layer_index, share_y):
    fig, axes = plt.subplots(1, 2, sharey=share_y, figsize=(7, 3))
    sns.lineplot(df_e, x='t', y='epsilon', hue='condition', style='high var',
                 estimator='mean', errorbar='ci', ax=axes[0])
    axes[0].set_title('w E model layer%i' % layer_index)
    axes[0].axvline(x=first_stim_t, color='black', linestyle='dotted')
    axes[0].get_legend().remove()

    sns.lineplot(df_ne, x='t', y='epsilon', hue='condition', style='high var',
                 estimator='mean', errorbar='ci', ax=axes[1])
    axes[1].set_title('w/o E model layer%i' % layer_index)
    axes[1].axvline(x=first_stim_t, color='black', label='change onset', linestyle='dotted')
    axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.show()


plot_error_sig(df_E_l1, df_nE_l1, 1, share_y=True)

# %%
###############################
# layer2
###############################
high_var_l2_E_map = var_E_l2_conti > np.quantile(var_E_l2_conti, 0.5)
plt.imshow(high_var_l2_E_map.reshape(16, int(hidden_dim[1] / 16)))

# %%
high_var_l2_nE_map = var_nE_l2_conti > np.quantile(var_nE_l2_conti, 0.5)
plt.imshow(high_var_l2_nE_map.reshape(16, int(hidden_dim[1] / 16)))

# %%
df_E_l2 = create_error_df(error_l2_wE, conti_error_l2_E, high_var_l2_E_map, 1, first_dig)
df_nE_l2 = create_error_df(error_l2_woE, conti_error_l2_nE, high_var_l2_nE_map, 1, first_dig)

# %%
plot_error_sig(df_E_l2, df_nE_l2, 2, share_y=False)

# %%
###############################
# layer3
###############################
### layer3
## w E model
conti_a_curr_l3_E = get_states([continuous_seq_hiddens_E], 10, hidden_dim[2], batch_size=conti_samplesize,
                               num_samples=conti_samplesize, T=T).squeeze()
conti_soma_l3_E = get_states([continuous_seq_hiddens_E], 8, hidden_dim[2], batch_size=conti_samplesize,
                             num_samples=conti_samplesize, T=T).squeeze()

conti_error_l3_E = (conti_a_curr_l3_E - conti_soma_l3_E) ** 2

# w/o E model
conti_a_curr_l3_nE = get_states([continuous_seq_hiddens_nE], 10, hidden_dim[2], batch_size=conti_samplesize,
                                num_samples=conti_samplesize, T=T).squeeze()
conti_soma_l3_nE = get_states([continuous_seq_hiddens_nE], 8, hidden_dim[2], batch_size=conti_samplesize,
                              num_samples=conti_samplesize, T=T).squeeze()

conti_error_l3_nE = (conti_a_curr_l3_nE - conti_soma_l3_nE) ** 2

# %%
var_E_l3_conti = [variance(conti_error_l3_E.mean(axis=0)[:, i]) for i in range(hidden_dim[2])]
var_nE_l3_conti = [variance(conti_error_l3_nE.mean(axis=0)[:, i]) for i in range(hidden_dim[2])]

# %%
high_var_l3_E_map = var_E_l3_conti > np.quantile(var_E_l3_conti, 0.5)
plt.imshow(high_var_l3_E_map.reshape(16, int(hidden_dim[2] / 16)))

# %%
high_var_l3_nE_map = var_nE_l3_conti > np.quantile(var_nE_l3_conti, 0.5)
plt.imshow(high_var_l3_nE_map.reshape(16, int(hidden_dim[2] / 16)))

# %%
df_E_l3 = create_error_df(error_l3_wE, conti_error_l3_E, high_var_l3_E_map, 2, first_dig)
df_nE_l3 = create_error_df(error_l3_woE, conti_error_l3_nE, high_var_l3_nE_map, 2, first_dig)

# %%
# plot
plot_error_sig(df_E_l3, df_nE_l3, 3, share_y=False)


# %%
##############################################################
# check acc during stim change seq
##############################################################
def get_acc_per_t(log_softmaxs, target):
    accs = []
    for i in range(len(log_softmaxs)):
        pred = log_softmaxs[i].data.max(1, keepdim=True)[1].cpu()
        accs.append(pred.eq(target.data.view_as(pred)).cpu().sum() / len(target))
    return accs


acc_E1 = get_acc_per_t(log_sm_E1, target_all[sample_image_nos[0]])
acc_E2 = get_acc_per_t(log_sm_E2, target_all[sample_image_nos[1]])

acc_E = np.concatenate((acc_E1, acc_E2))

acc_nE1 = get_acc_per_t(log_sm_nE1, target_all[sample_image_nos[0]])
acc_nE2 = get_acc_per_t(log_sm_nE2, target_all[sample_image_nos[1]])

acc_nE = np.concatenate((acc_nE1, acc_nE2))

sns.lineplot(acc_E, label='w E')
sns.lineplot(acc_nE, label='w/o E')
plt.title('acc in stim change sequence')
plt.legend()
plt.show()
# %%
###############################
# convergence speed 
###############################



# %%
###############################
# see how gaussian noise impact performance 
###############################

sns.heatmap(images_all[0] + torch.normal(mean=torch.zeros((28, 28)), std=torch.full((28, 28), fill_value=0.1)))


# %%
###############################
# the effect of silencing top down feedback connections 
###############################


# %%
def get_a_s(h, layer):
    n_samples = len(h[0][0])
    a = get_states([h], 2 + 4 * layer, hidden_dim[layer], n_samples, num_samples=n_samples, T=T)
    s = get_states([h], 0 + 4 * layer, hidden_dim[layer], n_samples, num_samples=n_samples, T=T)

    return a, s


match_a_E_l2, match_s_E_l2 = get_a_s(h_match_E, 1)
match_a_nE_l2, match_s_nE_l2 = get_a_s(h_match_nE, 1)

mismatch_a_E_l2, mismatch_s_E_l2 = get_a_s(h_mismatch_E, 1)
mismatch_a_nE_l2, mismatch_s_nE_l2 = get_a_s(h_mismatch_nE, 1)

match_a_E_l3, match_s_E_l3 = get_a_s(h_match_E, 2)
match_a_nE_l3, match_s_nE_l3 = get_a_s(h_match_nE, 2)

mismatch_a_E_l3, mismatch_s_E_l3 = get_a_s(h_mismatch_E, 2)
mismatch_a_nE_l3, mismatch_s_nE_l3 = get_a_s(h_mismatch_nE, 2)


# %%
# usi 

# delta F / F traces 
def compute_deltas(x):
    deltas = []
    for i in np.arange(1, len(x)):
        deltas.append((x[i] - x[i - 1]) / np.mean(x))
    return deltas


def usi_mismatch(expected_curr, unexpected_curr, layer_idx):
    df = pd.DataFrame(np.vstack((expected_curr.mean(axis=0).T, unexpected_curr.mean(axis=0).T)), \
                      columns=['t%i' % i for i in range(T)])
    df['neuron idx'] = np.concatenate((np.arange(hidden_dim[layer_idx]), np.arange(hidden_dim[layer_idx])))
    df['condition'] = ['match'] * hidden_dim[layer_idx] + ['mismatch'] * hidden_dim[layer_idx]

    # compute USI
    df['mean a'] = df.loc[:, 't10': 't199'].mean(axis=1)
    df['var a'] = df.loc[:, 't10': 't199'].var(axis=1)

    df_usi = pd.DataFrame({
        'neuron idx': np.arange(hidden_dim[layer_idx]),
        'usi': (df['mean a'][df['condition'] == 'match'].to_numpy() - \
                df['mean a'][df['condition'] == 'mismatch'].to_numpy()) / \
               np.sqrt((df['var a'][df['condition'] == 'match'].to_numpy() + \
                        df['var a'][df['condition'] == 'mismatch'].to_numpy()) / 2)
    })

    return df_usi

    # get index of neurons selective to stim change 
    # idx = df_usi.index[df_usi['usi'] <-0.5].tolist()
    # print(len(idx))

    # plt.plot(compute_deltas(expected_curr.mean(axis=0)[5:, idx].mean(axis=1)), label='match')
    # plt.plot(compute_deltas(unexpected_curr.mean(axis=0)[5:, idx].mean(axis=1)), label='mismatch')
    # plt.legend()
    # plt.show()


# %%
usi_l2_E = usi_mismatch(match_a_E_l2, mismatch_a_E_l2, 1)
usi_l2_nE = usi_mismatch(match_a_nE_l2, mismatch_a_nE_l2, 1)

df_l2 = pd.concat([usi_l2_E, usi_l2_nE])
df_l2['model type'] = ['E'] * hidden_dim[1] + ['w/o E'] * hidden_dim[1]

sns.histplot(df_l2, x='usi', hue='model type')
plt.show()

# %%
usi_l3_E = usi_mismatch(match_a_E_l3, mismatch_a_E_l3, 2)
usi_l3_nE = usi_mismatch(match_a_nE_l3, mismatch_a_nE_l3, 2)

df_l3 = pd.concat([usi_l3_E, usi_l3_nE])
df_l3['model type'] = ['E'] * hidden_dim[2] + ['w/o E'] * hidden_dim[2]

sns.histplot(df_l3, x='usi', hue='model type')
plt.show()

# %%
idx = df_l2[(df_l2['model type'] == 'E') & ((df_l2['usi'] > -0.2) & df_l2['usi'] < 0.2)]['neuron idx'].tolist()

plt.plot(match_a_E_l2.mean(axis=0)[5:, idx].mean(axis=1), label='match')
plt.plot(mismatch_a_E_l2.mean(axis=0)[5:, idx].mean(axis=1), label='mismatch')
plt.legend()
plt.show()

# %%

match_l1_E_error = get_error(h_match_E, 0)
match_l2_E_error = get_error(h_match_E, 1)
match_l3_E_error = get_error(h_match_E, 2)

mis_l1_E_error = get_error(h_mismatch_E, 0)
mis_l2_E_error = get_error(h_mismatch_E, 1)
mis_l3_E_error = get_error(h_mismatch_E, 2)


# %%
def mismatch_df(errors_match, errors_mismatch, layer_index):
    df = pd.DataFrame(np.vstack((errors_match.mean(axis=0), errors_mismatch.mean(axis=0))), \
                      columns=['neuron%i' % i for i in range(hidden_dim[layer_index])])
    df['t'] = np.concatenate((np.arange(T), np.arange(T)))
    df['condition'] = ['match'] * T + ['mismatch'] * T
    df = pd.melt(df, id_vars=['t', 'condition'], value_vars=['neuron%i' % i for i in range(hidden_dim[layer_index])],
                 var_name='neuron index', value_name='epsilon')
    return df


df_E_l3_mis = mismatch_df(match_l3_E_error, mis_l3_E_error, 2)

# %%
# with energy
spk_mis_l3 = get_states([h_mismatch_E], 9, hidden_dim[2], len(h_mismatch_E[0][0]), T,
                        num_samples=len(h_mismatch_E[0][0]))
spk_match_l3 = get_states([h_match_E], 9, hidden_dim[2], len(h_mismatch_E[0][0]), T,
                          num_samples=len(h_mismatch_E[0][0]))

sns.lineplot(df_E_l3_mis, x='t', y='epsilon', hue='condition')
plt.plot(spk_mis_l3.mean(axis=0).mean(axis=1), label='spk rate mismatch')
plt.plot(spk_match_l3.mean(axis=0).mean(axis=1), label='spk rate match')
plt.axvline(x=blank_t, color='black', linestyle='dotted', label='stim onset')
plt.axvline(x=T - end_seq_t, color='black', linestyle='dotted', label='stim offset')

plt.legend()

plt.show()

# %%
# w/o energy

match_l1_nE_error = get_error(h_match_nE, 0)
match_l2_nE_error = get_error(h_match_nE, 1)
match_l3_nE_error = get_error(h_match_nE, 2)

mis_l1_nE_error = get_error(h_mismatch_nE, 0)
mis_l2_nE_error = get_error(h_mismatch_nE, 1)
mis_l3_nE_error = get_error(h_mismatch_nE, 2)

df_nE_l3_mis = mismatch_df(match_l3_nE_error, mis_l3_nE_error, 2)

spk_mis_l3 = get_states([h_mismatch_nE], 9, hidden_dim[2], len(h_mismatch_nE[0][0]), T,
                        num_samples=len(h_mismatch_nE[0][0]))
spk_match_l3 = get_states([h_match_nE], 9, hidden_dim[2], len(h_mismatch_nE[0][0]), T,
                          num_samples=len(h_mismatch_nE[0][0]))

sns.lineplot(df_nE_l3_mis, x='t', y='epsilon', hue='condition')
plt.plot(spk_mis_l3.mean(axis=0).mean(axis=1), label='spk rate mismatch')
plt.plot(spk_match_l3.mean(axis=0).mean(axis=1), label='spk rate match')
plt.axvline(x=blank_t, color='black', linestyle='dotted', label='stim onset')
plt.axvline(x=T - end_seq_t, color='black', linestyle='dotted', label='stim offset')

plt.legend()

plt.show()

# %%
fig, axes = plt.subplots(1, 2, figsize=(7, 3))
sns.lineplot(df_E_l3_mis, x='t', y='epsilon', ax=axes[0])
axes[0].axvline(x=blank_t, color='black', linestyle='dotted')
axes[0].axvline(x=match_t + blank_t, color='red', linestyle='dotted')
axes[0].set_title('w E')
# axes[0].get_legend().remove()

sns.lineplot(df_nE_l3_mis, x='t', y='epsilon', ax=axes[1])
axes[1].axvline(x=blank_t, color='black', linestyle='dotted', label='match')
axes[1].axvline(x=match_t + blank_t, color='red', linestyle='dotted', label='mismatch')
axes[1].set_title('w/o E')
axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

# %%
##############################################################
# check what neurons are sensitive to vs projection from layer2 to apical tuft 
##############################################################
# essentially whether top down and bottom up weights are matching 
mean_zero = images_all[target_all == 8][0]
plt.imshow(mean_zero)

# %%
input2layer2 = param_dict_wE['layer1to2.weight'] @ mean_zero.cpu().numpy().flatten()
input2layer2[input2layer2 < 0] = 0
max = np.max(input2layer2)

plt.imshow((param_dict_wE['layer2to1.weight'] @ input2layer2).reshape(28, 28), vmin=-max, vmax=max, cmap='vlag')

# %%
##############################################################
# repetition suppresion 
############################################################## 

# make sequence 
for name, param in model_wE.named_parameters():
    if 'bias' in name:
        param.data = torch.zeros(1).to(device)
        print(name)

# %%
# whether to block feedback 
feedback_block = False
if feedback_block:
    model_wE.layer2to1.weight.data = torch.zeros(model_wE.layer2to1.weight.size()).to(device)
    model_wE.layer3to2.weight.data = torch.zeros(model_wE.layer3to2.weight.size()).to(device)
    model_wE.out2layer3.weight.data = torch.zeros(model_wE.out2layer3.weight.size()).to(device)

    model_wE.layer2to1.bias.data = torch.zeros(model_wE.layer2to1.bias.size()).to(device)
    model_wE.layer3to2.bias.data = torch.zeros(model_wE.layer3to2.bias.size()).to(device)
    model_wE.out2layer3.bias.data = torch.zeros(model_wE.out2layer3.bias.size()).to(device)

    model_woE.layer2to1.weight.data = torch.zeros(model_woE.layer2to1.weight.size()).to(device)
    model_woE.layer3to2.weight.data = torch.zeros(model_woE.layer3to2.weight.size()).to(device)
    model_woE.out2layer3.weight.data = torch.zeros(model_woE.out2layer3.weight.size()).to(device)

    model_woE.layer2to1.bias.data = torch.zeros(model_woE.layer2to1.bias.size()).to(device)
    model_woE.layer3to2.bias.data = torch.zeros(model_woE.layer3to2.bias.size()).to(device)
    model_woE.out2layer3.bias.data = torch.zeros(model_woE.out2layer3.bias.size()).to(device)
else:
    model_wE = SnnNetwork2Layer(IN_dim, hidden_dim, n_classes, is_adapt=adap_neuron, one_to_one=onetoone, dp_rate=0.4)
    model_wE.to(device)

    model_woE = SnnNetwork2Layer(IN_dim, hidden_dim, n_classes, is_adapt=adap_neuron, one_to_one=onetoone, dp_rate=0.4)
    model_woE.to(device)

    model_wE.load_state_dict(saved_dict1['state_dict'])
    model_woE.load_state_dict(saved_dict2['state_dict'])


# %%

def make_rep_sup_seq(index: int, iti: int, start_t=20, present_t=30, seq_len=100):
    """given index create repetition suppression sequence 

    Args:
        index (int): index from images 
        iti: intertrial interval 
        start_t: time steps before starting to present stim 
        present_t: how long to present stim 
    """
    img = images_all[index]
    w, h = img.size()
    seq = torch.concat([torch.zeros((start_t, w, h)),
                        img.repeat(present_t, 1, 1),
                        torch.zeros((iti, w, h)),
                        img.repeat(present_t, 1, 1),
                        torch.zeros((seq_len - start_t - 2 * present_t - iti), w, h)])
    print(seq.size())
    return seq


start_t = 20
present_t = 30

seq_len = 200
rep_seq = make_rep_sup_seq(3, 40, seq_len=seq_len, start_t=start_t, present_t=present_t)

with torch.no_grad():
    model_wE.eval()
    model_woE.eval()

    hidden_i = model_wE.init_hidden_allzero(1)

    log_sm_E_sup, hidden_sup_E = model_wE.inference(rep_seq.view(seq_len, -1, IN_dim).to(device), hidden_i, seq_len,
                                                    bystep=True)
    log_sm_nE_sup, hidden_sup_nE = model_woE.inference(rep_seq.view(seq_len, -1, IN_dim).to(device), hidden_i, seq_len,
                                                       bystep=True)
torch.cuda.empty_cache()

# %%
l1_spk_E = get_states([hidden_sup_E], 1, hidden_dim[0], batch_size=1, T=seq_len, num_samples=1)
l2_spk_E = get_states([hidden_sup_E], 5, hidden_dim[1], batch_size=1, T=seq_len, num_samples=1)
l3_spk_E = get_states([hidden_sup_E], 9, hidden_dim[2], batch_size=1, T=seq_len, num_samples=1)
print(l1_spk_E.shape)

l1_spk_nE = get_states([hidden_sup_nE], 1, hidden_dim[0], batch_size=1, T=seq_len, num_samples=1)
l2_spk_nE = get_states([hidden_sup_nE], 5, hidden_dim[1], batch_size=1, T=seq_len, num_samples=1)
l3_spk_nE = get_states([hidden_sup_nE], 9, hidden_dim[2], batch_size=1, T=seq_len, num_samples=1)

# %%
plt.plot(l1_spk_E.mean(0).mean(axis=1), label='l1 E')
plt.plot(l2_spk_E.mean(0).mean(axis=1), label='l2 E')
plt.plot(l3_spk_E.mean(0).mean(axis=1), label='l3 E')

# plt.plot(l1_spk_nE.mean(0).mean(axis=1), label='l1 nE')
# plt.plot(l2_spk_nE.mean(0).mean(axis=1), label='l2 nE')
# plt.plot(l3_spk_nE.mean(0).mean(axis=1), label='l3 nE')

plt.legend()
plt.show()

# %%
from scipy.ndimage import gaussian_filter

fig, ax = plt.subplots()
for iti_ in np.arange(5, 60, step=10):
    rep_seq = make_rep_sup_seq(3, iti_, seq_len=seq_len)

    with torch.no_grad():
        model_wE.eval()

        hidden_i = model_wE.init_hidden_allzero(1)

        log_sm_E_sup, hidden_sup_E = model_wE.inference(rep_seq.view(seq_len, -1, IN_dim).to(device), hidden_i, seq_len,
                                                        bystep=True)
    torch.cuda.empty_cache()

    l1_spk_E = get_states([hidden_sup_E], 5, hidden_dim[1], batch_size=1, T=seq_len, num_samples=1)

    plt.plot(gaussian_filter(l1_spk_E.mean(0).mean(axis=1), sigma=2), label='iti %i' % iti_)

plt.legend()
plt.show()

# %%


# %%
dig = 0
fig = plt.figure()
plt.plot(error_l1_wE[target_all == dig].mean(axis=0).mean(axis=1), label='w E')
plt.plot(error_l1_woE[target_all == dig].mean(axis=0).mean(axis=1), label='w/o E')
plt.title('layer 1 mean (a_curr - soma)**2 over T, class %i' % dig)
plt.legend()
plt.show()

# %%


fig = plt.figure()
plt.plot(error_l2_wE[target_all == dig].mean(axis=0).mean(axis=1), label='w E')
plt.plot(error_l2_woE[target_all == dig].mean(axis=0).mean(axis=1), label='w/o E')
plt.title('layer 2 mean (a_curr - soma)**2 over T, class %i' % dig)
plt.legend()
plt.show()

# %%
fig, axs = plt.subplots(1, 2, figsize=(7, 3))
sns.heatmap(param_dict_wE['output_layer.fc.weight'], ax=axs[0], cmap='vlag', vmin=-0.5, vmax=0.5)
axs[0].set_title('w E layer2 to out w')
sns.heatmap(param_dict_woE['output_layer.fc.weight'], ax=axs[1], cmap='vlag', vmin=-0.5, vmax=0.5)
axs[1].set_title('w/o E layer2 to out w')
plt.tight_layout()
plt.show()

# %%


# %%
plt.plot((conti_error_l2_E).mean(axis=0).mean(axis=1), label='w E')
# plt.plot((conti_error_l2_nE).mean(axis=0).mean(axis=1), label='w/o E')
plt.legend()
plt.title('layer 2 mean((a_curr - soma)**2) with stimulus change at t = %i' % T)
plt.show()

# %%
plt.plot(conti_a_curr_l2_E.mean(axis=0).mean(axis=1), label='a_curr')
plt.plot(conti_soma_l2_E.mean(axis=0).mean(axis=1), label='soma')
plt.legend()
plt.title('layer 2 w E soma and a_curr over t')
plt.show()

# %%
plt.plot((conti_error_l1_E).mean(axis=0).mean(axis=1), label='w E')
plt.plot((conti_error_l1_nE).mean(axis=0).mean(axis=1), label='w/o E')
plt.legend()
plt.title('layer 1 mean((a_curr - soma)**2) with stimulus change at t = %i' % int(T / 2))
plt.show()

# %%
plt.plot(conti_a_curr_l1_E.mean(axis=0).mean(axis=1), label='a_curr wE')
plt.plot(conti_soma_l1_E.mean(axis=0).mean(axis=1), label='soma wE')

plt.plot(conti_a_curr_l1_nE.mean(axis=0).mean(axis=1), label='a_curr nE')
plt.plot(conti_soma_l1_nE.mean(axis=0).mean(axis=1), label='soma nE')

plt.legend()
plt.title('layer 1 soma and a_curr over t')
plt.show()

# %%
# plot spk sequence 
fig, axes = plt.subplots(2, 30, figsize=(60, 3))
for i in range(T):
    axes[0, i].imshow(continuous_seq_hiddens_E[i * 2][1].detach().cpu().numpy().reshape(28, 28))
    axes[0, i].axis('off')
    axes[0, i].set_title('w E')

    axes[1, i].imshow(continuous_seq_hiddens_nE[i * 2][1].detach().cpu().numpy().reshape(28, 28))
    axes[1, i].axis('off')
    axes[1, i].set_title('w/o E')

# %%
# plot a_curr sequence 
fig, axes = plt.subplots(2, 30, figsize=(60, 3))
for i in range(T):
    pos1 = axes[0, i].imshow(continuous_seq_hiddens_E[i * 2][2].detach().cpu().numpy().reshape(28, 28))
    axes[0, i].axis('off')
    fig.colorbar(pos1, ax=axes[0, i], shrink=0.5)
    axes[0, i].set_title('w E')

    pos2 = axes[1, i].imshow(continuous_seq_hiddens_nE[i * 2][2].detach().cpu().numpy().reshape(28, 28))
    axes[1, i].axis('off')
    fig.colorbar(pos2, ax=axes[1, i], shrink=0.5)
    axes[1, i].set_title('w/o E')

# %%
# plot top down signal from layer2 to layer 1 
fig, axes = plt.subplots(2, 30, figsize=(60, 3))
for i in range(T):
    top_down_E = param_dict_wE['layer2to1.weight'] @ continuous_seq_hiddens_E[i * 2][5][
        0].detach().cpu().numpy().squeeze()
    pos1 = axes[0, i].imshow(top_down_E.reshape(28, 28))
    axes[0, i].axis('off')
    fig.colorbar(pos1, ax=axes[0, i], shrink=0.5)
    axes[0, i].set_title('w E')

    top_down_nE = param_dict_woE['layer2to1.weight'] @ continuous_seq_hiddens_nE[i * 2][5][
        0].detach().cpu().numpy().squeeze()
    pos2 = axes[1, i].imshow(top_down_nE.reshape(28, 28))
    fig.colorbar(pos2, ax=axes[1, i], shrink=0.5)
    axes[1, i].axis('off')
    axes[1, i].set_title('w/o E')

# %%
# plot error in layer1 a_curr - soma
fig, axes = plt.subplots(2, 30, figsize=(60, 3))
for i in range(T):
    error1 = continuous_seq_hiddens_E[i * 2][2].detach().cpu().numpy().squeeze() - \
             continuous_seq_hiddens_E[i * 2][0].detach().cpu().numpy().squeeze()
    pos1 = axes[0, i].imshow((error1 ** 2).reshape(28, 28), cmap='rocket')
    axes[0, i].axis('off')
    fig.colorbar(pos1, ax=axes[0, i], shrink=0.5)
    axes[0, i].set_title('w E')

    error2 = continuous_seq_hiddens_nE[i * 2][2].detach().cpu().numpy().squeeze() - \
             continuous_seq_hiddens_nE[i * 2][0].detach().cpu().numpy().squeeze()
    pos2 = axes[1, i].imshow((error2 ** 2).reshape(28, 28), cmap='rocket')
    fig.colorbar(pos2, ax=axes[1, i], shrink=0.5)
    axes[1, i].axis('off')
    axes[1, i].set_title('w/o E')

# %%

# %%


# %%
plt.plot((conti_error_l1_nE.mean(axis=0)[:, var < 2] - error_l1_woE[sample_image_nos[0]].mean(axis=0)[:, var < 2]))
plt.show()

# %%
