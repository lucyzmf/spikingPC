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
n_per_class = 40
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
weights = [x for x in param_names_wE if ('weight' in x) and ('bias' not in x)]

plt.style.use('seaborn-v0_8-deep')

fig, axes = plt.subplots(1, len(weights), figsize=(len(weights) * 3, 3))
for i in range(len(weights)):
    axes[i].hist(param_dict_wE[weights[i]].flatten(), label='w E', histtype='step')
    axes[i].hist(param_dict_woE[weights[i]].flatten(), label='w/o E', histtype='step')
    axes[i].legend()
    axes[i].set_title(weights[i])

plt.tight_layout()
plt.show()

# %%
taus = [x for x in param_names_wE if ('tau' in x)]

fig, axes = plt.subplots(1, len(taus), figsize=(len(taus) * 3, 3))
for i in range(len(taus)):
    axes[i].hist(param_dict_wE[taus[i]].flatten(), label='w E', histtype='step')
    axes[i].hist(param_dict_woE[taus[i]].flatten(), label='w/o E', histtype='step')
    axes[i].legend()
    axes[i].set_title(taus[i])

plt.tight_layout()
plt.show()

# %%
# get analysis data 
occlusion_p = None
batches = 2
hiddens_wE, preds_wE, images_all, _ = get_all_analysis_data(model_wE, test_loader, device, IN_dim, T, batch_no=batches,
                                                            occlusion_p=occlusion_p)
hiddens_woE, preds_woE, _, _ = get_all_analysis_data(model_woE, test_loader, device, IN_dim, T, batch_no=batches,
                                                     occlusion_p=occlusion_p)

n_samples = len(preds_wE)
target_all = testdata.targets.data[:n_samples]

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
def get_error(h, layer, n_samples=n_samples, b_size=batch_size):
    a = get_states(h, 2 + 4 * layer, hidden_dim[layer], b_size, num_samples=n_samples, T=T)
    s = get_states(h, 0 + 4 * layer, hidden_dim[layer], b_size, num_samples=n_samples, T=T)

    return (a - s) ** 2


# %%
error_l2 = get_error(hiddens_wE, 1)

error_byclass = get_value_byclass(error_l2)
for i in range(n_classes):
    plt.plot(error_byclass[i, :], label=str(i))
plt.legend()
plt.show()
# %%
error_l2 = get_error(hiddens_woE, 1)

error_byclass = get_value_byclass(error_l2)
for i in range(n_classes):
    plt.plot(error_byclass[i, :], label=str(i))
plt.legend()
plt.show()
# sum episolon need to be normalised by mean spk rate 


# %%
# spk mean at layer 1 
if occlusion_p is not None:
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

# get mean curr by class to compare between groups
def get_a_s_e(hidden, layer, batch_size, n_samples, T):
    a = get_states(hidden, 2 + layer * 4, hidden_dim[layer], batch_size, num_samples=n_samples, T=T)
    s = get_states(hidden, 0 + layer * 4, hidden_dim[layer], batch_size, num_samples=n_samples, T=T)
    e = (a - s) ** 2
    return s, a, e


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
conti_soma_l1_woE, conti_a_curr_l1_woE, conti_error_l1_woE = get_a_s_e([continuous_seq_hiddens_nE], 0, size_lim, size_lim, T)

conti_soma_l2_wE, conti_a_curr_l2_wE, conti_error_l2_wE = get_a_s_e([continuous_seq_hiddens_E], 1, size_lim, size_lim,
                                                                    T)
conti_soma_l2_woE, conti_a_curr_l2_woE, conti_error_l2_woE = get_a_s_e([continuous_seq_hiddens_nE], 1, size_lim, size_lim, T)

conti_soma_l3_wE, conti_a_curr_l3_wE, conti_error_l3_wE = get_a_s_e([continuous_seq_hiddens_E], 2, size_lim, size_lim,
                                                                    T)
conti_soma_l3_woE, conti_a_curr_l3_woE, conti_error_l3_woE = get_a_s_e([continuous_seq_hiddens_nE], 2, size_lim, size_lim, T)


# %%
###############################
# usi analysis
###############################
def usi(expected_curr, unexpected_curr, layer_idx, ts=None):
    df = pd.DataFrame(np.vstack((expected_curr.mean(axis=0).T, unexpected_curr.mean(axis=0).T)),
                      columns=['t%i' % i for i in range(T)])
    df['neuron idx'] = np.concatenate((np.arange(hidden_dim[layer_idx]), np.arange(hidden_dim[layer_idx])))
    df['condition'] = ['normal seq'] * hidden_dim[layer_idx] + ['stim change seq'] * hidden_dim[layer_idx]

    # compute USI
    if ts is not None:
        df['mean a'] = df.loc[:, 't'+str(ts[0]):'t'+str(ts[1])].mean(axis=1)
    else:
        df['mean a'] = df.loc[:, 't10': 't50'].mean(axis=1)
    df['var a'] = df.loc[:, 't10': 't199'].var(axis=1)

    df_usi = pd.DataFrame({
        'neuron idx': np.arange(hidden_dim[layer_idx]),
        'usi': (df['mean a'][df['condition'] == 'normal seq'].to_numpy() - df['mean a'][
            df['condition'] == 'stim change seq'].to_numpy()) /
               np.sqrt(df['var a'][df['condition'] == 'normal seq'].to_numpy())
    })

    return df, df_usi

# %%
df_l2_a_E, df_usi_l2_a_E = usi(a_curr_l2_wE, conti_a_curr_l2_wE, 1)
df_l2_s_E, df_usi_l2_s_E = usi(soma_l2_wE, conti_soma_l2_wE, 1)

df_l2_a_woE, df_usi_l2_a_woE = usi(a_curr_l2_woE, conti_a_curr_l2_woE, 1)
df_l2_s_woE, df_usi_l2_s_woE = usi(soma_l2_woE, conti_soma_l2_woE, 1)

# %%
df_usi_compare = pd.concat([df_usi_l2_s_E, df_usi_l2_s_woE])
df_usi_compare['model type'] = ['E'] * hidden_dim[1] + ['w/o E'] * hidden_dim[1]

sns.histplot(df_usi_compare, x='usi', hue='model type', element="step", stat="density")
plt.title('compare usi of change exp layer2 by model type (soma)')
plt.show()

# %%
high_usi_index = df_usi_l2_s_E.sort_values(by='usi')['neuron idx'][0]

from scipy.ndimage import median_filter, gaussian_filter

def compute_delta(signal):
    """compute delta of signal of a single neuron from multiple samples 

    Args:
        signal (_type_): n * T

    """
    _, T = signal.shape
    delta = signal[:, 1:] - signal[:, :T-1]
    return delta 

def df_single_neuron(expected_curr, unexpected_curr, neuron_idx, delta=None):
    steps = T-1
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

    df_['condition'] = ['normal seq'] * len(expected_curr) + ['stim change seq'] * len(expected_curr)
    df_ = pd.melt(df_, id_vars=['condition'], value_vars=['t%i' % i for i in range(steps)],
                  var_name='t', value_name='volt')
    return df_


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
mismatch_dig = 7
mismatch_dig = np.delete(np.arange(0, 10), match_dig)

sample_size = 50
zeros = images[targets == match_dig][:sample_size].to(device)

no_inputs = torch.zeros((zeros.size(0), hidden_dim[0])).to(device)

blank_t = 50
match_t = 100
end_seq_t = T - blank_t - match_t


def match_mismatch_ex(match_condition):
    h_E = []
    h_nE = []

    clamp_class = match_dig if match_condition else mismatch_dig

    with torch.no_grad():
        model_wE.eval()
        model_woE.eval()

        hidden_i = model_wE.init_hidden(zeros.size(0))

        _, h1_E = model_wE.inference(no_inputs, hidden_i, blank_t)
        _, h1_nE = model_woE.inference(no_inputs, hidden_i, blank_t)
        h_E += h1_E
        h_nE += h1_nE

        _, h2_E = model_wE.clamped_generate(clamp_class, zeros.view(-1, hidden_dim[0]), h1_E[-1], match_t,
                                            clamp_value=1, batch=True)
        _, h2_nE = model_woE.clamped_generate(clamp_class, zeros.view(-1, hidden_dim[0]), h1_nE[-1], match_t,
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
df_l2_a_matchexp_E, df_usi_l2_a_matchexp_E = usi(match_a_curr_l2_wE, mis_a_curr_l2_wE, 1, ts=[50, 150])
df_l2_s_matchexp_E, df_usi_l2_s_matchexp_E = usi(match_soma_l2_wE, mis_soma_l2_wE, 1, ts=[50, 150])

df_l2_a_matchexp_woE, df_usi_l2_a_matchexp_woE = usi(match_a_curr_l2_woE, mis_a_curr_l2_woE, 1, ts=[50, 150])
df_l2_s_matchexp_woE, df_usi_l2_s_matchexp_woE = usi(match_soma_l2_woE, mis_soma_l2_woE, 1, ts=[50, 150])

# %%
df_usi_compare = pd.concat([df_usi_l2_s_matchexp_E, df_usi_l2_s_matchexp_woE])
df_usi_compare['model type'] = ['E'] * hidden_dim[1] + ['w/o E'] * hidden_dim[1]

sns.histplot(df_usi_compare, x='usi', hue='model type')
plt.title('compare usi of match mismatch layer2 by model type (soma)')
plt.show()

# %%
low_usi_index = df_usi_l2_s_matchexp_E.sort_values(by='usi')['neuron idx'][256]

df_single_s = df_single_neuron(match_soma_l2_wE[:size_lim], mis_soma_l2_wE[:size_lim], low_usi_index)
sns.lineplot(df_single_s, x='t', y='volt', hue='condition')
plt.title('low usi neuron soma voltage during seq match mismatch')
plt.show()

# %%
high_usi_index = df_usi_l2_s_matchexp_E.sort_values(by='usi')['neuron idx'][0]

df_single_s = df_single_neuron(match_soma_l2_wE[:size_lim], mis_soma_l2_wE[:size_lim], high_usi_index)
sns.lineplot(df_single_s, x='t', y='volt', hue='condition')
plt.title('high usi neuron soma voltage during seq match mismatch')
plt.show()
# %%

high_usi_index = df_usi_l2_s_matchexp_E.sort_values(by='usi')['neuron idx'][0]

df_single_s = df_single_neuron(match_soma_l2_wE[:size_lim], mis_soma_l2_wE[:size_lim], high_usi_index, delta=True)
sns.lineplot(df_single_s, x='t', y='volt', hue='condition')
plt.title('high usi neuron delta soma voltage during seq match mismatch')
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
# see how occlussion impact performance 
###############################
occlusion_p = np.arange(0.1, 1., 0.1)


def occu_test(occ_p):
    accs_E = []
    accs_nE = []

    for p in occ_p:
        print(p)
        _, e = get_all_analysis_data(model_wE, test_loader, device, IN_dim, T, batch_no=50, occlusion_p=p, log=False)
        _, ne = get_all_analysis_data(model_woE, test_loader, device, IN_dim, T, batch_no=50, occlusion_p=p, log=False)
        accs_E.append(e)
        accs_nE.append(ne)

    return accs_E, accs_nE


occ_accE, occ_accnE = occu_test(occlusion_p)

# %%
df = pd.DataFrame({
    'model': ['E'] * len(occlusion_p) + ['w/o E'] * len(occlusion_p),
    'error rate': np.concatenate((occ_accE, occ_accnE)),
    'occ p': np.round(np.concatenate((1 - occlusion_p, 1 - occlusion_p)), decimals=1)
})
sns.barplot(df, x='occ p', y='error rate', hue='model')
sns.despine()
plt.show()

# %%
##############################################################
# test generative capacity of network with clamping 
##############################################################
dig = 0
no_input = torch.zeros((1, hidden_dim[0])).to(device)
with torch.no_grad():
    model_wE.eval()
    model_woE.eval()

    hidden_i = model_wE.init_hidden(1)

    log_sm_E_gen, hidden_gen_E = model_wE.clamped_generate(dig, no_input, hidden_i, T * 2, clamp_value=10)
    log_sm_nE_gen, hidden_gen_nE = model_woE.clamped_generate(dig, no_input, hidden_i, T * 2, clamp_value=10)
torch.cuda.empty_cache()

# %%
spk_gen_l1_E = get_states([hidden_gen_E], 1, hidden_dim[0], 1, T, num_samples=1).squeeze().mean(axis=0)
spk_gen_l1_nE = get_states([hidden_gen_nE], 1, hidden_dim[0], 1, T, num_samples=1).squeeze().mean(axis=0)

fig, axes = plt.subplots(1, 2)
pos = axes[0].imshow(spk_gen_l1_E.reshape(28, 28))
fig.colorbar(pos, ax=axes[0], shrink=0.5)
axes[0].set_title('w E mean spk l1 clamped generation')

pos = axes[1].imshow(spk_gen_l1_nE.reshape(28, 28))
fig.colorbar(pos, ax=axes[1], shrink=0.5)
axes[1].set_title('w/o E mean spk l1 clamped generation')

plt.tight_layout()
plt.show()

# %%
spk_gen_l2_E = get_states([hidden_gen_E], 5, hidden_dim[1], 1, T, num_samples=1).squeeze().mean(axis=0)
spk_gen_l2_nE = get_states([hidden_gen_nE], 5, hidden_dim[1], 1, T, num_samples=1).squeeze().mean(axis=0)

fig, axes = plt.subplots(1, 2)
pos = axes[0].imshow(spk_gen_l2_E.reshape(16, 32))
fig.colorbar(pos, ax=axes[0], shrink=0.5)
axes[0].set_title('w E mean spk l2 clamped generation')

pos = axes[1].imshow(spk_gen_l2_nE.reshape(16, 32))
fig.colorbar(pos, ax=axes[1], shrink=0.5)
axes[1].set_title('w/o E mean spk l2 clamped generation')

plt.tight_layout()
plt.show()

# %%
fig, axes = plt.subplots(1, 2)
pos = axes[0].imshow((param_dict_wE['layer2to1.weight'] @ spk_gen_l2_E).reshape(28, 28))
fig.colorbar(pos, ax=axes[0], shrink=0.5)
axes[0].set_title('w E mean l2 > l1 clamped generation')

pos = axes[1].imshow((param_dict_wE['layer2to1.weight'] @ spk_gen_l2_nE).reshape(28, 28))
fig.colorbar(pos, ax=axes[1], shrink=0.5)
axes[1].set_title('w/o E mean l2 > l1 clamped generation')

plt.tight_layout()
plt.show()

# %%
# compute mean l2, l3 reps from E and nE model with clamped mode 
l2_norm_E = np.zeros((10, hidden_dim[1]))
l3_norm_E = np.zeros((10, hidden_dim[2]))

l2_norm_nE = np.zeros((10, hidden_dim[1]))
l3_norm_nE = np.zeros((10, hidden_dim[2]))

# get means from normal condition
for i, (data, target) in enumerate(test_loader):
    data, target = data.to(device), target.to(device)
    data = data.view(-1, model_wE.in_dim)

    with torch.no_grad():
        model_wE.eval()
        model_woE.eval()

        hidden = model_wE.init_hidden(data.size(0))

        _, h_E = model_wE.inference(data, hidden, T)
        _, h_nE = model_woE.inference(data, hidden, T)

        l2_E = get_states([h_E], 5, hidden_dim[1], batch_size, T=60, num_samples=batch_size)
        l2_nE = get_states([h_nE], 5, hidden_dim[1], batch_size, T=60, num_samples=batch_size)

        l3_E = get_states([h_E], 9, hidden_dim[2], batch_size, T=60, num_samples=batch_size)
        l3_nE = get_states([h_nE], 9, hidden_dim[2], batch_size, T=60, num_samples=batch_size)

        for i in range(n_classes):
            l2_norm_E[i] += l2_E[target.cpu() == i].mean(axis=1).sum(axis=0)  # avg over t and sum over all samples 
            l2_norm_nE[i] += l2_nE[target.cpu() == i].mean(axis=1).sum(axis=0)

            l3_norm_E[i] += l3_E[target.cpu() == i].mean(axis=1).sum(axis=0)  # avg over t and sum over all samples 
            l3_norm_nE[i] += l3_nE[target.cpu() == i].mean(axis=1).sum(axis=0)

    torch.cuda.empty_cache()

# avg all samples 
l2_norm_E = l2_norm_E / 10000
l2_norm_nE = l2_norm_nE / 10000

l3_norm_E = l3_norm_E / 10000
l3_norm_nE = l3_norm_nE / 10000

# %%
# clamped condition
l2_clamp_E = np.zeros((10, hidden_dim[1]))
l3_clamp_E = np.zeros((10, hidden_dim[2]))

l2_clamp_nE = np.zeros((10, hidden_dim[1]))
l3_clamp_nE = np.zeros((10, hidden_dim[2]))

for i in range(10):
    print(i)
    with torch.no_grad():
        model_wE.eval()
        model_woE.eval()

        hidden_i = model_wE.init_hidden(1)

        _, hidden_gen_E_ = model_wE.clamped_generate(i, no_input, hidden_i, T * 2, clamp_value=10)
        _, hidden_gen_nE_ = model_woE.clamped_generate(i, no_input, hidden_i, T * 2, clamp_value=10)

        # get gen 
        l2_E = get_states([hidden_gen_E_], 5, hidden_dim[1], 1, T * 2, num_samples=1)
        l2_nE = get_states([hidden_gen_nE_], 5, hidden_dim[1], 1, T * 2, num_samples=1)

        l3_E = get_states([hidden_gen_E_], 9, hidden_dim[2], 1, T * 2, num_samples=1)
        l3_nE = get_states([hidden_gen_nE_], 9, hidden_dim[2], 1, T * 2, num_samples=1)

        l2_clamp_E[i] += np.squeeze(l2_E.mean(axis=1))
        l2_clamp_nE[i] += np.squeeze(l2_nE.mean(axis=1))

        l3_clamp_E[i] += np.squeeze(l3_E.mean(axis=1))
        l3_clamp_nE[i] += np.squeeze(l3_nE.mean(axis=1))

    torch.cuda.empty_cache()

# %%
fig, axes = plt.subplots(2, 10, figsize=(25, 6))
for i in range(10):
    pos = axes[0][i].imshow((param_dict_wE['layer2to1.weight'] @ l2_clamp_E[i]).reshape(28, 28))
    fig.colorbar(pos, ax=axes[0][i], shrink=0.5)
    axes[0][i].set_title('w E l2 > l1 class%i' % i)

    pos = axes[1][i].imshow((param_dict_woE['layer2to1.weight'] @ l2_clamp_nE[i]).reshape(28, 28))
    fig.colorbar(pos, ax=axes[1][i], shrink=0.5)
    axes[1][i].set_title('w/o E l2 > l1 class%i' % i)

plt.tight_layout()
plt.show()

# %%
# how classifiable are the generated representations 
from scipy.spatial import distance

dist_l2_E = []
dist_l2_nE = []

dist_l3_E = []
dist_l3_nE = []

for i in range(10):
    dist_l2_E.append(distance.cosine(l2_clamp_E[i], l2_norm_E[i]))
    dist_l2_nE.append(distance.cosine(l2_clamp_nE[i], l2_norm_nE[i]))

    dist_l3_E.append(distance.cosine(l3_clamp_E[i], l3_norm_E[i]))
    dist_l3_nE.append(distance.cosine(l3_clamp_nE[i], l3_norm_nE[i]))

df = pd.DataFrame({
    'cosine dist': dist_l2_E + dist_l3_E + dist_l2_nE + dist_l3_nE,
    'class': np.concatenate((np.arange(10), np.arange(10), np.arange(10), np.arange(10))),
    'model': ['w E'] * 20 + ['w/o E'] * 20,
    'layer': np.concatenate((np.full(10, 2), np.full(10, 3), np.full(10, 2), np.full(10, 3)))
})

sns.catplot(df, x='class', y='cosine dist', hue='model', kind='bar', col='layer')
plt.show()

# %%
from sklearn.metrics import pairwise_distances

pair_dist_E_l3 = pairwise_distances(l3_clamp_E, l3_norm_E, metric='cosine')
pair_dist_nE_l3 = pairwise_distances(l3_clamp_nE, l3_norm_nE, metric='cosine')

pair_dist_E_l2 = pairwise_distances(l2_clamp_E, l2_norm_E, metric='cosine')
pair_dist_nE_l2 = pairwise_distances(l2_clamp_nE, l2_norm_nE, metric='cosine')

fig, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True, sharey=True)
sns.despine()
cbar_ax = fig.add_axes([.93, .3, .03, .4])
sns.heatmap(pair_dist_E_l2, ax=axes[0, 0], vmin=0, vmax=0.6, cbar_ax=cbar_ax)
axes[0, 0].set_ylabel('clamped reps')
axes[0, 0].set_title('normal reps')
axes[0, 0].tick_params(left=False, bottom=False)

sns.heatmap(pair_dist_nE_l2, ax=axes[0, 1], vmin=0, vmax=0.6, cbar=False)
axes[0, 1].set_title('normal reps')
axes[0, 1].tick_params(left=False, bottom=False)

sns.heatmap(pair_dist_E_l3, ax=axes[1, 0], vmin=0, vmax=0.6, cbar=False)
axes[1, 0].set_ylabel('clamped reps')
axes[1, 0].tick_params(left=False, bottom=False)

sns.heatmap(pair_dist_nE_l3, ax=axes[1, 1], vmin=0, vmax=0.6, cbar=False)
axes[1, 1].tick_params(left=False, bottom=False)

cols = ['w E', 'w/o E']
rows = ['layer2', 'layer3']

pad = 20

for ax, col in zip(axes[0], cols):
    ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')

for ax, row in zip(axes[:, 0], rows):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad / 2, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')

# plt.tight_layout()
plt.show()

# %%
###############################
# exci inhi divide for w E model top down signal 
###############################
ex = []
inhi = []
for i in range(10):
    top_down = param_dict_wE['layer3to2.weight'] @ l3_norm_E[i]
    ex.append(top_down[top_down > 0].sum())
    inhi.append(-top_down[top_down < 0].sum())

df = pd.DataFrame({
    'class': np.concatenate((np.arange(10), np.arange(10))),
    'strength': ex + inhi,
    'type': ['ex'] * 10 + ['inhi'] * 10
})
sns.barplot(df, x='class', y='strength', hue='type')
plt.show()

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
