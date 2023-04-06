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
n_classes = 10
num_readout = 10
adap_neuron = True
onetoone = True

# %%
IN_dim = 784
hidden_dim = [784, 256]
T = 200  # sequence length, reading from the same image time_steps times

dp = 0.
# fptt_model_acc = []
# bp_model_acc = []

# for i in range(len(dp)):
    # define network
model_wE = SnnNetwork(IN_dim, hidden_dim, n_classes, is_adapt=adap_neuron, one_to_one=onetoone, dp_rate=0.4)
model_wE.to(device)

# define network
model_woE = SnnNetwork(IN_dim, hidden_dim, n_classes, is_adapt=adap_neuron, one_to_one=onetoone, dp_rate=0.4)
model_woE.to(device)

# load different models
exp_dir_wE = '/home/lucy/spikingPC/results/Mar-21-2023/fptt_ener0.5_taux2_scaledinput05_dt0.5_outsoftmax/'
saved_dict1 = model_result_dict_load(exp_dir_wE + 'onelayer_rec_best.pth.tar')

model_wE.load_state_dict(saved_dict1['state_dict'])

exp_dir_woE = '/home/lucy/spikingPC/results/Mar-21-2023/fptt_ener0.0_taux2_scaledinput05_dt0.5_outsoftmax/'
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
weights = [x for x in param_names_wE if ('weight'in x) and ('bias' not in x)]

fig, axes = plt.subplots(1, len(weights), figsize=(len(weights)*3, 3))
for i in range(len(weights)):
    axes[i].hist(param_dict_wE[weights[i]].flatten(), label='w E', histtype='step')
    axes[i].hist(param_dict_woE[weights[i]].flatten(), label='w/o E', histtype='step')
    axes[i].legend()
    axes[i].set_title(weights[i])

plt.tight_layout()
plt.show()


# %%
taus = [x for x in param_names_wE if ('tau'in x)]

fig, axes = plt.subplots(1, len(taus), figsize=(len(taus)*3, 3))
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
hiddens_wE, preds_wE, images_all, _ = get_all_analysis_data(model_wE, test_loader, device, IN_dim, T, batch_no=batches, occlusion_p=occlusion_p)
hiddens_woE, preds_woE, _, _ = get_all_analysis_data(model_woE, test_loader, device, IN_dim, T, batch_no=batches, occlusion_p=occlusion_p)

n_samples = len(preds_wE)
target_all = testdata.targets.data[:n_samples]

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
        pos = axes[0, i].imshow((param_dict_wE['layer2to1.weight'] @ s_wE2[i].mean(axis=0)).reshape(28, 28), vmin=-max, vmax=max, cmap='vlag')
        fig.colorbar(pos, ax=axes[0, i])
        axes[0, i].set_title('w E mean l2->l1 occlusion')

        max = np.max((param_dict_woE['layer2to1.weight'] @ s_woE2[i].mean(axis=0)))
        pos = axes[1, i].imshow((param_dict_woE['layer2to1.weight'] @ s_woE2[i].mean(axis=0)).reshape(28, 28), vmin=-max, vmax=max, cmap='vlag')
        fig.colorbar(pos, ax=axes[1, i])
        axes[1, i].set_title('w/o E mean l2->l1 spk occlusion')
    plt.show()


# %%
# look at why layer 1 neurons are not spiking 

sns.heatmap((s_wE[2].mean(axis=0)<1/T).reshape(28, 28))











# %%
###############################
# get enery signal for normal sequence 
###############################

# get mean curr by class to compare between groups 
a_curr_l1_wE = get_states(hiddens_wE, 2, hidden_dim[0], batch_size, num_samples=n_samples, T=T)
a_curr_l1_woE = get_states(hiddens_woE, 2, hidden_dim[0], batch_size, num_samples=n_samples, T=T)

soma_l1_wE = get_states(hiddens_wE, 0, hidden_dim[0], batch_size, num_samples=n_samples, T=T)
soma_l1_woE = get_states(hiddens_woE, 0, hidden_dim[0], batch_size, num_samples=n_samples, T=T)

error_l1_wE = ((a_curr_l1_wE - soma_l1_wE) ** 2)
error_l1_woE = ((a_curr_l1_woE - soma_l1_woE) ** 2)


# get mean curr by class to compare between groups 
a_curr_l2_wE = get_states(hiddens_wE, 6, hidden_dim[1], batch_size, num_samples=n_samples, T=T)
a_curr_l2_woE = get_states(hiddens_woE, 6, hidden_dim[1], batch_size, num_samples=n_samples, T=T)

soma_l2_wE = get_states(hiddens_wE, 4, hidden_dim[1], batch_size, num_samples=n_samples, T=T)
soma_l2_woE = get_states(hiddens_woE, 4, hidden_dim[1], batch_size, num_samples=n_samples, T=T)

error_l2_wE = (a_curr_l2_wE - soma_l2_wE)**2
error_l2_woE = (a_curr_l2_woE - soma_l2_woE) **2

# %%
###############################
# get enery signal for stimulus change sequence 
###############################

# plot energy consumption in network with two consecutive images
zeros = (target_all==3).nonzero(as_tuple=True)[0]
ones = (target_all==4).nonzero(as_tuple=True)[0]
if len(zeros) > len(ones):
    zeros = zeros[:len(ones)]
else: 
    ones = ones[:len(zeros)]

conti_samplesize = len(zeros)
print(conti_samplesize)

sample_image_nos = [zeros, ones]
print(target_all[sample_image_nos[0]])
print(target_all[sample_image_nos[1]])
continuous_seq_hiddens_E = []
continuous_seq_hiddens_nE = []

with torch.no_grad():
    model_wE.eval()
    model_woE.eval()

    hidden_i = model_wE.init_hidden(images_all[sample_image_nos[0], :, :].view(-1, IN_dim).size(0))

    log_sm_E1, hidden1_E = model_wE.inference(images_all[sample_image_nos[0], :, :].view(-1, IN_dim).to(device), hidden_i, int(T/4))
    log_sm_nE1, hidden1_nE = model_woE.inference(images_all[sample_image_nos[0], :, :].view(-1, IN_dim).to(device), hidden_i, int(T/4))
    continuous_seq_hiddens_E += hidden1_E
    continuous_seq_hiddens_nE += hidden1_nE

    # present second stimulus without reset
    # hidden1[-1] = model.init_hidden(images[sample_image_nos[0], :, :].view(-1, IN_dim).size(0))
    log_sm_E2, hidden2_E = model_wE.inference((images_all[sample_image_nos[1], :, :]).view(-1, IN_dim).to(device), hidden1_E[-1], (T-int(T/4)))
    log_sm_nE2, hidden2_nE = model_woE.inference((images_all[sample_image_nos[1], :, :]).view(-1, IN_dim).to(device), hidden1_nE[-1], (T-int(T/4)))

    continuous_seq_hiddens_E += hidden2_E
    continuous_seq_hiddens_nE += hidden2_nE
torch.cuda.empty_cache()

### layer2 
## with E model 
conti_a_curr_l2_E = get_states([continuous_seq_hiddens_E], 6, hidden_dim[1], batch_size=conti_samplesize, num_samples=conti_samplesize, T=T).squeeze()
conti_soma_l2_E = get_states([continuous_seq_hiddens_E], 4, hidden_dim[1], batch_size=conti_samplesize, num_samples=conti_samplesize, T=T).squeeze()

conti_error_l2_E = (conti_a_curr_l2_E - conti_soma_l2_E)**2

# w/o E model
conti_a_curr_l2_nE = get_states([continuous_seq_hiddens_nE], 6, hidden_dim[1], batch_size=conti_samplesize, num_samples=conti_samplesize, T=T).squeeze()
conti_soma_l2_nE = get_states([continuous_seq_hiddens_nE], 4, hidden_dim[1], batch_size=conti_samplesize, num_samples=conti_samplesize, T=T).squeeze()

conti_error_l2_nE = (conti_a_curr_l2_nE - conti_soma_l2_nE)**2


# %%
### layer1
## w E model 
conti_a_curr_l1_E = get_states([continuous_seq_hiddens_E], 2, hidden_dim[0], batch_size=conti_samplesize, num_samples=conti_samplesize, T=T).squeeze()
conti_soma_l1_E = get_states([continuous_seq_hiddens_E], 0, hidden_dim[0], batch_size=conti_samplesize, num_samples=conti_samplesize, T=T).squeeze()

conti_error_l1_E = (conti_a_curr_l1_E - conti_soma_l1_E)**2

# w/o E model
conti_a_curr_l1_nE = get_states([continuous_seq_hiddens_nE], 2, hidden_dim[0], batch_size=conti_samplesize, num_samples=conti_samplesize, T=T).squeeze()
conti_soma_l1_nE = get_states([continuous_seq_hiddens_nE], 0, hidden_dim[0], batch_size=conti_samplesize, num_samples=conti_samplesize, T=T).squeeze()

conti_error_l1_nE = (conti_a_curr_l1_nE - conti_soma_l1_nE)**2

# %%
###############################
# verify variance distribution 
###############################
def variance(data):
    x_mean = data.mean()
    deviation = [(x-x_mean)**2 for x in data]
    return sum(deviation) / len(deviation)

var_E_l1_conti = [variance(conti_error_l1_E.mean(axis=0)[:, i]) for i in range(784)]
var_nE_l1_conti = [variance(conti_error_l1_nE.mean(axis=0)[:, i]) for i in range(784)]

sns.histplot(np.sqrt(var_E_l1_conti), element='step', label='w E', binwidth=0.1, kde=True)
sns.histplot(np.sqrt(var_nE_l1_conti), element='step', label='w/o E', binwidth=0.1, kde=True)
plt.title('std of epislon per layer1 neuron')
plt.legend()
plt.show()

# %%
var_E_l2_conti = [variance(conti_error_l2_E.mean(axis=0)[int(T/5):, i]) for i in range(256)]
var_nE_l2_conti = [variance(conti_error_l2_nE.mean(axis=0)[int(T/5):, i]) for i in range(256)]


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
df_E_l1 = pd.DataFrame(np.vstack((error_l1_wE[target_all == 3].mean(axis=0), conti_error_l1_E.mean(axis=0))), \
    columns = ['neuron%i' %i for i in range(hidden_dim[0])])
df_E_l1['t'] = np.concatenate((np.arange(T), np.arange(T)))
df_E_l1['condition'] = ['normal seq'] * T + ['stim change seq'] * T
df_E_l1 = pd.melt(df_E_l1, id_vars=['t', 'condition'], value_vars=['neuron%i' %i for i in range(hidden_dim[0])], var_name='neuron index', value_name='epsilon')
df_E_l1['high var'] = 0
for i in range(hidden_dim[0]):
    df_E_l1['high var'][df_E_l1['neuron index'] == 'neuron%i' %i] = high_var_l1_E_map[i]

# %%
# w/o E layer1
df_nE_l1 = pd.DataFrame(np.vstack((error_l1_woE[target_all == 3].mean(axis=0), conti_error_l1_nE.mean(axis=0))), \
    columns = ['neuron%i' %i for i in range(hidden_dim[0])])
df_nE_l1['t'] = np.concatenate((np.arange(T), np.arange(T)))
df_nE_l1['condition'] = ['normal seq'] * T + ['stim change seq'] * T
df_nE_l1 = pd.melt(df_nE_l1, id_vars=['t', 'condition'], value_vars=['neuron%i' %i for i in range(hidden_dim[0])], var_name='neuron index', value_name='epsilon')
df_nE_l1['high var'] = 0
for i in range(hidden_dim[0]):
    df_nE_l1['high var'][df_nE_l1['neuron index'] == 'neuron%i' %i] = high_var_l1_nE_map[i]

# %%
# plot
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(7, 3))
sns.lineplot(df_E_l1, x='t', y='epsilon', hue='condition', style='high var', 
    estimator='mean', errorbar='ci', ax=axes[0])
axes[0].set_title('w E model')
axes[0].get_legend().remove()

sns.lineplot(df_nE_l1, x='t', y='epsilon', hue='condition', style='high var', 
    estimator='mean', errorbar='ci', ax=axes[1])
axes[1].set_title('w/o E model')
axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()

# %%
###############################
# layer2
###############################
high_var_l2_E_map = var_E_l2_conti > np.quantile(var_E_l2_conti, 0.5)
plt.imshow(high_var_l2_E_map.reshape(16, 16))

# %%
high_var_l2_nE_map = var_nE_l2_conti > np.quantile(var_nE_l2_conti, 0.5)
plt.imshow(high_var_l2_nE_map.reshape(16, 16))

# %%
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

# %%
# model with E 
df_E_l2 = pd.DataFrame(np.vstack((error_l2_wE[target_all == 3].mean(axis=0), conti_error_l2_E.mean(axis=0))), \
    columns = ['neuron%i' %i for i in range(hidden_dim[1])])
df_E_l2['t'] = np.concatenate((np.arange(T), np.arange(T)))
df_E_l2['condition'] = ['normal seq'] * T + ['stim change seq'] * T
df_E_l2 = pd.melt(df_E_l2, id_vars=['t', 'condition'], value_vars=['neuron%i' %i for i in range(hidden_dim[1])], var_name='neuron index', value_name='epsilon')
df_E_l2['high var'] = 0
for i in range(hidden_dim[1]):
    df_E_l2['high var'][df_E_l2['neuron index'] == 'neuron%i' %i] = high_var_l2_E_map[i]

# %%
# w/o E layer1
df_nE_l2 = pd.DataFrame(np.vstack((error_l2_woE[target_all == 3].mean(axis=0), conti_error_l2_nE.mean(axis=0))), \
    columns = ['neuron%i' %i for i in range(hidden_dim[1])])
df_nE_l2['t'] = np.concatenate((np.arange(T), np.arange(T)))
df_nE_l2['condition'] = ['normal seq'] * T + ['stim change seq'] * T
df_nE_l2 = pd.melt(df_nE_l2, id_vars=['t', 'condition'], value_vars=['neuron%i' %i for i in range(hidden_dim[1])], var_name='neuron index', value_name='epsilon')
df_nE_l2['high var'] = 0
for i in range(hidden_dim[1]):
    df_nE_l2['high var'][df_nE_l2['neuron index'] == 'neuron%i' %i] = high_var_l2_nE_map[i]

# %%
# plot
fig, axes = plt.subplots(1, 2, figsize=(7, 3))
sns.lineplot(df_E_l2, x='t', y='epsilon', hue='condition', style='high var', 
    estimator='mean', errorbar='ci', ax=axes[0])
axes[0].set_title('w E model layer2')
axes[0].get_legend().remove()

sns.lineplot(df_nE_l2, x='t', y='epsilon', hue='condition', style='high var', 
    estimator='mean', errorbar='ci', ax=axes[1])
axes[1].set_title('w/o E model layer2')
axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()

plt.show()

# %%
sns.lineplot(acc_E, label='w E')
sns.lineplot(acc_nE, label='w/o E')
plt.title('acc in stim change sequence')
plt.legend()
plt.show()

# %%
##############################################################
# test generative capacity of network with clamping 
##############################################################
dig = 7
no_input = torch.zeros((1, hidden_dim[0])).to(device)
with torch.no_grad():
    model_wE.eval()
    model_woE.eval()

    hidden_i = model_wE.init_hidden(1)

    log_sm_E_gen, hidden_gen_E = model_wE.clamped_generate(dig, no_input, hidden_i, T*2, clamp_value=100)
    log_sm_nE_gen, hidden_gen_nE = model_woE.clamped_generate(dig, no_input, hidden_i, T*2, clamp_value=100)
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
pos = axes[0].imshow(spk_gen_l2_E.reshape(16, 16))
fig.colorbar(pos, ax=axes[0], shrink=0.5)
axes[0].set_title('w E mean spk l2 clamped generation')

pos = axes[1].imshow(spk_gen_l2_nE.reshape(16, 16))
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
out_E = get_states([hidden_gen_E], -1, 10, 1, T, num_samples=1).squeeze()
out_nE = get_states([hidden_gen_nE], -1, 10, 1, T, num_samples=1).squeeze()

print(out_E.shape)

plt.plot(out_nE)

# %%
fig, axes = plt.subplots(1, 2)
pos = axes[0].imshow((param_dict_wE['out2layer2.weight'] @ out_E.mean(axis=0)).reshape(16, 16))
fig.colorbar(pos, ax=axes[0], shrink=0.5)
axes[0].set_title('w E mean out > l2 clamped generation')

pos = axes[1].imshow((param_dict_woE['out2layer2.weight'] @ out_nE.mean(axis=0)).reshape(16, 16))
fig.colorbar(pos, ax=axes[1], shrink=0.5)
axes[1].set_title('w/o E mean out > l2 clamped generation')

plt.tight_layout()
plt.show()

# %%
#examine top down from each class 
gen_l2_reps_E = []
gen_l2_reps_nE = []


fig, axes = plt.subplots(2, 10, figsize=(35, 6))
for i in range(10):
    print(i)
    with torch.no_grad():
        model_wE.eval()
        model_woE.eval()

        hidden_i = model_wE.init_hidden(1)

        log_sm_E_gen, hidden_gen_E_ = model_wE.clamped_generate(i, no_input, hidden_i, T, clamp_value=100)
        log_sm_nE_gen, hidden_gen_nE_ = model_woE.clamped_generate(i, no_input, hidden_i, T, clamp_value=100)

    torch.cuda.empty_cache()

    spk_gen_l2_E_ = get_states([hidden_gen_E_], 5, hidden_dim[1], 1, T, num_samples=1).squeeze().mean(axis=0)
    spk_gen_l2_nE_ = get_states([hidden_gen_nE_], 5, hidden_dim[1], 1, T, num_samples=1).squeeze().mean(axis=0)

    gen_l2_reps_E.append(param_dict_wE['layer2to1.weight'] @ spk_gen_l2_E_)
    gen_l2_reps_nE.append(param_dict_woE['layer2to1.weight'] @ spk_gen_l2_nE_)

    max = np.max((param_dict_wE['layer2to1.weight'] @ spk_gen_l2_E_))
    pos = axes[0][i].imshow((param_dict_wE['layer2to1.weight'] @ spk_gen_l2_E_).reshape(28, 28), vmin=-max, vmax=max, cmap='vlag')
    fig.colorbar(pos, ax=axes[0][i], shrink=0.5)
    axes[0][i].set_title('w E mean l2 > l1 clamped class%i' %i)

    max = np.max((param_dict_woE['layer2to1.weight'] @ spk_gen_l2_nE_))
    pos = axes[1][i].imshow((param_dict_woE['layer2to1.weight'] @ spk_gen_l2_nE_).reshape(28, 28), vmin=-max, vmax=max, cmap='vlag')
    fig.colorbar(pos, ax=axes[1][i], shrink=0.5)
    axes[1][i].set_title('w/o E mean l2 > l1 clamped class%i' %i)

plt.show()

# %%
# how classifiable are the generated representations 
from scipy.spatial import distance
corr_E = []
corr_nE = []
for i in range(10):
    class_mean = images_all[target_all==i].mean(axis=0).flatten()
    corr_E.append(distance.cosine(class_mean, gen_l2_reps_E[i]))
    corr_nE.append(distance.cosine(class_mean, gen_l2_reps_nE[i]))

df = pd.DataFrame({
    'cosine dist': corr_E + corr_nE, 
    'class': np.concatenate((np.arange(10), np.arange(10))), 
    'model': ['w E'] * 10 + ['w/o E'] * 10
})

sns.barplot(df, x='class', y='cosine dist', hue='model')
plt.show()


# %%
##############################################################
# check what neurons are sensitive to vs projection from layer2 to apical tuft 
##############################################################
# essentially whether top down and bottom up weights are matching 
mean_zero = images_all[target_all==8][0]
plt.imshow(mean_zero)

# %%
input2layer2 = param_dict_wE['layer1to2.weight'] @ mean_zero.cpu().numpy().flatten()
input2layer2[input2layer2 < 0] = 0
max = np.max(input2layer2)

plt.imshow((param_dict_wE['layer2to1.weight'] @ input2layer2).reshape(28, 28), vmin=-max, vmax=max, cmap='vlag')


# %%
##############################################################
# 
############################################################## 














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
plt.title('layer 1 mean((a_curr - soma)**2) with stimulus change at t = %i' % int(T/2))
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
    axes[0, i].imshow(continuous_seq_hiddens_E[i*2][1].detach().cpu().numpy().reshape(28, 28))
    axes[0, i].axis('off')
    axes[0, i].set_title('w E')

    axes[1, i].imshow(continuous_seq_hiddens_nE[i*2][1].detach().cpu().numpy().reshape(28, 28))
    axes[1, i].axis('off')
    axes[1, i].set_title('w/o E')

# %%
# plot a_curr sequence 
fig, axes = plt.subplots(2, 30, figsize=(60, 3))
for i in range(T):
    pos1 = axes[0, i].imshow(continuous_seq_hiddens_E[i*2][2].detach().cpu().numpy().reshape(28, 28))
    axes[0, i].axis('off')
    fig.colorbar(pos1, ax=axes[0, i], shrink=0.5)
    axes[0, i].set_title('w E')

    pos2 = axes[1, i].imshow(continuous_seq_hiddens_nE[i*2][2].detach().cpu().numpy().reshape(28, 28))
    axes[1, i].axis('off')
    fig.colorbar(pos2, ax=axes[1, i], shrink=0.5)
    axes[1, i].set_title('w/o E')

# %%
# plot top down signal from layer2 to layer 1 
fig, axes = plt.subplots(2, 30, figsize=(60, 3))
for i in range(T):
    top_down_E = param_dict_wE['layer2to1.weight'] @ continuous_seq_hiddens_E[i*2][5][0].detach().cpu().numpy().squeeze()
    pos1 = axes[0, i].imshow(top_down_E.reshape(28, 28))
    axes[0, i].axis('off')
    fig.colorbar(pos1, ax=axes[0, i], shrink=0.5)
    axes[0, i].set_title('w E')

    top_down_nE = param_dict_woE['layer2to1.weight'] @ continuous_seq_hiddens_nE[i*2][5][0].detach().cpu().numpy().squeeze()
    pos2 = axes[1, i].imshow(top_down_nE.reshape(28, 28))
    fig.colorbar(pos2, ax=axes[1, i], shrink=0.5)
    axes[1, i].axis('off')
    axes[1, i].set_title('w/o E')

# %%
# plot error in layer1 a_curr - soma
fig, axes = plt.subplots(2, 30, figsize=(60, 3))
for i in range(T):
    error1 = continuous_seq_hiddens_E[i*2][2].detach().cpu().numpy().squeeze() - \
        continuous_seq_hiddens_E[i*2][0].detach().cpu().numpy().squeeze()
    pos1 = axes[0, i].imshow((error1 **2).reshape(28, 28), cmap='rocket')
    axes[0, i].axis('off')
    fig.colorbar(pos1, ax=axes[0, i], shrink=0.5)
    axes[0, i].set_title('w E')

    error2 = continuous_seq_hiddens_nE[i*2][2].detach().cpu().numpy().squeeze() - \
        continuous_seq_hiddens_nE[i*2][0].detach().cpu().numpy().squeeze()    
    pos2 = axes[1, i].imshow((error2**2).reshape(28, 28), cmap='rocket')
    fig.colorbar(pos2, ax=axes[1, i], shrink=0.5)
    axes[1, i].axis('off')
    axes[1, i].set_title('w/o E')

# %%

# %%


# %%
plt.plot((conti_error_l1_nE.mean(axis=0)[:, var<2] - error_l1_woE[sample_image_nos[0]].mean(axis=0)[:, var<2]))
plt.show()

# %%
