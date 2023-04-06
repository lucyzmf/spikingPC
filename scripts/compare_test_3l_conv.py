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

from conv_net import *
from conv_local import *
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
conv_adp = False

hidden_channels = [10, 10]
kernel_size = [3, 3]
stride = [1, 1]
paddings = [0, 0]
pooling = None

IN_dim = 784
T = 100  # sequence length, reading from the same image time_steps times
dp = 0.4
is_rec = [False, False]


exp_dir_wE = '/home/lucy/spikingPC/results/Apr-03-2023/fptt_ener0.1_taux2_dt0.5_exptau05_threeh_absloss/'
exp_dir_woE = '/home/lucy/spikingPC/results/Apr-03-2023/fptt_ener0_taux2_dt0.5_exptau05_threeh/'
conv_type = 'shared'  # or shared or local

# define network
if conv_type == 'local':
    model_wE = SnnLocalConvNet(IN_dim, hidden_channels, kernel_size, stride,
                               paddings, n_classes, is_adapt_conv=conv_adp,
                               dp_rate=dp, p_size=num_readout,
                               pooling=pooling, is_rec=is_rec)
    model_woE = SnnLocalConvNet(IN_dim, hidden_channels, kernel_size, stride,
                                paddings, n_classes, is_adapt_conv=conv_adp,
                                dp_rate=dp, p_size=num_readout,
                                pooling=pooling, is_rec=is_rec)
else:
    model_wE = SnnConvNet(IN_dim, hidden_channels, kernel_size, stride,
                          paddings, n_classes, is_adapt_conv=conv_adp,
                          dp_rate=dp, p_size=num_readout,
                          pooling=pooling, is_rec=is_rec)
    model_woE = SnnConvNet(IN_dim, hidden_channels, kernel_size, stride,
                           paddings, n_classes, is_adapt_conv=conv_adp,
                           dp_rate=dp, p_size=num_readout,
                           pooling=pooling, is_rec=is_rec)

model_wE.to(device)
model_woE.to(device)

# load different models
saved_dict1 = model_result_dict_load(exp_dir_wE + 'onelayer_rec_best.pth.tar')
model_wE.load_state_dict(saved_dict1['state_dict'])

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

# plt.style.use('seaborn-v0_8-deep')

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
# function for get state in convnet
def get_states_conv(hiddens_all_: list, idx: int, hidden_dims_: list, batch_size, T=20, num_samples=10000):
    """
    get a particular internal state depending on index passed to hidden
    :param hidden_dim_: list containing channel size, w, h
    :param T: total time steps
    :param hiddens_all_: list containing hidden states of all batch and time steps during inference
    :param idx: which index in h is taken out
    :return: np array containing desired states
    """
    all_states = []
    for batch_idx in range(len(hiddens_all_)):  # iterate over batch
        batch_ = []
        for t in range(T):
            seq_ = []
            for b in range(batch_size):
                seq_.append(hiddens_all_[batch_idx][t][idx][b].detach().cpu().numpy())
            seq_ = np.stack(seq_)
            batch_.append(seq_)
        batch_ = np.stack(batch_)
        all_states.append(batch_)

    all_states = np.stack(all_states)

    return all_states.transpose(0, 2, 1, 3, 4, 5).reshape(num_samples, T, hidden_dims_[0], hidden_dims_[1], hidden_dims_[2])

# %%
########################################################################################################################
#########################           test generative capacity of network with clamping            #######################
########################################################################################################################
dig = 0
no_input = torch.zeros((1, IN_dim)).to(device)
with torch.no_grad():
    model_wE.eval()
    model_woE.eval()

    hidden_i = model_wE.init_hidden(1)

    log_sm_E_gen, hidden_gen_E = model_wE.clamped_generate(dig, no_input, hidden_i, T * 2, clamp_value=10)
    log_sm_nE_gen, hidden_gen_nE = model_woE.clamped_generate(dig, no_input, hidden_i, T * 2, clamp_value=10)
torch.cuda.empty_cache()

# %%
spk_gen_l1_E = get_states_conv([hidden_gen_E], 1, model_wE.conv1.output_shape, 1, T, num_samples=1).squeeze().mean(axis=0)
spk_gen_l1_nE = get_states_conv([hidden_gen_nE], 1, model_woE.conv1.output_shape, 1, T, num_samples=1).squeeze().mean(axis=0)

fig, axes = plt.subplots(1, 2)
pos = axes[0].imshow(spk_gen_l1_E.mean(axis=0)) # avg over channel
fig.colorbar(pos, ax=axes[0], shrink=0.5)
axes[0].set_title('w E mean spk l1 clamped generation')

pos = axes[1].imshow(spk_gen_l1_nE.mean(axis=0))
fig.colorbar(pos, ax=axes[1], shrink=0.5)
axes[1].set_title('w/o E mean spk l1 clamped generation')

plt.tight_layout()
plt.show()

# %%
spk_gen_l2_E = get_states_conv([hidden_gen_E], 5, model_wE.conv2.output_shape, 1, T, num_samples=1).squeeze().mean(axis=0)
spk_gen_l2_nE = get_states_conv([hidden_gen_nE], 5, model_wE.conv2.output_shape, 1, T, num_samples=1).squeeze().mean(axis=0)

fig, axes = plt.subplots(1, 2)
pos = axes[0].imshow(spk_gen_l2_E.mean(axis=0)) # avg over channel
fig.colorbar(pos, ax=axes[0], shrink=0.5)
axes[0].set_title('w E mean spk l2 clamped generation')

pos = axes[1].imshow(spk_gen_l2_nE.mean(axis=0))
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
def create_empty_array(output_shape_, nclass=10):
    return np.zeros(nclass, output_shape_[0], output_shape_[1], output_shape_[2])


l1_norm_E = create_empty_array(model_wE.conv1.output_shape)
l1_norm_nE = create_empty_array(model_woE.conv1.output_shape)

l2_norm_E = create_empty_array(model_wE.conv2.output_shape)
l3_norm_E = np.zeros((10, model_wE.pop_enc.hidden_dim))

l2_norm_nE = create_empty_array(model_woE.conv2.output_shape)
l3_norm_nE = np.zeros((10, model_woE.pop_enc.hidden_dim))

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

        l1_E = get_states_conv([h_E], 1, model_wE.conv1.output_shape, batch_size, T=60, num_samples=batch_size)
        l1_nE = get_states([h_nE], 1,model_woE.conv1.output_shape, batch_size, T=60, num_samples=batch_size)

        l2_E = get_states_conv([h_E], 5, model_wE.conv2.output_shape, batch_size, T=60, num_samples=batch_size)
        l2_nE = get_states_conv([h_nE], 5, model_woE.conv2.output_shape, batch_size, T=60, num_samples=batch_size)

        l3_E = get_states([h_E], 9, model_wE.pop_enc.hidden_dim, batch_size, T=60, num_samples=batch_size)
        l3_nE = get_states([h_nE], 9, model_woE.pop_enc.hidden_dim, batch_size, T=60, num_samples=batch_size)

        for i in range(n_classes):
            l1_norm_E[i] += l1_E[target.cpu() == i].mean(axis=1).sum(axis=0)  # avg over t and sum over all samples
            l1_norm_nE[i] += l1_nE[target.cpu() == i].mean(axis=1).sum(axis=0)

            l2_norm_E[i] += l2_E[target.cpu() == i].mean(axis=1).sum(axis=0)  # avg over t and sum over all samples
            l2_norm_nE[i] += l2_nE[target.cpu() == i].mean(axis=1).sum(axis=0)

            l3_norm_E[i] += l3_E[target.cpu() == i].mean(axis=1).sum(axis=0)  # avg over t and sum over all samples
            l3_norm_nE[i] += l3_nE[target.cpu() == i].mean(axis=1).sum(axis=0)

    torch.cuda.empty_cache()

# avg all samples
l1_norm_E = l1_norm_E / len(test_loader)
l1_norm_nE = l1_norm_nE / len(test_loader)

l2_norm_E = l2_norm_E / len(test_loader)
l2_norm_nE = l2_norm_nE / len(test_loader)

l3_norm_E = l3_norm_E / len(test_loader)
l3_norm_nE = l3_norm_nE / len(test_loader)

# %%
# clamped condition
l1_clamp_E = create_empty_array(model_wE.conv1.output_shape)
l1_clamp_nE = create_empty_array(model_woE.conv1.output_shape)

l2_clamp_E = create_empty_array(model_wE.conv2.output_shape)
l3_clamp_E = np.zeros((10, model_wE.pop_enc.hidden_dim))

l2_clamp_nE = create_empty_array(model_woE.conv2.output_shape)
l3_clamp_nE = np.zeros((10, model_woE.pop_enc.hidden_dim))

for i in range(10):
    print(i)
    with torch.no_grad():
        model_wE.eval()
        model_woE.eval()

        hidden_i = model_wE.init_hidden(1)

        _, hidden_gen_E_ = model_wE.clamped_generate(i, no_input, hidden_i, T * 2, clamp_value=10)
        _, hidden_gen_nE_ = model_woE.clamped_generate(i, no_input, hidden_i, T * 2, clamp_value=10)

        #
        l1_E = get_states_conv([hidden_gen_E_], 1, model_wE.conv1.output_shape, 1, T * 2, num_samples=1)
        l1_nE = get_states_conv([hidden_gen_nE_], 1, model_woE.conv1.output_shape, 1, T * 2, num_samples=1)

        # get gen
        l2_E = get_states_conv([hidden_gen_E_], 5, model_wE.conv2.output_shape, 1, T * 2, num_samples=1)
        l2_nE = get_states_conv([hidden_gen_nE_], 5, model_woE.conv2.output_shape, 1, T * 2, num_samples=1)

        l3_E = get_states([hidden_gen_E_], 9, model_wE.pop_enc.hidden_dim, 1, T * 2, num_samples=1)
        l3_nE = get_states([hidden_gen_nE_], 9, model_woE.pop_enc.hidden_dim, 1, T * 2, num_samples=1)

        l1_clamp_E[i] += np.squeeze(l1_E.mean(axis=1))
        l1_clamp_nE[i] += np.squeeze(l1_nE.mean(axis=1))

        l2_clamp_E[i] += np.squeeze(l2_E.mean(axis=1))
        l2_clamp_nE[i] += np.squeeze(l2_nE.mean(axis=1))

        l3_clamp_E[i] += np.squeeze(l3_E.mean(axis=1))
        l3_clamp_nE[i] += np.squeeze(l3_nE.mean(axis=1))

    torch.cuda.empty_cache()

# %%
# how classifiable are the generated representations
from scipy.spatial import distance

dist_l2_E = []
dist_l2_nE = []

dist_l3_E = []
dist_l3_nE = []

for i in range(10):
    dist_l2_E.append(distance.cosine(l2_clamp_E[i].flatten(), l2_norm_E[i].flatten()))
    dist_l2_nE.append(distance.cosine(l2_clamp_nE[i].flatten(), l2_norm_nE[i].flatten()))

    dist_l3_E.append(distance.cosine(l3_clamp_E[i].flatten(), l3_norm_E[i].flatten()))
    dist_l3_nE.append(distance.cosine(l3_clamp_nE[i].flatten(), l3_norm_nE[i].flatten()))

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

pair_dist_E_l3 = pairwise_distances(l3_clamp_E.reshape(10, -1), l3_norm_E.reshape(10, -1), metric='cosine')
pair_dist_nE_l3 = pairwise_distances(l3_clamp_nE.reshape(10, -1), l3_norm_nE.reshape(10, -1), metric='cosine')

pair_dist_E_l2 = pairwise_distances(l2_clamp_E.reshape(10, -1), l2_norm_E.reshape(10, -1), metric='cosine')
pair_dist_nE_l2 = pairwise_distances(l2_clamp_nE.reshape(10, -1), l2_norm_nE.reshape(10, -1), metric='cosine')

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




