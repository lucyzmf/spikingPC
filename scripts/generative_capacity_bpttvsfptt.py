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
model_wE_fptt = SnnNetwork2Layer(IN_dim, hidden_dim, n_classes, is_adapt=adap_neuron, one_to_one=onetoone, dp_rate=dp,
                            is_rec=is_rec)
model_wE_fptt.to(device)

# define network
model_woE_fptt = SnnNetwork2Layer(IN_dim, hidden_dim, n_classes, is_adapt=adap_neuron, one_to_one=onetoone, dp_rate=dp,
                             is_rec=is_rec)
model_woE_fptt.to(device)

# load different models
exp_dir_wE_fptt = '/home/lucy/spikingPC/results/Apr-17-2023/fptt_ener0.05_taux2_dt0.5_exptau05_absloss_bias025/'
saved_dict1 = model_result_dict_load(exp_dir_wE_fptt + 'onelayer_rec_best.pth.tar')

model_wE_fptt.load_state_dict(saved_dict1['state_dict'])

exp_dir_woE_fptt = '/home/lucy/spikingPC/results/Apr-17-2023/fptt_ener0.0_taux2_dt0.5_exptau05_absloss_bias025/'
saved_dict2 = model_result_dict_load(exp_dir_woE_fptt + 'onelayer_rec_best.pth.tar')

model_woE_fptt.load_state_dict(saved_dict2['state_dict'])

# %%
# load bptt models 
model_wE_bptt = SnnNetwork2Layer(IN_dim, hidden_dim, n_classes, is_adapt=adap_neuron, one_to_one=onetoone, dp_rate=dp,
                            is_rec=is_rec)
model_wE_bptt.to(device)

# define network
model_woE_bptt = SnnNetwork2Layer(IN_dim, hidden_dim, n_classes, is_adapt=adap_neuron, one_to_one=onetoone, dp_rate=dp,
                                is_rec=is_rec)
model_woE_bptt.to(device)

# load different models
exp_dir_wE_bptt = '/home/lucy/spikingPC/results/Apr-19-2023/bptt_ener0.05_taux2_dt0.5_exptau05_absloss_bias0_scheduler1025/'
saved_dict1 = model_result_dict_load(exp_dir_wE_bptt + 'onelayer_rec_best.pth.tar')

model_wE_bptt.load_state_dict(saved_dict1['state_dict'])

exp_dir_woE_bptt = '/home/lucy/spikingPC/results/Apr-19-2023/bptt_ener0.0_taux2_dt0.5_exptau05_absloss_bias0_scheduler1025/'
saved_dict2 = model_result_dict_load(exp_dir_woE_bptt + 'onelayer_rec_best.pth.tar')

model_woE_bptt.load_state_dict(saved_dict2['state_dict'])

# %%
# get params and put into dict
def get_params(model):
    param_names_wE = []
    param_dict_wE = {}
    for name, param in model_wE_fptt.named_parameters():
        if param.requires_grad:
            param_names_wE.append(name)
            param_dict_wE[name] = param.detach().cpu().numpy()

    print(param_names_wE)
    return param_names_wE, param_dict_wE

param_name_E_fptt, param_dict_E_fptt = get_params(model_wE_fptt)
param_name_nE_fptt, param_dict_nE_fptt = get_params(model_woE_fptt)

param_name_E_bptt, param_dict_E_bptt = get_params(model_wE_bptt)
param_name_nE_bptt, param_dict_nE_bptt = get_params(model_woE_bptt)

# %%
# plot one sample image 

# get one sample image
sample = testdata[0][0]
sample = sample.unsqueeze(0)

# plot image
fig, ax = plt.subplots()
ax.imshow(sample.squeeze().numpy())
ax.axis('off')
plt.show()
# %%
##############################################################
# test generative capacity of network with clamping + noise
##############################################################
dig = 0
no_input = torch.zeros((1, IN_dim)).to(device)
with torch.no_grad():
    model_wE_fptt.eval()
    model_woE_fptt.eval()

    hidden_i = model_wE_fptt.init_hidden(1)

    log_sm_E_gen, hidden_gen_E = model_wE_fptt.clamped_generate(dig, no_input, hidden_i, T * 2, clamp_value=1)
    log_sm_nE_gen, hidden_gen_nE = model_woE_fptt.clamped_generate(dig, no_input, hidden_i, T * 2, clamp_value=1)
torch.cuda.empty_cache()

# %%
spk_gen_l1_E = get_states([hidden_gen_E], 1, hidden_dim[0], 1, T, num_samples=1).squeeze().mean(axis=0)
spk_gen_l1_nE = get_states([hidden_gen_nE], 1, hidden_dim[0], 1, T, num_samples=1).squeeze().mean(axis=0)

fig, axes = plt.subplots(1, 2)
pos = axes[0].imshow(spk_gen_l1_E.reshape(20, -1))
fig.colorbar(pos, ax=axes[0], shrink=0.5)
axes[0].set_title('w E mean spk l1 clamped generation')

pos = axes[1].imshow(spk_gen_l1_nE.reshape(20, -1))
fig.colorbar(pos, ax=axes[1], shrink=0.5)
axes[1].set_title('w/o E mean spk l1 clamped generation')

plt.tight_layout()
plt.show()

# %%
spk_gen_l2_E = get_states([hidden_gen_E], 5, hidden_dim[1], 1, T, num_samples=1).squeeze().mean(axis=0)
spk_gen_l2_nE = get_states([hidden_gen_nE], 5, hidden_dim[1], 1, T, num_samples=1).squeeze().mean(axis=0)

fig, axes = plt.subplots(2, 1)
pos = axes[0].imshow(spk_gen_l2_E.reshape(20, -1))
axes[0].axis('off')
fig.colorbar(pos, ax=axes[0], shrink=0.5)

axes[0].set_title('w E mean spk L2 (clamp)')

pos = axes[1].imshow(spk_gen_l2_nE.reshape(20, -1))
axes[1].axis('off')
fig.colorbar(pos, ax=axes[1], shrink=0.5)
axes[1].set_title('w/o E mean spk l2 (clamp)')

plt.tight_layout()
plt.show()


# %%
##############################################################
# test generative capacity of network with clamping
##############################################################
# compute mean l2, l3 reps from E and nE model with clamped mode 
l1_norm_E_fptt, l1_norm_nE_fptt, l1_norm_E_bptt, l1_norm_nE_bptt = [np.zeros((10, hidden_dim[0])) for i in range(4)]
l2_norm_E_fptt, l2_norm_nE_fptt, l2_norm_E_bptt, l2_norm_nE_bptt = [np.zeros((10, hidden_dim[1])) for i in range(4)]
l3_norm_E_fptt, l3_norm_nE_fptt, l3_norm_E_bptt, l3_norm_nE_bptt = [np.zeros((10, hidden_dim[2])) for i in range(4)]

def get_reps(h_e, h_ne):
    l1_E = get_states([h_e], 1, hidden_dim[0], batch_size, T=T, num_samples=batch_size)
    l1_nE = get_states([h_ne], 1, hidden_dim[0], batch_size, T=T, num_samples=batch_size)

    l2_E = get_states([h_e], 5, hidden_dim[1], batch_size, T=T, num_samples=batch_size)
    l2_nE = get_states([h_ne], 5, hidden_dim[1], batch_size, T=T, num_samples=batch_size)

    l3_E = get_states([h_e], 9, hidden_dim[2], batch_size, T=T, num_samples=batch_size)
    l3_nE = get_states([h_ne], 9, hidden_dim[2], batch_size, T=T, num_samples=batch_size)

    return l1_E, l1_nE, l2_E, l2_nE, l3_E, l3_nE


def log_reps(log_list, reps_list):
    """log and rep list follow the same order by layer, E then no E for each layer """
    for i in range(n_classes):
        for j in range(len(log_list)):
            log_list[j][i] += reps_list[j][target.cpu() == i].mean(axis=1).sum(axis=0)
    
    return log_list

def get_from_model(models, hidden, log_list):
    _, h_E = models[0].inference(data, hidden, T)
    _, h_nE = models[1].inference(data, hidden, T)

    l1_E, l1_nE, l2_E, l2_nE, l3_E, l3_nE = get_reps(h_E, h_nE)

    l1_norm_E, l1_norm_nE, l2_norm_E, l2_norm_nE, l3_norm_E, l3_norm_nE = log_reps(log_list, 
                                                                                   [l1_E, l1_nE, l2_E, l2_nE, l3_E, l3_nE])
    
    return l1_norm_E, l1_norm_nE, l2_norm_E, l2_norm_nE, l3_norm_E, l3_norm_nE

# %%
# test loader with all images
test_loader_all = torch.utils.data.DataLoader(testdata, batch_size=batch_size, shuffle=False, num_workers=2)

# get means from normal condition
for i, (data, target) in enumerate(test_loader_all):
    data, target = data.to(device), target.to(device)
    data = data.view(-1, model_wE_fptt.in_dim)

    with torch.no_grad():
        model_wE_fptt.eval()
        model_woE_fptt.eval()

        model_wE_bptt.eval()
        model_woE_bptt.eval()

        hidden = model_wE_fptt.init_hidden(data.size(0))

        l1_norm_E_fptt, l1_norm_nE_fptt, l2_norm_E_fptt, l2_norm_nE_fptt, \
            l3_norm_E_fptt, l3_norm_nE_fptt = get_from_model([model_wE_fptt, model_woE_fptt], hidden, 
                                                             [l1_norm_E_fptt, l1_norm_nE_fptt, l2_norm_E_fptt, l2_norm_nE_fptt,
                                                              l3_norm_E_fptt, l3_norm_nE_fptt])
        
        l1_norm_E_bptt, l1_norm_nE_bptt, l2_norm_E_bptt, l2_norm_nE_bptt, \
            l3_norm_E_bptt, l3_norm_nE_bptt = get_from_model([model_wE_bptt, model_woE_bptt], hidden, 
                                                             [l1_norm_E_bptt, l1_norm_nE_bptt, l2_norm_E_bptt, l2_norm_nE_bptt,
                                                              l3_norm_E_bptt, l3_norm_nE_bptt])

    torch.cuda.empty_cache()

# create iterator with all norm reps 
for arr in [l1_norm_E_fptt, l1_norm_nE_fptt, l2_norm_E_fptt, l2_norm_nE_fptt, l3_norm_E_fptt, l3_norm_nE_fptt, \
    l1_norm_E_bptt, l1_norm_nE_bptt, l2_norm_E_bptt, l2_norm_nE_bptt, l3_norm_E_bptt, l3_norm_nE_bptt]:
    arr /= len(testdata.data)


# %%
# clamped condition
no_input = torch.zeros((1, IN_dim)).to(device)
clamp_T = T * 5


l1_clamp_E_fptt, l1_clamp_nE_fptt, l1_clamp_E_bptt, l1_clamp_nE_bptt = [np.zeros((10, hidden_dim[0])) for i in range(4)]
l2_clamp_E_fptt, l2_clamp_nE_fptt, l2_clamp_E_bptt, l2_clamp_nE_bptt = [np.zeros((10, hidden_dim[1])) for i in range(4)]
l3_clamp_E_fptt, l3_clamp_nE_fptt, l3_clamp_E_bptt, l3_clamp_nE_bptt = [np.zeros((10, hidden_dim[2])) for i in range(4)]

def model_clamp(model, clamp_class, log_list, i):
    hidden_i = model.init_hidden(1)

    _, h_ = model.clamped_generate(clamp_class, no_input, hidden_i, clamp_T, clamp_value=1)

    l1 = get_states([h_], 1, hidden_dim[0], 1, clamp_T, num_samples=1)
    l2 = get_states([h_], 5, hidden_dim[1], 1, clamp_T, num_samples=1)
    l3 = get_states([h_], 9, hidden_dim[2], 1, clamp_T, num_samples=1)

    log_list[0][i] += np.squeeze(l1.mean(axis=1))
    log_list[1][i] += np.squeeze(l2.mean(axis=1))
    log_list[2][i] += np.squeeze(l3.mean(axis=1))

    return log_list
    

for i in range(10):
    print(i)
    with torch.no_grad():
        model_wE_fptt.eval()
        model_woE_fptt.eval()

        model_wE_bptt.eval()
        model_woE_bptt.eval()


        l1_clamp_E_fptt, l2_clamp_E_fptt, l3_clamp_E_fptt = model_clamp(model_wE_fptt, i,
                                                                        [l1_clamp_E_fptt, l2_clamp_E_fptt, l3_clamp_E_fptt], i)
        l1_clamp_nE_fptt, l2_clamp_nE_fptt, l3_clamp_nE_fptt = model_clamp(model_woE_fptt, i,
                                                                        [l1_clamp_nE_fptt, l2_clamp_nE_fptt, l3_clamp_nE_fptt], i)
        
        l1_clamp_E_bptt, l2_clamp_E_bptt, l3_clamp_E_bptt = model_clamp(model_wE_bptt, i,
                                                                        [l1_clamp_E_bptt, l2_clamp_E_bptt, l3_clamp_E_bptt], i)
        l1_clamp_nE_bptt, l2_clamp_nE_bptt, l3_clamp_nE_bptt = model_clamp(model_woE_bptt, i,
                                                                           [l1_clamp_nE_bptt, l2_clamp_nE_bptt, l3_clamp_nE_bptt], i)

    torch.cuda.empty_cache()


# %%
# how classifiable are the generated representations 
from scipy.spatial import distance

dist_l2_E = []
dist_l2_nE = []

dist_l3_E = []
dist_l3_nE = []

for i in range(10):
    dist_l2_E.append(distance.cosine(l2_clamp_E_fptt[i], l2_norm_E_fptt[i]))
    dist_l2_nE.append(distance.cosine(l2_clamp_nE_fptt[i], l2_norm_nE_fptt[i]))

    dist_l3_E.append(distance.cosine(l3_clamp_E_fptt[i], l3_norm_E_fptt[i]))
    dist_l3_nE.append(distance.cosine(l3_clamp_nE_fptt[i], l3_norm_nE_fptt[i]))

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

pair_dist_E_l3_b = pairwise_distances(l3_clamp_E_bptt, l3_norm_E_bptt, metric='cosine')
pair_dist_E_l3_f = pairwise_distances(l3_clamp_E_fptt, l3_norm_E_fptt, metric='cosine')

pair_dist_E_l2_b = pairwise_distances(l2_clamp_E_bptt, l2_norm_E_bptt, metric='cosine')
pair_dist_E_l2_f = pairwise_distances(l2_clamp_E_fptt, l2_norm_E_fptt, metric='cosine')

pair_dist_E_l1_b = pairwise_distances(l1_clamp_E_bptt, l1_norm_E_bptt, metric='cosine')
pair_dist_E_l1_f = pairwise_distances(l1_clamp_E_fptt, l1_norm_E_fptt, metric='cosine')

sns.set(font_scale=1.5)

fig, axes = plt.subplots(3, 2, figsize=(9, 11), sharex=True, sharey=True)
sns.despine()
sns.heatmap(1-pair_dist_E_l1_b, ax=axes[0, 0], cbar=True)
axes[0, 0].set_ylabel('clamped reps')
axes[0, 0].set_title('normal reps')
axes[0, 0].tick_params(left=False, bottom=False)

sns.heatmap(1-pair_dist_E_l1_f, ax=axes[0, 1], cbar=True)
axes[0, 1].set_title('normal reps')
axes[0, 1].tick_params(left=False, bottom=False)

sns.heatmap(1-pair_dist_E_l2_b, ax=axes[1, 0], cbar=True)
axes[1, 0].set_ylabel('clamped reps')
axes[1, 0].tick_params(left=False, bottom=False)

sns.heatmap(1-pair_dist_E_l2_f, ax=axes[1, 1], cbar=True)
axes[1, 1].tick_params(left=False, bottom=False)

sns.heatmap(1-pair_dist_E_l3_b, ax=axes[2, 0], cbar=True)
axes[2, 0].set_ylabel('clamped reps')
axes[2, 0].tick_params(left=False, bottom=False)

sns.heatmap(1-pair_dist_E_l3_f, ax=axes[2, 1], cbar=True)
axes[2, 1].tick_params(left=False, bottom=False)

cols = ['BPTT', 'FPTT']
rows = ['L1', 'L2', 'L3']

pad = 30

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
# compute the statistics of within and between class distances for each layer in both models 
df_dist = pd.DataFrame(columns=['model', 'layer', 'within', 'between', 'class'])

distances = [[pair_dist_E_l1_b, pair_dist_E_l1_b], [pair_dist_E_l2_b, pair_dist_E_l2_f], [pair_dist_E_l3_b, pair_dist_E_l3_f]]

for i in range(10):
    for j in range(3):
        df_1 = pd.DataFrame({'Model': 'Energy', 'L': str(j + 1), 
                                  'within': distances[j][0][i, i], 
                                  'between': np.delete(distances[j][0][i], i, 0).mean(), 
                                  'class': i}, index=[0])
        df_2 = pd.DataFrame({'Model': 'Control', 'L': str(j + 1),
                                    'within': distances[j][1][i, i], 
                                    'between': np.delete(distances[j][1][i], i, 0).mean(),
                                    'class': i}, index=[0])
        
        df_dist = pd.concat([df_dist, df_1, df_2])

df_dist = pd.melt(df_dist, id_vars=['Model', 'L', 'class'], value_vars=['within', 'between'],
                    var_name='distance type', value_name='value')
df_dist.head()

# %%
sns.set(font_scale=1.5)
sns.set_style("whitegrid", {'axes.grid' : False})
colors = [(0.1271049596309112, 0.4401845444059977, 0.7074971164936563), 
                     (0.9949711649365629, 0.5974778931180315, 0.15949250288350636)]

sns.catplot(data=df_dist, x='L', y='value', hue='distance type', col='Model', kind='bar', palette=colors)
plt.show()

# %%
# compute confidence interval for each layer and model type and distance type
for model in ['w E', 'w/o E']:
    for layer in [1, 2, 3]:
        for dist_type in ['within', 'between']:
            df_temp = df_dist[(df_dist['model'] == model) & (df_dist['layer'] == layer) & (df_dist['distance type'] == dist_type)]
            stats = df_temp['value'].agg(['mean', 'sem'])

            print(model, layer, dist_type)
            stats['ci95_hi'] = stats['mean'] + 1.96* stats['sem']
            stats['ci95_lo'] = stats['mean'] - 1.96* stats['sem']
            print('cf: %.4f, %.4f' % (stats['ci95_lo'], stats['ci95_hi']))

# %%
##############################################################
# decode from clamped representations 
##############################################################
no_input = torch.zeros((1, IN_dim)).to(device)

MSE_loss = nn.MSELoss()

test_loader2 = torch.utils.data.DataLoader(testdata, batch_size=batch_size,
                                          shuffle=False, num_workers=2)

# build a MLP with 2 hidden layers                                                                              
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(in_dim, out_dim)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc3 = nn.Linear(hidden_dim, out_dim)

        # # xavier initialisation
        # nn.init.xavier_uniform_(self.fc1.weight)
        # nn.init.xavier_uniform_(self.fc2.weight)
        # nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        # x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        x = self.fc1(x)
        return x                                          

# %%
def plot_projection(rep, label, weights, bias):
    img = (weights @ rep + bias).reshape(28, 28)
    plt.imshow(img)
    plt.title(str(label))
    plt.show()
    return img

def train_linear_proj(layer, model):
    mlp = MLP(hidden_dim[layer], 700, IN_dim).to(device)
    optimiser = optim.Adam(mlp.parameters(), lr=0.001, weight_decay=0.0001)

    loss_log = []

    for e in range(20): 
        for i, (data, target) in enumerate(test_loader2):
            data, target = data.to(device), target.to(device)
            data = data.view(-1, model.in_dim)

            with torch.no_grad():
                model.eval()

                hidden = model.init_hidden(data.size(0))

                _, h = model.inference(data, hidden, T)

            spks = get_states([h], 1+layer*4, hidden_dim[layer], batch_size, T, batch_size)

            train_data = torch.tensor(spks.mean(axis=1)).to(device) 
            # print(train_data.size())

            optimiser.zero_grad()

            out = mlp(train_data)
            loss = MSE_loss(out, data)
            loss_log.append(loss.data.cpu())

            loss.backward()
            optimiser.step()

        print('%i train loss: %.4f' % (e, loss))

        if e %2 == 0:
            plt.imshow(out[target == 0][0].cpu().detach().reshape(28, 28))
            plt.title('sample1 %i' % target[target == 0][0].item())
            plt.show()

            # find the next image with class 0
            plt.imshow(out[target == 0][1].cpu().detach().reshape(28, 28))
            plt.title('sample2 %i' % target[target == 0][1].item())
            plt.show()

    torch.cuda.empty_cache()

    mlp.eval()

    return mlp, [i.cpu() for i in loss_log]

# %%
layer = 1
l2_E_decoder_f, loss_E_f = train_linear_proj(layer, model_wE_fptt)
l2_nE_decoder_f, loss_nE_f = train_linear_proj(layer, model_woE_fptt)

l2_E_decoder_b, loss_E_b = train_linear_proj(layer, model_wE_bptt)
l2_nE_decoder_b, loss_nE_b = train_linear_proj(layer, model_woE_bptt)

decoders_f = [l2_E_decoder_f, l2_nE_decoder_f]
decoders_b = [l2_E_decoder_b, l2_nE_decoder_b]

# %%
# plot loss curve of training
colors = [(0.1271049596309112, 0.4401845444059977, 0.7074971164936563), 
                     (0.9949711649365629, 0.5974778931180315, 0.15949250288350636)]

sns.set(font_scale=1.5)
sns.set_style("whitegrid", {'axes.grid' : False})

# turn loss into float
loss_E_b = [i.item() for i in loss_E_b]
loss_nE_b = [i.item() for i in loss_nE_b]

fig, ax = plt.subplots(figsize=(5, 4))
# ax.plot(loss_E_f, label='FPTT Energy L%i' % (layer+1))
# ax.plot(loss_nE_f, label='FPTT Control L%i' % (layer+1))    

sns.lineplot(x=np.arange(len(loss_E_b)), y=loss_E_b, label='BPTT Energy L%i' % (layer+1), color=colors[0], ax=ax)
sns.lineplot(x=np.arange(len(loss_E_b)), y=loss_nE_b, label='BPTT Control L%i' % (layer+1), color=colors[1], ax=ax)

ax.legend()
# frame off 
ax.spines[['right', 'top']].set_visible(False)
ax.set_ylabel('MES loss')
ax.set_xlabel('steps')
plt.legend(frameon=False)
# increase font size
plt.show()


# %%
# plot fptt projection
fptt_decoded_E = []
fptt_decoded_nE = []

fig, axes = plt.subplots(2, 10, figsize=(10, 2))

with torch.no_grad():
    for proj_class in range(n_classes):
        img1 = decoders_f[0](torch.tensor(l3_clamp_E_fptt[proj_class].astype('float32')).to(device).view(-1, hidden_dim[layer])).reshape(28, 28).cpu()
        fptt_decoded_E.append(img1.flatten())
        axes[0][proj_class].imshow(img1)
        axes[0, proj_class].set_title(str(proj_class))
        # axes[0][proj_class].axis('off')
        axes[0, proj_class].tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)

        img2 = decoders_f[1](torch.tensor(l3_clamp_nE_fptt[proj_class].astype('float32')).to(device).view(-1, hidden_dim[layer])).reshape(28, 28).cpu()
        fptt_decoded_nE.append(img2.flatten())
        axes[1][proj_class].imshow(img2)
        axes[1, proj_class].tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
        # axes[1][proj_class].axis('off')

fig.suptitle('FPTT projection from clampled rep back to image plane layer %i' % (layer+1))
axes[0, 0].set_ylabel('w E', rotation=0, labelpad=20)
axes[1, 0].set_ylabel('w/o E', rotation=0, labelpad=20)

plt.tight_layout()
plt.show()

fptt_decoded_E = np.vstack(fptt_decoded_E)
fptt_decoded_nE = np.vstack(fptt_decoded_nE)
print(fptt_decoded_E.shape, fptt_decoded_nE.shape)

# %%
# plot bptt projection
bptt_decoded_E = []
bptt_decoded_nE = []

fig, axes = plt.subplots(2, 10, figsize=(10, 3))

with torch.no_grad():
    for proj_class in range(n_classes):
        img1 = decoders_b[0](torch.tensor(l3_clamp_E_bptt[proj_class].astype('float32')).to(device).view(-1, hidden_dim[layer])).reshape(28, 28).cpu()
        bptt_decoded_E.append(img1.flatten())
        axes[0][proj_class].imshow(img1, cmap='viridis')
        axes[0, proj_class].set_title(str(proj_class))
        # axes[0][proj_class].axis('off')
        axes[0, proj_class].tick_params(left = False, right = False , labelleft = False ,
                                        labelbottom = False, bottom = False)

        img2 = decoders_b[1](torch.tensor(l3_clamp_nE_bptt[proj_class].astype('float32')).to(device).view(-1, hidden_dim[layer])).reshape(28, 28).cpu()
        bptt_decoded_nE.append(img2.flatten())
        axes[1][proj_class].imshow(img2, cmap='viridis')
        axes[1, proj_class].tick_params(left = False, right = False , labelleft = False ,
                                        labelbottom = False, bottom = False)
        
fig.suptitle('BPTT projection from clampled rep back to image plane layer %i' % (layer+1))
axes[0, 0].set_ylabel('Energy', rotation=0, labelpad=40)
axes[1, 0].set_ylabel('Control', rotation=0, labelpad=40)

plt.tight_layout()
plt.show()

bptt_decoded_nE = np.vstack(bptt_decoded_nE)
bptt_decoded_E = np.vstack(bptt_decoded_E)
print(bptt_decoded_E.shape, bptt_decoded_nE.shape)

# %%
# compute the mean pixel value of each class
mean_pixel = torch.zeros((10, IN_dim))

for idx, (data, target) in enumerate(test_loader_all):
    for i in range(10):
        mean_pixel[i] += data[target == i].view(-1, IN_dim).sum(axis=0)

for i in range(10):
    mean_pixel[i] /= len(testdata.targets[testdata.targets == i])

mean_pixel = mean_pixel.cpu().numpy()

# %%
bptt_diff = []
fptt_diff = []

for i in range(10):
    bptt_diff.append(distance.cosine(mean_pixel[i], bptt_decoded_E[i]))
    fptt_diff.append(distance.cosine(mean_pixel[i], fptt_decoded_E[i]))

# create df 
df = pd.DataFrame({
    'Mean pixel diff': np.concatenate((bptt_diff, fptt_diff)),
    'Class': np.concatenate((np.arange(10), np.arange(10))),
    'Model': ['BPTT'] * 10 + ['FPTT'] * 10,
    'Layer': np.concatenate((np.full(10, layer+1), np.full(10, layer+1)))
})

# save df as csv
df.to_csv('/home/lucy/spikingPC/results/mean_pixel_diff_layer_%i.csv' % (layer+1))

sns.set_style("whitegrid", {'axes.grid' : False})

fig = plt.figure(figsize=(7, 5))
sns.barplot(data=df, x='Class', y='Mean pixel diff', hue='Model', palette=colors)
sns.despine()
plt.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('L%i' %(layer+1))
plt.show()
# %%
# load csv files containing mean pixel diff for each layer
df1 = pd.read_csv('/home/lucy/spikingPC/results/mean_pixel_diff_layer_1.csv')
df2 = pd.read_csv('/home/lucy/spikingPC/results/mean_pixel_diff_layer_2.csv')
df3 = pd.read_csv('/home/lucy/spikingPC/results/mean_pixel_diff_layer_3.csv')

df = pd.concat([df1, df2, df3])

# change layer number to string
df['Layer'] = df['Layer'].astype(str)

# change mean pixel diff column name to cosine distance 
df = df.rename(columns={'Mean pixel diff': 'Cosine distance'})

df.head()

# %%
sns.barplot(data=df, x='Layer', y='Cosine distance', hue='Model', palette=colors)
plt.legend(frameon=False)
sns.despine()
plt.title('Cosine distance')
plt.show()
# %%
