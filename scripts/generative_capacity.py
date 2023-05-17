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
    model_wE.eval()
    model_woE.eval()

    hidden_i = model_wE.init_hidden(1)

    log_sm_E_gen, hidden_gen_E = model_wE.clamped_generate(dig, no_input, hidden_i, T * 2, clamp_value=1)
    log_sm_nE_gen, hidden_gen_nE = model_woE.clamped_generate(dig, no_input, hidden_i, T * 2, clamp_value=1)
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
fig, axes = plt.subplots(1, 2)
pos = axes[0].imshow((param_dict_wE['layer2to1.weight'] @ spk_gen_l2_E).reshape(20, -1))
fig.colorbar(pos, ax=axes[0], shrink=0.5)
axes[0].set_title('w E mean l2 > l1 clamped generation')

pos = axes[1].imshow((param_dict_wE['layer2to1.weight'] @ spk_gen_l2_nE).reshape(20, -1))
fig.colorbar(pos, ax=axes[1], shrink=0.5)
axes[1].set_title('w/o E mean l2 > l1 clamped generation')

plt.tight_layout()
plt.show()

# %%
##############################################################
# test generative capacity of network with clamping
##############################################################
# compute mean l2, l3 reps from E and nE model with clamped mode 
l1_norm_E = np.zeros((10, hidden_dim[0]))
l1_norm_nE = np.zeros((10, hidden_dim[0]))

l2_norm_E = np.zeros((10, hidden_dim[1]))
l3_norm_E = np.zeros((10, hidden_dim[2]))

l2_norm_nE = np.zeros((10, hidden_dim[1]))
l3_norm_nE = np.zeros((10, hidden_dim[2]))

# test loader with all images
test_loader_all = torch.utils.data.DataLoader(testdata, batch_size=batch_size, shuffle=False, num_workers=2)

# get means from normal condition
for i, (data, target) in enumerate(test_loader_all):
    data, target = data.to(device), target.to(device)
    data = data.view(-1, model_wE.in_dim)

    with torch.no_grad():
        model_wE.eval()
        model_woE.eval()

        hidden = model_wE.init_hidden(data.size(0))

        _, h_E = model_wE.inference(data, hidden, T)
        _, h_nE = model_woE.inference(data, hidden, T)

        l1_E = get_states([h_E], 1, hidden_dim[0], batch_size, T=T, num_samples=batch_size)
        l1_nE = get_states([h_nE], 1, hidden_dim[0], batch_size, T=T, num_samples=batch_size)

        l2_E = get_states([h_E], 5, hidden_dim[1], batch_size, T=T, num_samples=batch_size)
        l2_nE = get_states([h_nE], 5, hidden_dim[1], batch_size, T=T, num_samples=batch_size)

        l3_E = get_states([h_E], 9, hidden_dim[2], batch_size, T=T, num_samples=batch_size)
        l3_nE = get_states([h_nE], 9, hidden_dim[2], batch_size, T=T, num_samples=batch_size)

        for i in range(n_classes):
            l1_norm_E[i] += l1_E[target.cpu() == i].mean(axis=1).sum(axis=0)  # avg over t and sum over all samples 
            l1_norm_nE[i] += l1_nE[target.cpu() == i].mean(axis=1).sum(axis=0)

            l2_norm_E[i] += l2_E[target.cpu() == i].mean(axis=1).sum(axis=0)  # avg over t and sum over all samples 
            l2_norm_nE[i] += l2_nE[target.cpu() == i].mean(axis=1).sum(axis=0)

            l3_norm_E[i] += l3_E[target.cpu() == i].mean(axis=1).sum(axis=0)  # avg over t and sum over all samples 
            l3_norm_nE[i] += l3_nE[target.cpu() == i].mean(axis=1).sum(axis=0)

    torch.cuda.empty_cache()

# avg all samples 
l1_norm_E = l1_norm_E / len(testdata.data)
l1_norm_nE = l1_norm_nE / len(testdata.data)

l2_norm_E = l2_norm_E / len(testdata.data)
l2_norm_nE = l2_norm_nE / len(testdata.data)

l3_norm_E = l3_norm_E / len(testdata.data)
l3_norm_nE = l3_norm_nE / len(testdata.data)

# %%
# clamped condition
no_input = torch.zeros((1, IN_dim)).to(device)
clamp_T = T * 5


l1_clamp_E = np.zeros((10, hidden_dim[0]))
l1_clamp_nE = np.zeros((10, hidden_dim[0]))

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

        _, hidden_gen_E_ = model_wE.clamped_generate(i, no_input, hidden_i, clamp_T, clamp_value=1)
        _, hidden_gen_nE_ = model_woE.clamped_generate(i, no_input, hidden_i, clamp_T, clamp_value=-1)

        # 
        l1_E = get_states([hidden_gen_E_], 1, hidden_dim[0], 1, clamp_T, num_samples=1)
        l1_nE = get_states([hidden_gen_nE_], 1, hidden_dim[0], 1, clamp_T, num_samples=1)

        # get gen 
        l2_E = get_states([hidden_gen_E_], 5, hidden_dim[1], 1, clamp_T, num_samples=1)
        l2_nE = get_states([hidden_gen_nE_], 5, hidden_dim[1], 1, clamp_T, num_samples=1)

        l3_E = get_states([hidden_gen_E_], 9, hidden_dim[2], 1, clamp_T, num_samples=1)
        l3_nE = get_states([hidden_gen_nE_], 9, hidden_dim[2], 1, clamp_T, num_samples=1)

        l1_clamp_E[i] += np.squeeze(l1_E.mean(axis=1))
        l1_clamp_nE[i] += np.squeeze(l1_nE.mean(axis=1))

        l2_clamp_E[i] += np.squeeze(l2_E.mean(axis=1))
        l2_clamp_nE[i] += np.squeeze(l2_nE.mean(axis=1))

        l3_clamp_E[i] += np.squeeze(l3_E.mean(axis=1))
        l3_clamp_nE[i] += np.squeeze(l3_nE.mean(axis=1))

    torch.cuda.empty_cache()

# %%
fig, axes = plt.subplots(2, 10, figsize=(25, 6))
for i in range(10):
    pos = axes[0][i].imshow(((param_dict_wE['layer2to1.weight']@l2_clamp_E[i]) @ param_dict_wE['input_fc.weight'] ).reshape(28, 28))
    fig.colorbar(pos, ax=axes[0][i], shrink=0.5)
    axes[0][i].set_title('w E l2 > l1 class%i' % i)

    pos = axes[1][i].imshow(((param_dict_woE['layer2to1.weight']@l2_clamp_nE[i]) @ param_dict_woE['input_fc.weight'] ).reshape(28, 28))
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

pair_dist_E_l1 = pairwise_distances(l1_clamp_E, l1_norm_E, metric='cosine')
pair_dist_nE_l1 = pairwise_distances(l1_clamp_nE, l1_norm_nE, metric='cosine')

max = np.max(np.concatenate((pair_dist_E_l2, pair_dist_nE_l2, pair_dist_E_l3, 
                             pair_dist_nE_l3, pair_dist_E_l1, pair_dist_nE_l1)))

fig, axes = plt.subplots(3, 2, figsize=(9, 11), sharex=True, sharey=True)
sns.despine()
sns.heatmap(1-pair_dist_E_l1, ax=axes[0, 0], cbar=True)
axes[0, 0].set_ylabel('clamped reps')
axes[0, 0].set_title('normal reps')
axes[0, 0].tick_params(left=False, bottom=False)

sns.heatmap(1-pair_dist_nE_l1, ax=axes[0, 1], cbar=True)
axes[0, 1].set_title('normal reps')
axes[0, 1].tick_params(left=False, bottom=False)

sns.heatmap(1-pair_dist_E_l2, ax=axes[1, 0], cbar=True)
axes[1, 0].set_ylabel('clamped reps')
axes[1, 0].tick_params(left=False, bottom=False)

sns.heatmap(1-pair_dist_nE_l2, ax=axes[1, 1], cbar=True)
axes[1, 1].tick_params(left=False, bottom=False)

sns.heatmap(1-pair_dist_E_l3, ax=axes[2, 0], cbar=True)
axes[2, 0].set_ylabel('clamped reps')
axes[2, 0].tick_params(left=False, bottom=False)

sns.heatmap(1-pair_dist_nE_l3, ax=axes[2, 1], cbar=True)
axes[2, 1].tick_params(left=False, bottom=False)

cols = ['Energy', 'Control']
rows = ['L1', 'L2', 'L3']

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
# compute the statistics of within and between class distances for each layer in both models 
df_dist = pd.DataFrame(columns=['model', 'layer', 'within', 'between', 'class'])

distances = [[pair_dist_E_l1, pair_dist_nE_l1], [pair_dist_E_l2, pair_dist_nE_l2], [pair_dist_E_l3, pair_dist_nE_l3]]

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
                    var_name='Similarity type', value_name='Similarity')
df_dist.head()

# %%
sns.set(font_scale=1.5)
sns.set_style("whitegrid", {'axes.grid' : False})
colors = [(0.1271049596309112, 0.4401845444059977, 0.7074971164936563), 
                     (0.9949711649365629, 0.5974778931180315, 0.15949250288350636)]

sns.catplot(data=df_dist, x='L', y='Similarity', hue='Similarity type', col='Model', kind='bar', palette=colors)
plt.show()

# %%
# compute confidence interval for each layer and model type and distance type
for model in ['Energy', 'Control']:
    for layer in [1, 2, 3]:
        for dist_type in ['within', 'between']:
            df_temp = df_dist[(df_dist['Model'] == model) & (df_dist['L'] == str(layer)) & (df_dist['distance type'] == dist_type)]
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
l2_E_decoder, loss_E = train_linear_proj(layer, model_wE)
l2_nE_decoder, loss_nE = train_linear_proj(layer, model_woE)

decoders = [l2_E_decoder, l2_nE_decoder]

# %%
# plot loss curve of training
colors = [(0.1271049596309112, 0.4401845444059977, 0.7074971164936563), 
                     (0.9949711649365629, 0.5974778931180315, 0.15949250288350636)]
sns.set_style("whitegrid", {'axes.grid' : False})

fig, ax = plt.subplots(figsize=(5, 4))
plt.rcParams.update({'font.size': 14})

ax.plot(loss_E, label='Energy L%i' % (layer+1), color=colors[0])
ax.plot(loss_nE, label='Control L%i' % (layer+1), color=colors[1])    
ax.legend()
# frame off 
ax.spines[['right', 'top']].set_visible(False)
ax.set_ylabel('MES loss')
ax.set_xlabel('steps')
plt.legend(frameon=False)
# increase font size
plt.show()


# %%
fig, axes = plt.subplots(2, 10, figsize=(10, 3))

with torch.no_grad():
    for proj_class in range(n_classes):
        img1 = decoders[0](torch.tensor(l2_clamp_E[proj_class].astype('float32')).to(device).view(-1, hidden_dim[layer])).reshape(28, 28).cpu()
        axes[0][proj_class].imshow(img1, cmap='viridis')
        axes[0, proj_class].set_title(str(proj_class))
        # axes[0][proj_class].axis('off')
        axes[0, proj_class].tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)

        img2 = decoders[1](torch.tensor(l2_clamp_nE[proj_class].astype('float32')).to(device).view(-1, hidden_dim[layer])).reshape(28, 28).cpu()
        axes[1][proj_class].imshow(img2, cmap='viridis')
        axes[1, proj_class].tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
        # axes[1][proj_class].axis('off')

fig.suptitle('projection from clampled rep back to image plane layer %i' % (layer+1))
axes[0, 0].set_ylabel('Energy', rotation=0, labelpad=40)
axes[1, 0].set_ylabel('Control', rotation=0, labelpad=40)

plt.tight_layout()
plt.show()

# %%
# clamp with noise 
noise = torch.zeros(n_classes).to(device=device)
clamp_class = 0
steps=10
clamp_T = T * 5

def clamp_with_noise(layer, model, clamp_class=0):

    noise_clamp = np.zeros((10, hidden_dim[layer])) 

    for i in range(steps):
        noise = torch.rand(10).to(device) 
        with torch.no_grad():
            model.eval()

            hidden_i = model.init_hidden(1)

            _, hidden_gen_E_ = model.clamped_generate(clamp_class, no_input, hidden_i, clamp_T, clamp_value=1., noise=noise)

            # 
            l1_E = get_states([hidden_gen_E_], 1+layer*4, hidden_dim[layer], 1, clamp_T, num_samples=1)


            noise_clamp[i] += np.squeeze(l1_E.mean(axis=1))

    torch.cuda.empty_cache()

    return noise_clamp

noise_l3_clamp_E = clamp_with_noise(layer, model_wE, clamp_class)

# %%
fig, axes = plt.subplots(1, steps, figsize=(10, 1.5))
with torch.no_grad():
    for s in range(steps):
        img1 = decoders[0](torch.tensor(noise_l3_clamp_E[s].astype('float32')).to(device).view(-1, hidden_dim[layer])).reshape(28, 28).cpu()
        axes[s].imshow(img1)
        # axes[0][proj_class].axis('off')
        axes[s].tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)

fig.suptitle('projection from clampled rep with random noise back to image plane layer %i' % layer)
axes[0].set_ylabel('Energy', rotation=0, labelpad=20)

plt.tight_layout()
plt.show()

# %%
# get a batch of images from test data loader 
images_all, target_all = next(iter(test_loader2))
# generate from half occluded image 

# from occluded image clamp generate 
dig = 3
idx = 9 #np.random.choice(10, size=1)
print(idx)
three = images_all[target_all == dig][idx].squeeze()
three[:14, :] = -1

with torch.no_grad():
    model_wE.eval()

    hidden_i = model_wE.init_hidden(1)

    _, hidden_gen_E_ = model_wE.clamped_generate(dig, three.view(-1, IN_dim).to(device), 
                                                    hidden_i, clamp_T, clamp_value=1)
    
    _, hidden_gen_nE_ = model_woE.clamped_generate(dig, three.view(-1, IN_dim).to(device), 
                                                    hidden_i, clamp_T, clamp_value=1)

    spk_rep_e = get_states([hidden_gen_E_], 1+layer*4, hidden_dim[layer], 1, clamp_T, num_samples=1)
    spk_rep_ne = get_states([hidden_gen_nE_], 1+layer*4, hidden_dim[layer], 1, clamp_T, num_samples=1)

fig, axes = plt.subplots(1, 3, figsize=(6, 2))

sns.set(font_scale=1.)

axes[0].imshow(three, cmap='viridis')
axes[0].set_title('Occluded input')
axes[0].axis('off')

axes[1].imshow(decoders[0](torch.tensor(spk_rep_e.mean(axis=1)).to(device).view(-1, hidden_dim[layer])).reshape(28, 28).cpu().detach(), 
               cmap='viridis')
axes[1].set_title('Energy')
axes[1].axis('off')

axes[2].imshow(decoders[1](torch.tensor(spk_rep_ne.mean(axis=1)).to(device).view(-1, hidden_dim[layer])).reshape(28, 28).cpu().detach(), 
               cmap='viridis')
axes[2].set_title('Control')
axes[2].axis('off')

plt.tight_layout()
plt.show()

# %%
# reconstruction with noise 
sampled_image_reps = np.zeros((20, hidden_dim[layer]))

for i in range(20):
    # sample uniform noise with range [-1, 1] with size 10
    noise = torch.rand(10).to(device) * 2 - 1
    with torch.no_grad():
        model_wE.eval()

        hidden_i = model_wE.init_hidden(1)

        _, hidden_gen_E_ = model_wE.clamped_generate(0, three.view(-1, IN_dim).to(device), 
                                                     hidden_i, clamp_T, clamp_value=0, noise=noise)

        spk_rep = get_states([hidden_gen_E_], 1+layer*4, hidden_dim[layer], 1, T*3, num_samples=1)
        sampled_image_reps[i] += np.squeeze(spk_rep.mean(axis=1))

print(sampled_image_reps.shape)

# %%
# create a grid of images from the sampled image representations
fig, axes = plt.subplots(4, 5, figsize=(10, 5))
for i in range(4):
    for j in range(5):
        img = decoders[0](torch.tensor(sampled_image_reps[i*5+j].astype('float32')).to(device).view(-1, hidden_dim[layer])).reshape(28, 28).cpu().detach()
        axes[i][j].imshow(img)
        axes[i, j].tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)

plt.tight_layout()
plt.show()

# %%
# generate from noise at the input of layer 
# randomly sample one image from class 3
sampled_image_reps = np.zeros((50, hidden_dim[layer]))
noise_p = 0.05  # percentage of neurons that is given nosie

for i in range(50):
    # sample uniform noise with range [-1, 1] with size 10
    noise_vector = torch.zeros(hidden_dim[layer]).to(device)
    noise_vector[torch.randint(high=hidden_dim[layer], size=(int(hidden_dim[layer]*noise_p),))] = \
          torch.rand((int(hidden_dim[layer] * noise_p),)).to(device) - 0.5
    
    with torch.no_grad():
        model_wE.eval()

        hidden_i = model_wE.init_hidden(1)

        _, hidden_gen_E_ = model_wE.clamp_withnoise(1, no_input, hidden_i, clamp_T, noise=noise_vector, 
                                                    index = (2+layer*4), clamp_value=0.5)

        spk_rep = get_states([hidden_gen_E_], 1+layer*4, hidden_dim[layer], 1, clamp_T, num_samples=1)
        sampled_image_reps[i] += np.squeeze(spk_rep.mean(axis=1))

print(sampled_image_reps.shape)

# %%
# create a grid of images from the sampled image representations
fig, axes = plt.subplots(5, 10, figsize=(10, 5))
for i in range(5):
    for j in range(10):
        img = decoders[0](torch.tensor(sampled_image_reps[i*10+j].astype('float32')).to(device).view(-1, hidden_dim[layer])).reshape(28, 28).cpu().detach()
        axes[i][j].imshow(img)
        axes[i, j].tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)

plt.tight_layout()
plt.show()


# %%
# # plt clamp result of pc1min
# clamp_T = T * 5

# with torch.no_grad():
#     model_wE.eval()

#     hidden_i = model_wE.init_hidden(1)

#     _, hidden_gen_E_ = model_wE.clamped_generate(0, no_input, hidden_i, clamp_T, clamp_value=0, 
#                                                  noise=torch.tensor(out_E[idx_min]).to(device))
#     print('clamp done')
#     # 
#     l1_E = get_states([hidden_gen_E_], 1, hidden_dim[0], 1, clamp_T, num_samples=1)
#     l2_E = get_states([hidden_gen_E_], 5, hidden_dim[1], 1, clamp_T, num_samples=1)
#     l3_E = get_states([hidden_gen_E_], 9, hidden_dim[2], 1, clamp_T, num_samples=1)


# # %%
# fig, axes = plt.subplots(1, 2)

# img1 = plot_projection(l3_E[0].mean(axis=1), 0, 
#             decoders[0].weight.data.cpu().numpy(), 
#             decoders[0].bias.data.cpu().numpy())
# axes[0].imshow(img1)
# # axes[0][proj_class].axis('off')
# axes[0].tick_params(left = False, right = False , labelleft = False ,
#         labelbottom = False, bottom = False)
# axes[0].set_title('decoded sample specific clamp')

# # plot original sample from subset
# axes[1].set_title('original sample')
# axes[1].imshow(subset.data[idx_min])

# fig.suptitle('decoded image from sample specific clamp pattern %i' % layer)

# plt.tight_layout()
# plt.show()