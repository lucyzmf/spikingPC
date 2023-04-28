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
# cluster top current setups and see if it is in a generative space 
# get subset of data from one class for one neuron

chosen_class = 1
b_size=5
subset = torch.utils.data.Subset(testdata, (testdata.targets == chosen_class).nonzero().squeeze())
print(len(subset))
test_loader3 = torch.utils.data.DataLoader(subset, batch_size=b_size, shuffle=False, num_workers=2)

# %%
# get the spk pattern of all neurons in the layer
layer = 1
model = model_wE

n_spk = np.zeros((len(subset), T, hidden_dim[layer]))

# get means from normal condition
for i, (data, target) in enumerate(test_loader3):
    data, target = data.to(device), target.to(device)
    data = data.view(-1, model.in_dim)

    with torch.no_grad():
        model.eval()

        hidden = model.init_hidden(data.size(0))

        _, h_E = model.inference(data, hidden, T)

        spks = get_states([h_E], 1+layer*4, hidden_dim[layer], b_size, T=T, num_samples=b_size)
        n_spk[i*b_size:(i+1)*b_size, :, :] = spks
        
    torch.cuda.empty_cache()
    
# %%
# find the neuron with biggest varaince of spk pattern across samples 
var = np.var(n_spk.mean(axis=1), axis=0)
max_var_idx = np.argmax(var)

# get a ranked list of indices of neurons based on var of spk rate 
ranked_idx_list = np.argsort(var)[::-1]

max_var_idx = ranked_idx_list[0]

# plot distribution of varaince
plt.hist(var)
plt.show()

# %%
# PCA
from sklearn.preprocessing import StandardScaler

# x = StandardScaler().fit_transform(n_spk[:, :, max_var_idx])
x = n_spk[:, :, max_var_idx]

# plot historgram of spk rate of a single neuron 
plt.hist(x.mean(axis=1), bins=20)
plt.title('dist spk rate of neuron %d' % max_var_idx)
plt.show()

# %%
# plot the likelihood of spk rate of a single neuron per time step
plt.plot(x.mean(axis=0))
plt.title('spk rate of neuron %d at time t' % max_var_idx)
plt.show()

# %%   
# make a ranked list of index of samples in chosen class based on the spike rate of a single neuron 
ranked_idx = np.argsort(x.mean(axis=1))

# plot every 20th sample in the ranked list in a grid
fig, axes = plt.subplots(5, 10, figsize=(10, 5))
count = 0
for i in range(5):
    for j in range(10):
        axes[i][j].imshow(testdata.data[testdata.targets == chosen_class][ranked_idx[count*22]].squeeze())
        axes[i, j].tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
        axes[i][j].set_title('%.2f' % x.mean(axis=1)[ranked_idx[count*22]])
        count+=1

fig.suptitle('ranked samples of class %d based on spk rate' % chosen_class)
plt.tight_layout()
plt.show()

# %%
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['pc1', 'pc2'])

principalDf['index'] = (testdata.targets == chosen_class).nonzero().squeeze().numpy()

# save df to exp_dir 
principalDf.to_csv(os.path.join(exp_dir_wE, 'pca_df.csv'))
# %%
# visualisation 
sns.scatterplot(principalDf, x='pc1', y='pc2')
plt.show()

# %%
# find correlation betweem mean spk rate and pc1
from scipy.stats import pearsonr

corr, _ = pearsonr(principalDf['pc1'], x.mean(axis=1))

plt.scatter(principalDf['pc1'], x.mean(axis=1))
plt.xlabel('pc1')
plt.ylabel('mean spk rate')
plt.title('corr %.2f' % corr)
plt.show()

# %%
pc1_max_idx = principalDf.sort_values(by='pc1')['index'].values[1]
# get index of the above selected row of the data frame
idx_max = principalDf[principalDf['index'] == pc1_max_idx].index[0]

pc1_min_idx = principalDf.sort_values(by='pc1')['index'].values[-1]
idx_min = principalDf[principalDf['index'] == pc1_min_idx].index[0]

# %%
fig, axes = plt.subplots(1, 2)
axes[0].set_title('pc1 max')
axes[0].imshow(testdata.data[pc1_max_idx])

axes[1].set_title('pc1 min')
axes[1].imshow(testdata.data[pc1_min_idx])

plt.show()

# %%
# choose one point from each bin of pc1 and plot image of the corresponding index
# get the pc1 values of each bin of pc1
pc1_bins = np.linspace(principalDf['pc1'].min(), principalDf['pc1'].max(), 15)
# get the index of the point in each bin
pc1_bin_idx = np.zeros((len(pc1_bins)-1, 1))
for i in range(len(pc1_bins)-1):
    # check if there are values in that bin
    if len(principalDf[(principalDf['pc1'] >= pc1_bins[i]) & (principalDf['pc1'] < pc1_bins[i+1])]) > 0:
        pc1_bin_idx[i] = principalDf[(principalDf['pc1'] >= pc1_bins[i]) & (principalDf['pc1'] < pc1_bins[i+1])].sort_values(by='pc1')['index'].values[0]

# plot the images of the points in each bin
fig, axes = plt.subplots(1, len(pc1_bins)-1)
for i in range(len(pc1_bins)-1):
    axes[i].imshow(testdata.data[int(pc1_bin_idx[i])])
    axes[i].set_title(str(i))
    axes[i].axis('off')
plt.show()

# %%
# find interval of pc1 where the variance of pc2 is the largest
# find in each bin the variance of pc2
pc2_bin_var = np.zeros((len(pc1_bins)-1, 1))
for i in range(len(pc1_bins)-1):
    pc2_bin_var[i] = principalDf[(principalDf['pc1'] >= pc1_bins[i]) & (principalDf['pc1'] < pc1_bins[i+1])]['pc2'].var()

# find the bin with the largest variance
max_var_idx = np.argmax(pc2_bin_var)

# within the max var interval, divide the pc2 values into 5 bins
pc2_bins = np.linspace(principalDf[(principalDf['pc1'] >= pc1_bins[max_var_idx]) & (principalDf['pc1'] < pc1_bins[max_var_idx+1])]['pc2'].min(),
                          principalDf[(principalDf['pc1'] >= pc1_bins[max_var_idx]) & (principalDf['pc1'] < pc1_bins[max_var_idx+1])]['pc2'].max(), 10)

# get the index of the point in each bin
pc2_bin_idx = np.zeros((len(pc2_bins)-1, 1))
for i in range(len(pc2_bins)-1):
    pc2_bin_idx[i] = principalDf[(principalDf['pc1'] >= pc1_bins[max_var_idx]) & (principalDf['pc1'] < pc1_bins[max_var_idx+1]) & 
                                 (principalDf['pc2'] >= pc2_bins[i]) & (principalDf['pc2'] < pc2_bins[i+1])].sort_values(by='pc2')['index'].values[0]
    
# plot the images of the points in each bin
fig, axes = plt.subplots(1, len(pc2_bins)-1)
for i in range(len(pc2_bins)-1):
    axes[i].imshow(testdata.data[int(pc2_bin_idx[i])])
    axes[i].set_title(str(i))
    axes[i].axis('off')

plt.show()

# %%
