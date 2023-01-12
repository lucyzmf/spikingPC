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

import matplotlib.pyplot as plt
import seaborn as sns
import IPython.display as ipd

from tqdm import tqdm

from network_seq import *
from utils import *
from FTTP import *

# %%
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

batch_size = 100

traindata = torchvision.datasets.MNIST(root='./data', train=True,
                                       download=True, transform=transform)

testdata = torchvision.datasets.MNIST(root='./data', train=False,
                                      download=True, transform=transform)

# data loading 
train_loader = torch.utils.data.DataLoader(traindata, batch_size=batch_size,
                                           shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(testdata, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

# %%
# set input and t param
pad_size = 2
IN_dim = (28 + pad_size) * (28 + pad_size * 2)
img_dim = (28 + pad_size, 28 + pad_size * 2)
T = 20  # sequence length, reading from the same image T times 


# %%
###############################################################################################
##########################          Test function             ###############################
###############################################################################################
# test function
def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    all_spikes = []
    all_inputs = []
    all_targets = []

    # for data, target in test_loader:
    for i, (data, target) in enumerate(test_loader):
        # pad input
        p2d = (pad_size, pad_size, pad_size, 0)  # pad last dim by (1, 1) and 2nd to last by (2, 2)
        data = F.pad(data, p2d, 'constant', -1)
        # log 
        all_targets.append(target)

        data, target = data.to(device), target.to(device)
        B = data.size()[0]

        with torch.no_grad():
            model.eval()
            hidden = model.init_hidden(data.size(0))

            probs_outputs = []  # for pred computation
            log_softmax_outputs = []  # for loss computation
            spikes_ = []
            inputs_ = []
            spike_sum = torch.zeros(B, 10).to(device)

            # iterate over sequence
            for t in range(T):
                # transform data
                data_shifted = shift_input(t, T, data)
                # log
                inputs_.append(data_shifted)

                data_shifted = data_shifted.view(-1, IN_dim)

                if t == 0:
                    hidden = model.init_hidden(data.size(0))
                    f_output, hidden, hiddens = model.network.forward(data_shifted, hidden)
                elif t % omega == 0:
                    f_output, hidden, hiddens = model.network.forward(data_shifted, hidden)

                # log all spikes to spikes_
                spikes_.append(hidden[1])

                # read out from 10 populations
                output_spikes = hidden[1][:, :10 * num_readout].view(-1, 10,
                                                                     num_readout)  # take the first 10*28 neurons for read out
                output_spikes_sum = output_spikes.sum(dim=2)  # mean firing of neurons for each class
                spike_sum += output_spikes_sum

                prob_out = F.softmax(output_spikes_sum, dim=1)
                output = F.log_softmax(output_spikes_sum, dim=1)

                probs_outputs.append(prob_out)
                log_softmax_outputs.append(output)

            prob_out_sum = F.softmax(spike_sum, dim=1)

            test_loss += F.nll_loss(log_softmax_outputs[-1], target, reduction='sum').data.item()
            # pred = prob_outputs[-1].data.max(1, keepdim=True)[1]

            # if use line below, prob output here computed from sum of spikes over entire seq 
            pred = prob_out_sum.data.max(1, keepdim=True)[1]

            # stack spikes before appending to all spikes, should contain 20*batch*layer size spikes
            spikes_ = torch.stack(spikes_)
            inputs_ = torch.stack(inputs_)
            all_spikes.append(spikes_)
            all_inputs.append(inputs_)


        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        torch.cuda.empty_cache()

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        test_acc))

    all_spikes = torch.stack(all_spikes)
    all_inputs = torch.stack(all_inputs)
    all_targets = torch.stack(all_targets)

    print('all spikes: ' + str(all_spikes.size()))
    print('all inputs: ' + str(all_inputs.size()))
    print('all targets: ' + str(all_targets.size()))

    return all_spikes, all_inputs, all_targets


# %%
###############################################################
# DEFINE NETWORK
###############################################################
# training parameters
K = T  # K is num updates per sequence
omega = int(T / K)  # update frequency
clip = 1.
log_interval = 100
lr = 1e-3
epoch = 10
n_classes = 10

# define network
model = SeqModel_pop(IN_dim, IN_dim, n_classes, is_rec=True, is_LTC=False, oneToOne=True)
model.to(device)
print(model)

# define new loss and optimiser 
total_params = count_parameters(model)
print('total param count %i' % total_params)

# define optimiser
optimizer = optim.Adamax(model.parameters(), lr=lr, weight_decay=0.0001)
# reduce the learning after 20 epochs by a factor of 10
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# %%
# untar saved dict 
exp_dir = '/home/lucy/spikingPC/results/Dec-16-2022/exp_11_adp_memloss_clf1ener1_10popencode/'
saved_dict = model_result_dict_load(exp_dir + 'onelayer_rec_best.pth.tar')
# %%
model.load_state_dict(saved_dict['state_dict'])
# %%
###############################################################################################
##########################          analysis             ###############################
###############################################################################################
# get params and put into dict
param_names = []
param_dict = {}
for name, param in model.named_parameters():
    if param.requires_grad:
        param_names.append(name)
        param_dict[name] = param.detach().cpu().numpy()

print(param_names)

# %%
# plot weight distribution 
plot_distribution(param_names, param_dict, 'weight')
# %%
# get all the hidden states for the last batch in test loader 
all_spikes, data, targets = test(model, test_loader)

all_spikes = all_spikes.cpu().numpy().transpose(0, 2, 1, 3).reshape(10000, 20, IN_dim)
data = data.cpu().numpy().transpose(0, 2, 1, 3, 4, 5).reshape(10000, 20, IN_dim)
targets = torch.flatten(targets).cpu().numpy()

print(all_spikes.shape)
print(data.shape)
print(targets.shape)
# %%
# here each entry to spike_all is 16*784 (batchsize*num neurons) at each time step 
# spikes_one_img = np.mean(spikes_all, axis=1).transpose()
plot_spike_heatmap(all_spikes[0, :, :].T)

# %%
################################
# compute batch mean energy consumption for each time step  
################################
rec_layer_weight = param_dict['network.snn_layer.layer1_x.weight']

mean_spike_seq, mean_internal_drive_seq, energy_seq = compute_energy_consumption(all_spikes[:10, :, :].transpose(0, 2, 1), rec_layer_weight)

# plot
t = np.arange(T)
plt.plot(t, energy_seq)
plt.title('energy by t')
plt.show()
# %%
################################
# mean spiking along sequence 
################################ 
plt.plot(t, mean_spike_seq)
plt.title('mean spiking by t')
plt.show()

# %%
################################
# mean internal drive along sequence 
################################ 
plt.plot(t, mean_internal_drive_seq)
plt.title('mean internal drive by t')
plt.show()

# %%
################################
# 2d visualisation of internal drive of single seqeunce 
################################ 
n = 4  # sample number from batch 
img = torch.unsqueeze(data[n, :].reshape(img_dim), dim=0)
rec_drive = all_spikes[n, :, :].T @ rec_layer_weight

fig, axs = plt.subplots(2, T, figsize=(30, 3))
for i in range(T):
    img_ = shift_input(i, T, img)
    pos1 = axs[0][i].imshow(torch.squeeze(img_))
    axs[0][i].axis('off')

    pos2 = axs[1][i].imshow(rec_drive[i, :].reshape(img_dim))
    fig.colorbar(pos2, ax=axs[1][i], shrink=0.3)
    axs[1][i].axis('off')

plt.tight_layout()
plt.show()


# %%
################################
# sequence recurrent drive and spiking visualisation  
################################ 
n = 4 # sample number 
fig, axs = plt.subplots(3, 20, figsize=(40, 4))
for i in range(T):
    axs[0, i].imshow(data[n, i, :].reshape(img_dim))
    axs[0, i].axis('off')

    axs[1, i].imshow(all_spikes[n, i, :].reshape(img_dim))
    axs[1, i].axis('off')

    # projections from prediction neurons 
    pos = axs[2, i].imshow((all_spikes[n, i, :10*num_readout]@rec_layer_weight[:, :10*num_readout].T).reshape(img_dim)[4:])
    fig.colorbar(pos, ax=axs[2][i], shrink=0.3)
    axs[2, i].axis('off')

plt.savefig(exp_dir + 'sample seq full')
# plt.show()


# %%
################################
# class disentanglement 
################################ 

# class mean spiking 
fig, axs = plt.subplots(1, 10, figsize=(20, 3), sharex=True)
for i in range(10):
    class_mean = all_spikes[targets == i, :, :].mean(axis=0).mean(axis=0)
    pos = axs[i].imshow(class_mean.reshape(img_dim)[0:, :])
    fig.colorbar(pos, ax=axs[i], shrink=0.3)
    axs[i].axis('off')
plt.title('spiking mean per class')
plt.show()
# %%
# class mean rec drive 
rec_layer_weight = param_dict['network.snn_layer.layer1_x.weight']

rec_drive = all_spikes @ rec_layer_weight
fig, axs = plt.subplots(1, 10, figsize=(20, 3), sharex=True)
for i in range(10):
    class_mean = rec_drive[targets == i, :, :].mean(axis=0).mean(axis=0)
    pos = axs[i].imshow(class_mean.reshape(img_dim))
    fig.colorbar(pos, ax=axs[i], shrink=0.3)
    axs[i].axis('off')
plt.title('rec drive mean per class')
plt.show()

# %%
# class mean rec projection from 10 popluation neuron 
rec_drive = all_spikes[:, :, :10 * num_readout] @ rec_layer_weight[:, :10 * num_readout].T
fig, axs = plt.subplots(1, 10, figsize=(20, 3), sharex=True)
for i in range(10):
    class_mean = rec_drive[targets == i, :, :].mean(axis=0).mean(axis=0)
    pos = axs[i].imshow(class_mean.reshape(img_dim)[4:, :])
    fig.colorbar(pos, ax=axs[i], shrink=0.3)
    axs[i].axis('off')
plt.title('rec projection from 10 neuron populations')
plt.show()

# %%
# weights from pred neuron for class 0 to other pred neurons 
sns.heatmap(rec_layer_weight[:10 * 10, :10 * 10], cmap="vlag")
plt.show()

# %%
################################
# prediction neuron behaviour 
################################ 
# check how selective each neuron is in each group 
sum_spikes_per_pred = all_spikes.sum(axis=0).sum(axis=0)[:10*num_readout]
selectivity_per_pred = [] # list of 10 entrys consisting of spike sum of each of 100 neurons 
# compute ratio between firing per class/total spikes as selectivity measure 
for i in range(n_classes):
    spikes_ = all_spikes[targets == i, :, :10*num_readout].sum(axis=0).sum(axis=0)
    selectivity_idx = spikes_ / sum_spikes_per_pred
    selectivity_per_pred.append(selectivity_idx)

selectivity_per_pred = np.stack(selectivity_per_pred)
print(selectivity_per_pred.shape)

# %%
ax = sns.scatterplot(selectivity_per_pred[:, :].T)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.vlines(x=np.arange(10, 100, step=10), ymin = 0, ymax = 0.3, colors='grey')
plt.show()
# %%
