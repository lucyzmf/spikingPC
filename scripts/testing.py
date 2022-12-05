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

from network import *
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

batch_size = 10

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
IN_dim = 28 * 28
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

    # for data, target in test_loader:
    for i, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(-1, IN_dim)

        with torch.no_grad():
            model.eval()
            init_hidden = model.init_hidden(data.size(0))

            outputs, hiddens = model(data, init_hidden, T)

            output = outputs[-1]
            # output = torch.stack(outputs[-10:]).mean(dim=0)

            test_loss += F.nll_loss(output, target, reduction='sum').data.item()
            pred = output.data.max(1, keepdim=True)[1]

        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        torch.cuda.empty_cache()

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
           test_loss, correct, len(test_loader.dataset),
           test_acc))
    return hiddens, test_loss, 100. * correct / len(test_loader.dataset), data.detach().cpu(), target.cpu()
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
model = one_layer_SeqModel(IN_dim, 784, n_classes, is_rec=True, is_LTC=False)
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
saved_dict = model_result_dict_load('/home/lucy/spikingPC/results/Dec-05-2022/energy_loss_3_adp_memloss/onelayer_rec_best.pth.tar')
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
hiddens, test_loss, _, data, targets = test(model, test_loader)


# %%
# get spiking pattern along the sequence 
spikes_all = get_spikes(hiddens)
# %%
# here each entry to spike_all is 16*784 (batchsize*num neurons) at each time step 
# spikes_one_img = np.mean(spikes_all, axis=1).transpose()
plot_spike_heatmap(spikes_all[0, :, :])

# %%
################################
# compute batch mean energy consumption for each time step  
################################
rec_layer_weight = param_dict['network.snn_layer.layer1_x.weight']

mean_spike_seq, mean_internal_drive_seq, energy_seq = compute_energy_consumption(spikes_all, rec_layer_weight)

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
# 2d visualisation of internal drive 
################################ 
n = 4  # sample number from batch 
rec_drive = get_internal_drive(spikes_all[n, :, :], rec_layer_weight, type='rec')

def plot_drive(internal_drive, name, sample, step_size): 
    fig, axs = plt.subplots(1, int(len(internal_drive)/step_size)+1, sharey=True)
    axs[0].imshow(data[sample, :].numpy().reshape((28, 28)))
    step = np.arange(len(internal_drive), step=step_size) # plot every 4 time steps 
    for i in range(len(step)): 
        pos=axs[i+1].imshow(internal_drive[step[i], :].reshape((28, 28)))
        axs[i+1].set_title('t = %i' % step[i])
    fig.suptitle(name + ' drive')
    fig.colorbar(pos, ax=axs[-1], shrink=0.6)
    fig.tight_layout()
    fig.subplots_adjust(top=1.5)
    plt.show()

plot_drive(rec_drive, 'recurrent', n, 4)


# %%
################################
# elongated sequence testing 
################################ 
# run inferece on two images continuously for T steps each 
data_sample = [3, 4]
# take mean of two different number samples to poke error 
abnor_sample = (data[data_sample[1], :] + data[2, :])/2
# visualise 
fig, axs = plt.subplots(1, 2)
axs[0].imshow(data[data_sample[0], :].reshape((28, 28)))
axs[0].set_title('sample1')
axs[1].imshow(abnor_sample.reshape((28, 28)))
axs[1].set_title('sample2 (abnormal)')
plt.show()


with torch.no_grad():
    model.eval()
    init_hidden = model.init_hidden(1) # batch size 1

    # get outputs and hiddens 
    outputs1, hiddens1 = model(data[data_sample[0], :].unsqueeze(0).to(device), init_hidden, T)
    # pass hidden to next sequence continuously 
    outputs2, hiddens2 = model(abnor_sample.unsqueeze(0).to(device), hiddens1[-1][0], T)

    # ge predictions 
    outputs1 = torch.stack(outputs1).squeeze()
    outputs2 = torch.stack(outputs2).squeeze()

    pred1 = outputs1.data.max(1, keepdim=True)[1]
    pred2 = outputs2.data.max(1, keepdim=True)[1]

# %%
# plot spikes
spikes_all_elong = get_spikes(hiddens1+hiddens2)
plot_spike_heatmap(spikes_all_elong[0, :, :])

# %%
# compute energy
mean_spike_elong, mean_internal_drive_elong, energy_elong = compute_energy_consumption(spikes_all_elong, rec_layer_weight)

# plot
t = np.arange(T*2)
plt.plot(t, energy_elong, label='energy')
plt.plot(t, mean_spike_elong, label='mean spiking')
plt.plot(t, mean_internal_drive_elong, label='mean internal drive by t')
plt.legend()
plt.title('elongated sequence exp, pred1 %i, pred2 %i'% (pred1[-1], pred2[-1]))
plt.show()


# %%
# plot rec
rec_drive_elong = get_internal_drive(spikes_all_elong[0, :, :], rec_layer_weight, type='rec')

# %%
plot_drive(rec_drive_elong, 'recurrent', data_sample[0], 5)
# %%
fig, axs = plt.subplots(1, 2)
pos = axs[0].imshow(spikes_all_elong[0, :, :10].mean(axis=1).reshape((28, 28)))
axs[0].set_title('mean t=0-10 for target %i' % targets[data_sample[0]])
fig.colorbar(pos, ax=axs[0], shrink=0.5)

pos = axs[1].imshow(spikes_all_elong[0, :, 20:30].mean(axis=1).reshape((28, 28)))
axs[1].set_title('mean t=20-30 for target %i' % targets[data_sample[1]])
fig.colorbar(pos, ax=axs[1], shrink=0.5)
plt.show()
# %%
# plot rec drive at specific time steps t=20, 21
fig, axs = plt.subplots(1, 4, figsize=(10, 3))
pos = axs[0].imshow(spikes_all_elong[0, :, 19].reshape((28, 28)))
axs[0].set_title('t=19 for target %i' % targets[data_sample[0]])
fig.colorbar(pos, ax=axs[0], shrink=0.5)

pos = axs[1].imshow(spikes_all_elong[0, :, 20].reshape((28, 28)))
axs[1].set_title('t=20 for target abn')
fig.colorbar(pos, ax=axs[1], shrink=0.5)

pos = axs[2].imshow(spikes_all_elong[0, :, 21].reshape((28, 28)))
axs[2].set_title('t=21 for target abn')
fig.colorbar(pos, ax=axs[2], shrink=0.5)

pos = axs[3].imshow(spikes_all_elong[0, :, 22].reshape((28, 28)))
axs[3].set_title('t=22 for target abn')
fig.colorbar(pos, ax=axs[3], shrink=0.5)
plt.show()

# %%
