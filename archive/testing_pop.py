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

from network_populationcode import *
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
                                           shuffle=False, num_workers=2)
test_loader = torch.utils.data.DataLoader(testdata, batch_size=batch_size,
                                          shuffle=False, num_workers=2)

# %%
# set input and t param
###############################################################
# DEFINE NETWORK
###############################################################
# training parameters
T = 20
K = T  # K is num updates per sequence
omega = int(T / K)  # update frequency
clip = 1.
log_interval = 100
lr = 1e-3
epoch = 10
n_classes = 10
num_readout = 10
adap_neuron = True
onetoone = True

# set input and t param
IN_dim = 256
hidden_dim = 256 + 10 * num_readout
T = 20  # sequence length, reading from the same image T times

# define network
# define network
model = OneLayerSeqModelPop(IN_dim, hidden_dim, n_classes, is_rec=True, is_LTC=False,
                            is_adapt=adap_neuron, one_to_one=onetoone)
model.to(device)
print(model)

# define new loss and optimiser
total_params = count_parameters(model)
print('total param count %i' % total_params)

# %%
# untar saved dict
exp_dir = '/home/lucy/spikingPC/results/Jan-11-2023/fc_relu_rec_10readout_noener/'
saved_dict = model_result_dict_load(exp_dir + 'onelayer_rec_best.pth.tar')

model.load_state_dict(saved_dict['state_dict'])

# load feature extractor weights
feature_w = torch.load(exp_dir + 'feature_extractor_weights.pt')
relu = nn.ReLU()

# %%
# get all the test data in the right shape
target_all = testdata.targets.data
images = testdata.data.data

# %%
###############################################################################################
##########################          Get analysis data           ###############################
###############################################################################################
sum_over_seq = True
def get_analysis_data(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    preds_all_ = []
    hiddens_log = []

    # for data, target in test_loader:
    for i, (data, target) in enumerate(test_loader):

        # feature extract
        data = data.view(-1, 784) @ feature_w.T
        data = relu(data)

        data, target = data.to(device), target.to(device)
        data = data.view(-1, IN_dim)

        with torch.no_grad():
            model.eval()
            init_hidden = model.init_hidden(data.size(0))

            prob_outputs, log_softmax_outputs, hiddens = model(data, init_hidden, T)

            test_loss += F.nll_loss(log_softmax_outputs[-1], target, reduction='sum').data.item()
            if sum_over_seq:
                pred = prob_outputs.data.max(1, keepdim=True)[1]
            else:
                # if use line below, prob output here computed from sum of spikes at last time step
                pred = log_softmax_outputs[-1].data.max(1, keepdim=True)[1]

            # log network predictions
            preds_all_.append(pred.detach().cpu().numpy())
            hiddens_log.append(hiddens)

        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        torch.cuda.empty_cache()

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        test_acc))

    preds_all_ = np.stack(preds_all_)

    return hiddens_log, preds_all_


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
# plot_distribution(param_names, param_dict, 'weight')
# %%
# get all the hidden states
hiddens_all, preds_all = get_analysis_data(model, test_loader)

# %%
# get spikes from all test samples
spikes_all = []
for b in range(len(hiddens_all)):  # iter over each batch
    batch_spike = []
    for t in range(T):  # iter over time
        seq_spike = []
        for s in range(batch_size):  # per sample
            seq_spike.append(hiddens_all[b][t][1][s].detach().cpu().numpy())  # mean spiking for each sample
        seq_spike = np.stack(seq_spike)
        batch_spike.append(seq_spike)
    batch_spike = np.stack(batch_spike)
    spikes_all.append(batch_spike)

spikes_all = np.stack(spikes_all)
spikes_all = spikes_all.transpose(0, 2, 1, 3).reshape(10000, 20, hidden_dim)

# %%
spike_sums = np.zeros((10, 10))
for i in range(10):
    class_spikes = spikes_all[target_all == i, :, :].sum(axis=0).sum(axis=0)[:10*num_readout].reshape(10, num_readout)

    group_sum = class_spikes.sum(axis=1)

    spike_sums[i, :] = group_sum.T

sns.heatmap(spike_sums)
# plt.show()
plt.savefig(exp_dir+'spike sum per neuron group per class')
plt.close()



# %%
# class mean spiking
fig, axs = plt.subplots(2, 10, figsize=(20, 3), sharex=True)
for i in range(10):
    class_mean = spikes_all[target_all == i, :, :].mean(axis=0).mean(axis=0)

    # prediction neuron spiking 
    pos1 = axs[0][i].imshow(class_mean[:10*num_readout].reshape(10, num_readout))
    fig.colorbar(pos1, ax=axs[0][i], shrink=0.3)
    axs[0][i].axis('off')

    # error neuron spiking 
    pos2 = axs[1][i].imshow(class_mean[10*num_readout:].reshape(16, 16))
    fig.colorbar(pos2, ax=axs[1][i], shrink=0.3)
    axs[1][i].axis('off')


plt.title('spiking mean per class')
# plt.show()
plt.savefig(exp_dir+'spiking mean per class')
plt.close()


# %%
rec_layer_weight = param_dict['network.snn_layer.layer1_x.weight']

# class mean rec projection from 10 popluation neuron
rec_drive = spikes_all[:, :, :10*num_readout] @ rec_layer_weight[:, :10*num_readout].T
fig, axs = plt.subplots(1, 10, figsize=(20, 3), sharex=True)
for i in range(10):
    # correct 
    rec_drive_by_class = spikes_all[:, :, i*num_readout:(i+1)*num_readout] @ rec_layer_weight[:, i*num_readout:(i+1)*num_readout].T

    class_mean = rec_drive_by_class[target_all == i, :, :].mean(axis=0).mean(axis=0)
    pos1 = axs[i].imshow((class_mean[10*num_readout:]@feature_w.numpy()).reshape(28, 28))
    fig.colorbar(pos1, ax=axs[i], shrink=0.3)
    axs[i].axis('off')

plt.title('rec projection from class neuron populations')
# plt.show()
plt.savefig(exp_dir+'rec projection from class neuron populations')
plt.close()


# %%
# weights from pred neuron for class 0 to other pred neurons
abs_max = np.max(np.abs(rec_layer_weight[:10*num_readout, :10*num_readout]))
sns.heatmap(rec_layer_weight[:10*num_readout, :10*num_readout], vmax=abs_max, vmin=-abs_max, cmap='icefire')
# plt.show()
plt.savefig(exp_dir+'rec weights between readout neurons')
plt.close()


# %%
# plot a sequence 
sample = 0 
spike_seq = spikes_all[sample, :, :]
fig, axs = plt.subplots(3, T, figsize=(40, 6), sharex=True)
for i in range(T):
    # pred spiking 
    pos1 = axs[0][i].imshow(spike_seq[i, :10*num_readout].reshape(10, num_readout))
    # fig.colorbar(pos1, ax=axs[0][i], shrink=0.3)
    axs[0][i].axis('off')

    # error spiking 
    pos2 = axs[1][i].imshow(spike_seq[i, 10*num_readout:].reshape(16, 16))
    # fig.colorbar(pos2, ax=axs[1][i], shrink=0.3)
    axs[1][i].axis('off')

    # pred to error drive 
    pos3 = axs[2][i].imshow((rec_drive[sample, i, 10*num_readout:] @ feature_w.numpy()).reshape(28, 28))
    fig.colorbar(pos3, ax=axs[2][i], shrink=0.3)
    axs[1][i].axis('off')

# plt.show()
plt.savefig(exp_dir+'sample sequence spiking and pred to error drive')
plt.close()


# %%
# rec drive from one single neuron 
sample=3
neuron_number = 1
one_neuron = np.expand_dims(spikes_all[:, :, neuron_number], axis=2) @ np.expand_dims(rec_layer_weight[:, neuron_number], axis=1).T
sns.heatmap((one_neuron[sample, 19, 10*num_readout:] @ feature_w.numpy()).reshape(28, 28))
plt.show()
plt.close()














# %%
################################
# compute batch mean energy consumption for each time step  
################################
# rec_layer_weight = param_dict['network.snn_layer.layer1_x.weight']

# mean_spike_seq, mean_internal_drive_seq, energy_seq = compute_energy_consumption(spikes_all, rec_layer_weight)

# # plot
# t = np.arange(T)
# plt.plot(t, energy_seq)
# plt.title('energy by t')
# plt.show()
# # %%
# ################################
# # mean spiking along sequence 
# ################################ 
# plt.plot(t, mean_spike_seq)
# plt.title('mean spiking by t')
# plt.show()


# # %%
# ################################
# # elongated sequence testing 
# ################################ 
# # run inferece on two images continuously for T steps each 
# data_sample = [3, 4]
# # take mean of two different number samples to poke error 
# # abnor_sample = (data[data_sample[1], :] + data[2, :]) / 2
# abnor_sample = data[data_sample[1], :]
# # visualise 
# fig, axs = plt.subplots(1, 2)
# axs[0].imshow(data[data_sample[0], :].reshape((28+pad_size, 28)))
# axs[0].set_title('sample1')
# axs[1].imshow(abnor_sample.reshape((28+pad_size, 28)))
# axs[1].set_title('sample2 (abnormal)')
# plt.show()

# with torch.no_grad():
#     model.eval()
#     init_hidden = model.init_hidden(1)  # batch size 1

#     # get outputs and hiddens 
#     prob_outputs1, _, hiddens1 = model(data[data_sample[0], :].unsqueeze(0).to(device), init_hidden, 10)
#     # pass hidden to next sequence continuously 
#     prob_outputs2, _, hiddens2 = model(abnor_sample.unsqueeze(0).to(device), hiddens1[-1][0], 10)

#     # ge predictions 
#     # outputs1 = torch.stack(prob_outputs1).squeeze()
#     # outputs2 = torch.stack(prob_outputs2).squeeze()

#     pred1 = prob_outputs1.data.max(1, keepdim=True)[1]
#     pred2 = prob_outputs2.data.max(1, keepdim=True)[1]

# # %%
# # plot spikes
# spikes_all_elong = get_spikes(hiddens1 + hiddens2)
# plot_spike_heatmap(spikes_all_elong[0, :, :])

# # %%
# # compute energy
# mean_spike_elong, mean_internal_drive_elong, energy_elong = compute_energy_consumption(spikes_all_elong,
#                                                                                        rec_layer_weight)

# # plot
# t = np.arange(T)
# plt.plot(t, energy_elong, label='energy')
# plt.plot(t, mean_spike_elong, label='mean spiking')
# plt.plot(t, mean_internal_drive_elong, label='mean internal drive by t')
# plt.legend()
# plt.title('elongated sequence exp, pred1 %i, pred2 %i' % (pred1[-1], pred2[-1]))
# plt.show()

# # %%
# # plot rec
# rec_drive_elong = get_internal_drive(spikes_all_elong[0, :, :], rec_layer_weight)

# # %%
# plot_drive(rec_drive_elong, 'recurrent', data_sample[0], 5)
# # %%
# fig, axs = plt.subplots(1, 2)
# pos = axs[0].imshow(spikes_all_elong[0, :, :10].mean(axis=1).reshape((28+pad_size, 28)))
# axs[0].set_title('mean t=0-10 for target %i' % targets[data_sample[0]])
# fig.colorbar(pos, ax=axs[0], shrink=0.5)

# pos = axs[1].imshow(spikes_all_elong[0, :, 10:20].mean(axis=1).reshape((28+pad_size, 28)))
# axs[1].set_title('mean t=20-30 for target %i' % targets[data_sample[1]])
# fig.colorbar(pos, ax=axs[1], shrink=0.5)
# plt.show()
# # %%
# # plot spiking and rec drive at specific time steps 
# fig, axs = plt.subplots(3, 20, figsize=(40, 7))
# for i in range(20):
#     # spikes 
#     axs[0][i].imshow(spikes_all_elong[0, :, i].reshape((28+pad_size, 28)))
#     axs[0][i].axis('off')

#     # rec drive from prediction neurons 
#     pos = axs[1][i].imshow((spikes_all_elong[0, :10*10, i] @ rec_layer_weight[:, :10*10].T).reshape((28+pad_size, 28))[4:, :], 'bwr')
#     fig.colorbar(pos, ax=axs[1][i], shrink=0.5)
#     axs[1][i].axis('off')

#     # rec drive from other neurons 
#     pos = axs[2][i].imshow((spikes_all_elong[0, 10*10:, i] @ rec_layer_weight[:, 10*10:].T).reshape((28+pad_size, 28)), 'bwr')
#     fig.colorbar(pos, ax=axs[2][i], shrink=0.5)
#     axs[2][i].axis('off')

# # plt.show()
# plt.savefig(exp_dir+'sample seq')

# # %%
# # plot inputs to each predictive neuron at each timestep 
# t = np.arange(20)
# received_input_pred = spikes_all_elong[0, :, :].T @ rec_layer_weight [:, :10*10]

# for i in range(10): 
#     plt.plot(t, received_input_pred[:, i], label=str(i))
# plt.legend()
# plt.show()

# # %%
