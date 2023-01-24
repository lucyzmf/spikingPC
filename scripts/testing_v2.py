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

from network_class import *
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

# %%
IN_dim = 784
hidden_dim = [10 * num_readout, 784]
T = 20  # sequence length, reading from the same image T times

# define network
model = SnnNetwork(IN_dim, hidden_dim, n_classes, is_adapt=adap_neuron, one_to_one=onetoone)
model.to(device)
print(model)

# define new loss and optimiser 
total_params = count_parameters(model)
print('total param count %i' % total_params)
# %%

exp_dir = '/home/lucy/spikingPC/results/Jan-24-2023/ener_onetoone/'
saved_dict = model_result_dict_load(exp_dir + 'onelayer_rec_best.pth.tar')

model.load_state_dict(saved_dict['state_dict'])

# %%
# get params and put into dict
param_names = []
param_dict = {}
for name, param in model.named_parameters():
    if param.requires_grad:
        param_names.append(name)
        param_dict[name] = param.detach().cpu().numpy()

print(param_names)

# %%
# plot p to r weights
fig, axs = plt.subplots(1, 10, figsize=(35, 3))
for i in range(10):
    sns.heatmap(model.rout2rin.weight[:, 10 * i:(i + 1) * 10].detach().cpu().numpy().mean(axis=1).reshape(28, 28),
                ax=axs[i])
plt.title('r_out weights to r_in by class')

# plt.show()
plt.savefig(exp_dir + 'r_out weights to r_in by class')
plt.close()


# %%
# get all hidden states
def get_all_analysis_data(trained_model):
    trained_model.eval()
    test_loss = 0
    correct = 0

    hiddens_all_ = []
    preds_all_ = []

    # for data, target in test_loader:
    for i, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(-1, IN_dim)

        with torch.no_grad():
            trained_model.eval()
            hidden = trained_model.init_hidden(data.size(0))

            log_softmax_outputs, hidden = trained_model.inference(data, hidden, T)
            hiddens_all_.append(hidden)

            test_loss += F.nll_loss(log_softmax_outputs[-1], target, reduction='sum').data.item()

            pred = log_softmax_outputs[-1].data.max(1, keepdim=True)[1]
            preds_all_.append(pred)

        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        torch.cuda.empty_cache()

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        test_acc))
    return hiddens_all_, preds_all_


hiddens_all, preds_all = get_all_analysis_data(model)

# %%
# get all hiddens and corresponding pred, target, and images into dict
target_all = testdata.targets.data.numpy()
images = testdata.data.data.numpy()


def get_states(hiddens_all_: list, idx: int, hidden_dim_: int, T=20):
    """
    get a particular internal state depending on index passed to hidden
    :param hidden_dim_: the size of a state, eg. num of r or p neurons
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

    return all_states.transpose(0, 2, 1, 3).reshape(10000, 20, hidden_dim_)


# %%
# get spks from r and p for plotting a sequence
r_spk_all = get_states(hiddens_all, 1, hidden_dim[1])
p_spk_all = get_states(hiddens_all, 4, hidden_dim[0])
# get necessary weights
p2r_w = model.rout2rin.weight.detach().cpu().numpy()
r_rec_w = model.r_in_rec.rec_w.weight.detach().cpu().numpy()

print(r_spk_all.shape)
print(p_spk_all.shape)

# plot example seq
sample_no = 0
fig, axs = plt.subplots(4, 20, figsize=(80, 20))  # p spiking, r spiking, rec drive from p, rec drive from r
# axs[0].imshow(images[sample_no, :, :])
# axs[0].set_title('class %i, prediction %i' % (target_all[sample_no], preds_all[sample_no]))
for t in range(T):
    # p spiking
    axs[0][t].imshow(p_spk_all[sample_no, t, :].reshape(10, int(hidden_dim[0] / 10)))
    axs[0][t].axis('off')

    # drive from p to r
    pos1 = axs[1][t].imshow((p2r_w @ p_spk_all[sample_no, t, :]).reshape(28, 28))
    fig.colorbar(pos1, ax=axs[1][t], shrink=0.3)
    axs[1][t].axis('off')

    # r spiking
    axs[2][t].imshow(r_spk_all[sample_no, t, :].reshape(28, 28))
    axs[2][t].axis('off')

    # drive from r to r
    pos2 = axs[3][t].imshow((r_rec_w @ r_spk_all[sample_no, t, :]).reshape(28, 28))
    fig.colorbar(pos2, ax=axs[3][t], shrink=0.3)
    axs[3][t].axis('off')

plt.title('example sequence p spk, p2r drive, r spk, r2r drive')
plt.tight_layout()
# plt.show()
plt.savefig(exp_dir + 'example sequence p spk, p2r drive, r spk, r2r drive')
plt.close()

# %%
# plot energy consumption in network with two consecutive images
sample_image_nos = [10, 15]
continuous_seq_hiddens = []
with torch.no_grad():
    model.eval()
    hidden = model.init_hidden(images[sample_image_nos[0]].size(0))

    _, hidden1 = model.inference(images[sample_image_nos[0]], hidden, int(T / 2))
    continuous_seq_hiddens.append(hidden1)
    # present second stimulus without reset
    _, hidden2 = model.inference(images[sample_image_nos[1]], hidden1, int(T / 2))
    continuous_seq_hiddens.append(hidden2)


# compute energy consumption
def get_energy(hidden_):
    """
    given hidden list, compute energy of some seq length
    :param hidden_: hidden list containing mem, spk, thre
    :return: array of energy consumption during seq
    """

    seq_t = len(hidden_)
    energy_log = []

    for t in range(seq_t):
        energy = torch.norm(hidden_[t][0], p=1) + torch.norm(hidden_[t][3], p=1)
        energy_log.append(energy)

    return energy_log


energy1 = get_energy(continuous_seq_hiddens[0])
energy2 = get_energy(continuous_seq_hiddens[1])

continuous_energy = np.concatenate((energy1, energy2))

fig = plt.figure()
plt.plot(np.arange(T), continuous_energy)
plt.title('energy consumption two continuously presented images')
plt.show()
