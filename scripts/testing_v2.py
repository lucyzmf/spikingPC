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

exp_dir = '/home/lucy/spikingPC/results/Feb-01-2023/curr1530_withener_outmemconstantdecay/'
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
    sns.heatmap(model.rout2rin.weight[:, num_readout * i:(i + 1) * num_readout].detach().cpu().numpy().sum(axis=1).reshape(28, 28),
                ax=axs[i])
plt.title('r_out weights to r_in by class')

# plt.show()
plt.savefig(exp_dir + 'r_out weights to r_in by class')
plt.close()

# %%
# plot p to r weights for class 0
fig, axs = plt.subplots(1, 10, figsize=(35, 3))
for i in range(10):
    sns.heatmap(model.rout2rin.weight[:, i].detach().cpu().numpy().reshape(28, 28),
                ax=axs[i])
plt.title('r_out weights to r_in class 0')

# plt.show()
plt.savefig(exp_dir + 'r_out weights to r_in class 0')
plt.close()

# %%
# get all hidden states
def get_all_analysis_data(trained_model):
    trained_model.eval()
    test_loss = 0
    correct = 0

    hiddens_all_ = []
    preds_all_ = []
    data_all_ = []  # get transformed data 

    # for data, target in test_loader:
    for i, (data, target) in enumerate(test_loader):
        data_all_.append(data.data)
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
    
    data_all_ = torch.stack(data_all_).reshape(10000, 28, 28)
    
    return hiddens_all_, preds_all_, data_all_


hiddens_all, preds_all, images_all = get_all_analysis_data(model)
target_all = testdata.targets.data


# %%
# get all hiddens and corresponding pred, target, and images into dict

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
p_spk_all = get_states(hiddens_all, 5, hidden_dim[0])
# get necessary weights
p2r_w = model.rout2rin.weight.detach().cpu().numpy()
r_rec_w = model.r_in_rec.rec_w.weight.detach().cpu().numpy()

print(r_spk_all.shape)
print(p_spk_all.shape)

# plot example seq
sample_no = 3
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
sample_image_nos = [3, 4]
print(target_all[sample_image_nos[0]])
print(target_all[sample_image_nos[1]])
continuous_seq_hiddens = []
with torch.no_grad():
    model.eval()
    hidden = model.init_hidden(images_all[sample_image_nos[0], :, :].view(-1, IN_dim).size(0))

    _, hidden1 = model.inference(images_all[sample_image_nos[0], :, :].view(-1, IN_dim).to(device), hidden, int(T/2))
    continuous_seq_hiddens.append(hidden1)
    # present second stimulus without reset
    # hidden1[-1] = model.init_hidden(images[sample_image_nos[0], :, :].view(-1, IN_dim).size(0))
    _, hidden2 = model.inference((images_all[sample_image_nos[1], :, :]).view(-1, IN_dim).to(device), hidden1[-1], int(T/2))
    continuous_seq_hiddens.append(hidden2)

# normal sequence for comparison
normal_seq = []
with torch.no_grad():
    model.eval()
    hidden = model.init_hidden(images_all[sample_image_nos[0], :, :].view(-1, IN_dim).size(0))

    _, hidden1 = model.inference(images_all[sample_image_nos[0], :, :].view(-1, IN_dim).to(device), hidden, T)
    normal_seq.append(hidden1)

# compute energy consumption
def get_energy(hidden_, alpha=1/3 ):
    """
    given hidden list, compute energy of some seq length
    :param alpha: scaler for mem activity vs synaptic transmission
    :param hidden_: hidden list containing mem, spk, thre
    :return: array of energy consumption during seq
    """

    seq_t = len(hidden_)
    energy_log = []

    for t in range(seq_t):
        # spk output
        activity = (hidden_[t][1].mean() + hidden_[t][5].mean()).cpu().numpy()
        # synaptic transmission
        synaptic_transmission = ((torch.abs(model.r_in_rec.rec_w.weight) @ torch.abs(hidden_[t][1].T)).mean() +
                                 (torch.abs(model.rin2rout.weight) @ torch.abs(hidden_[t][1].T)).mean() +
                                 (torch.abs(model.rout2rin.weight) @ torch.abs(hidden_[t][5].T)).mean() +
                                 (torch.abs(model.r_out_rec.rec_w.weight) @ torch.abs(hidden_[t][5].T)).mean()).detach().cpu().numpy()
        energy = alpha * activity + (1 - alpha) * synaptic_transmission
        energy_log.append(energy)

    energy_log = np.hstack(energy_log)

    return energy_log


# %%

energy1 = get_energy(continuous_seq_hiddens[0])
energy2 = get_energy(continuous_seq_hiddens[1])

continuous_energy = np.concatenate((energy1, energy2))

energy_normal_sequence = get_energy(normal_seq[0])

fig = plt.figure(figsize=(10, 3))
plt.plot(np.arange(T), continuous_energy, label='continuous')
plt.plot(np.arange(T), energy_normal_sequence, label='normal')
plt.title('energy consumption two continuously presented images vs normal sequence')
plt.legend()
# plt.show()
plt.savefig(exp_dir + 'energy consumption two continuously presented images')
plt.close()

# %%
# decompose input signals in normal sequence 
normalseq_hidden = normal_seq[0]
r_from_p_ex = []
r_from_p_inh = [] 
p_from_r_ex = []
p_from_r_inh = []
p_from_p_ex = []
p_from_p_inh = []
r_from_r_ex = []
r_from_r_inh = []
r_spk_rate = []
p_spk_rate =[]

timesteps = len(normalseq_hidden)

for t in np.arange(1, timesteps):
    r_from_p_ex.append(((model.rout2rin.weight.ge(0) * model.rout2rin.weight) @ normalseq_hidden[t-1][5].T).mean().detach().cpu().numpy())
    r_from_p_inh.append(((model.rout2rin.weight.le(0) * model.rout2rin.weight) @ normalseq_hidden[t-1][5].T).mean().detach().cpu().numpy())
    p_from_r_ex.append(((model.rin2rout.weight.ge(0) * model.rin2rout.weight) @ normalseq_hidden[t][1].T).mean().detach().cpu().numpy())
    p_from_r_inh.append(((model.rin2rout.weight.le(0) * model.rin2rout.weight) @ normalseq_hidden[t][1].T).mean().detach().cpu().numpy())
    p_from_p_ex.append(((model.r_out_rec.rec_w.weight.ge(0) * model.r_out_rec.rec_w.weight) @ normalseq_hidden[t-1][5].T).mean().detach().cpu().numpy())
    p_from_p_inh.append(((model.r_out_rec.rec_w.weight.le(0) * model.r_out_rec.rec_w.weight) @ normalseq_hidden[t-1][5].T).mean().detach().cpu().numpy())
    r_from_r_ex.append(((model.r_in_rec.rec_w.weight.ge(0) * model.r_in_rec.rec_w.weight) @ normalseq_hidden[t-1][1].T).mean().detach().cpu().numpy())
    r_from_r_inh.append(((model.r_in_rec.rec_w.weight.ge(0) * model.r_in_rec.rec_w.weight) @ normalseq_hidden[t-1][1].T).mean().detach().cpu().numpy())
    r_spk_rate.append(normalseq_hidden[t][1].mean().detach().cpu().numpy())
    p_spk_rate.append(normalseq_hidden[t][5].mean().detach().cpu().numpy())



fig = plt.figure()
x = np.arange(1, timesteps)
# plt.plot(x, np.hstack(r_from_p_ex), label='r_from_p_ex')
# plt.plot(x, np.hstack(r_from_p_inh), label='r_from_p_inh')
# plt.plot(x, np.hstack(p_from_r_ex), label='p_from_r_ex')
# plt.plot(x, np.hstack(p_from_r_inh), label='p_from_r_inh')
# plt.plot(x, np.hstack(p_from_p_ex), label='p_from_p_ex')
# plt.plot(x, np.hstack(p_from_p_inh), label='p_from_p_inh')
# plt.plot(x, np.hstack(r_from_r_ex), label='r_from_r_ex')
# plt.plot(x, np.hstack(r_from_r_inh), label='r_from_r_inh')
plt.plot(x, (np.hstack(r_from_p_ex)+np.hstack(r_from_p_inh)), label='p2r input')
plt.plot(x, (np.hstack(r_from_r_ex)+np.hstack(r_from_r_inh)), label='r2r input')
plt.plot(x, np.hstack(r_spk_rate), label='r spk rate', linestyle='dashed')
# plt.plot(x, np.hstack(p_spk_rate), label='p spk rate', linestyle='dashed')

plt.legend()
# plt.title('mean exhitatory and inhibitory signals by source and target type')
# plt.show()
plt.savefig(exp_dir + 'mean exhitatory and inhibitory signals by source and target type')
plt.close()

# %%
# sns.scatterplot(x=(np.hstack(r_from_p_ex)+np.hstack(r_from_p_inh))[2:], y=np.hstack(r_spk_rate)[2:])
# plt.show()




# %%
# plot continuous sequence spike pattern
fig, axs = plt.subplots(4, 20, figsize=(80, 20))  # p spiking, r spiking, rec drive from p, rec drive from r
# axs[0].imshow(images[sample_no, :, :])
# axs[0].set_title('class %i, prediction %i' % (target_all[sample_no], preds_all[sample_no]))
for t in range(T):
    if t<10:
        hidden = continuous_seq_hiddens[0]
    else:
        hidden = continuous_seq_hiddens[1]
    # p spiking
    axs[0][t].imshow(hidden[t%10][5][0].detach().cpu().numpy().reshape(10, int(hidden_dim[0] / 10)))
    axs[0][t].axis('off')

    # drive from p to r
    pos1 = axs[1][t].imshow((p2r_w @ hidden[t%10][5][0].detach().cpu().numpy()).reshape(28, 28))
    fig.colorbar(pos1, ax=axs[1][t], shrink=0.3)
    axs[1][t].axis('off')

    # r spiking
    axs[2][t].imshow(hidden[t%10][1][0].detach().cpu().numpy().reshape(28, 28))
    axs[2][t].axis('off')

    # drive from r to r
    pos2 = axs[3][t].imshow((r_rec_w @ hidden[t%10][1][0].detach().cpu().numpy()).reshape(28, 28))
    fig.colorbar(pos2, ax=axs[3][t], shrink=0.3)
    axs[3][t].axis('off')

plt.title('continuous seq with change in img p spk, p2r drive, r spk, r2r drive')
plt.tight_layout()
# plt.show()
plt.savefig(exp_dir + 'continuous change in image p spk, p2r drive, r spk, r2r drive')
plt.close()

# %%
# weight matrix for p2p

fig = plt.figure()
abs_max = np.max(np.abs(model.r_out_rec.rec_w.weight.detach().cpu().numpy()))
sns.heatmap(model.r_out_rec.rec_w.weight.detach().cpu().numpy(), vmax=abs_max, vmin=-abs_max, cmap='icefire')
plt.title('p2p weights')
# plt.show()
plt.savefig(exp_dir + 'p2p weights')
plt.close()

# %%
# weight matrix for r2r

fig = plt.figure()
abs_max = np.max(np.abs(model.r_in_rec.rec_w.weight.detach().cpu().numpy()))
sns.heatmap(model.r_in_rec.rec_w.weight.detach().cpu().numpy(), vmax=abs_max, vmin=-abs_max, cmap='icefire')
# plt.show()
plt.title('r2r weights')
plt.savefig(exp_dir + 'r2r weights')
plt.close()

# %%
# weight matrix for r2p

fig = plt.figure()
abs_max = np.max(np.abs(model.rin2rout.weight.detach().cpu().numpy()))
sns.heatmap(model.rin2rout.weight.detach().cpu().numpy(), vmax=abs_max, vmin=-abs_max, cmap='icefire')
# plt.show()
plt.title('r2p weights')
plt.savefig(exp_dir + 'r2p weights')
plt.close()


# %%

# cluster neuron time constants 
def get_real_constants(pseudo_constants):
    return -1/np.log(1/(1 + np.exp(-pseudo_constants)))
    # return pseudo_constants

fig = plt.figure()
sns.scatterplot(x=get_real_constants(model.r_in_rec.tau_adp.detach().cpu().numpy()), y=get_real_constants(model.r_in_rec.tau_m.detach().cpu().numpy()), label='r neurons')
sns.scatterplot(x=get_real_constants(model.r_out_rec.tau_adp.detach().cpu().numpy()), y=get_real_constants(model.r_out_rec.tau_m.detach().cpu().numpy()), label='p neurons')
plt.title('time constants scatter')
plt.legend()
# plt.show()
plt.savefig(exp_dir + 'time constants scatter')
plt.close()


# %%
# time constant of output neuron 
fig = plt.figure()
plt.plot(np.arange(10), model.output_layer.tau_m.detach().cpu().numpy())
plt.title('output neuron time constants')
# plt.show()
plt.savefig(exp_dir + 'output neuron time constants')
plt.close()


# %%
# plot correlation between mismatch between p and r at t-1 and energy at t 
# idea is that if previous mismatch in prediction, error corrects it after 
# first batch
prediction_t = p_spk_all[:200, :, :] @ p2r_w.T 
mismatch_t = torch.norm(torch.tensor((r_spk_all[:200, 1:, :] - prediction_t[:, :19, :])), p=1, dim=2).T


def get_energy_batch(hidden_, alpha=1 ):
    """
    given hidden list, compute energy of some seq length
    :param alpha: scaler for mem activity vs synaptic transmission
    :param hidden_: hidden list containing mem, spk, thre
    :return: array of energy consumption during seq
    """

    seq_t = len(hidden_)
    energy_log = []

    for t in range(seq_t):
        # spk output
        activity = (hidden_[t][1].mean(dim=-1) + hidden_[t][5].mean(dim=-1)).cpu().numpy()
        # synaptic transmission
        synaptic_transmission = ((torch.abs(model.r_in_rec.rec_w.weight) @ torch.abs(hidden_[t][1].T)).mean(dim=0) +
                                 (torch.abs(model.rin2rout.weight) @ torch.abs(hidden_[t][1].T)).mean(dim=0) +
                                 (torch.abs(model.rout2rin.weight) @ torch.abs(hidden_[t][5].T)).mean(dim=0) +
                                 (torch.abs(model.r_out_rec.rec_w.weight) @ torch.abs(hidden_[t][5].T)).mean(dim=0)).detach().cpu().numpy()
        energy = alpha * activity + (1 - 0) * synaptic_transmission
        energy_log.append(energy)

    energy_log = np.vstack(energy_log)

    return energy_log

energy_t1 = get_energy_batch(hiddens_all[0])

# %%
fig = plt.figure()
for i in range(50):
    sns.scatterplot(x=F.normalize(mismatch_t[:, i], dim=0), y=F.normalize(torch.tensor(energy_t1[1:, i]), dim=0))
plt.xlabel('mismatch t')
plt.ylabel('energy t+1')
plt.title('corr between mismatch in prediction and energy at next time step')
# plt.show()
plt.savefig(exp_dir + 'corr between mismatch in prediction and energy at next time step')
plt.close()
# %%
# see if can find class specific r lateral inhibition 
r2p_w = model.rin2rout.weight 

fig, axs = plt.subplots(1, 10, figsize=(35, 3))
for i in range(10):
    r_idx = (r2p_w[i*num_readout:(i+1)*num_readout, :]>0.06).nonzero()[:, 1].cpu().numpy()
    sns.heatmap(model.r_in_rec.rec_w.weight.detach().cpu().numpy()[:, r_idx].sum(axis=1).reshape(28, 28), ax=axs[i])

plt.title('sum recurrent r to r connectivity of r neurons that strongly contribute to one class')

# plt.show()
plt.savefig(exp_dir + 'sum recurrent r to r connectivity of r neurons that strongly contribute to one class')
plt.close()

# %%
# 
fig, axs = plt.subplots(1, 10, figsize=(35, 3))
for i in range(10):
    r_idx = (r2p_w[i*num_readout:(i+1)*num_readout, :]<-0.1).nonzero()[:, 1].cpu().numpy()
    sns.heatmap(model.r_in_rec.rec_w.weight.detach().cpu().numpy()[:, r_idx].sum(axis=1).reshape(28, 28), ax=axs[i])

plt.title('sum recurrent r to r connectivity of r neurons that inhibits p of one class')

# plt.show()
plt.savefig(exp_dir + 'sum recurrent r to r connectivity of r neurons that inhibits p of one class')
plt.close()

# %%
