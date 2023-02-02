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
model_constant = SnnNetwork(IN_dim, hidden_dim, n_classes, is_adapt=adap_neuron, one_to_one=onetoone)
model_constant.to(device)
print(model_constant)

# define network
model_trainable = SnnNetwork(IN_dim, hidden_dim, n_classes, is_adapt=adap_neuron, one_to_one=onetoone)
model_trainable.to(device)

# define new loss and optimiser
total_params = count_parameters(model_trainable)
print('total param count %i' % total_params)

# %%
# load different models
exp_dir_constant = '/home/lucy/spikingPC/results/Jan-31-2023/spkener_outmemconstantdecay/'
saved_dict1 = model_result_dict_load(exp_dir_constant + 'onelayer_rec_best.pth.tar')

model_constant.load_state_dict(saved_dict1['state_dict'])

exp_dir_trainable = '/home/lucy/spikingPC/results/Jan-31-2023/spkener_outmemdecay/'
saved_dict2 = model_result_dict_load(exp_dir_trainable + 'onelayer_rec_best.pth.tar')

model_trainable.load_state_dict(saved_dict2['state_dict'])

# %%
# get params and put into dict
param_names = []
param_dict = {}
for name, param in model_trainable.named_parameters():
    if param.requires_grad:
        param_names.append(name)
        param_dict[name] = param.detach().cpu().numpy()

print(param_names)

# %%
# compare strength of inhibitory weights from p to r per class
inhibition_strength_per_class = {'class': np.concatenate((np.arange(10), np.arange(10))),
                                 'inhibition': [], 'model type': []}
for i in range(10 * 2):
    if i < 10:
        model = model_trainable
        model_type = 'trainable'
    else:
        model = model_constant
        model_type = 'constant'
    w = model.rout2rin.weight[:, num_readout * (i % 10):((i % 10) + 1) * num_readout].detach()
    inhibition_strength_per_class['inhibition'].append(((w < 0) * w).sum().cpu().item())
    inhibition_strength_per_class['model type'].append(model_type)

inhibition_strength_df = pd.DataFrame.from_dict(inhibition_strength_per_class)

fig = plt.figure()
sns.barplot(inhibition_strength_df, x='class', y='inhibition', hue='model type')
plt.title('p to r inhibitory weight sum')
plt.show()

# %%
# compare acc at each time step of prediction
acc_per_step = {
    'time step': [],
    'acc': [],
    'model type': [],
    'condition': []  # constant vs change in stimulus
}

# get all predictions for normal sequences
hiddens_b, preds_b, images_b = get_all_analysis_data(model_trainable, test_loader, device, IN_dim, T)
hiddens_l, preds_l, images_l = get_all_analysis_data(model_constant, test_loader, device, IN_dim, T)


# get predictions from list of logsoftmax outputs per time step
def get_predictions(preds_all, bsz, T):
    preds_all_by_t = []
    for b in range(int(10000 / bsz)):
        preds_per_batch = []
        for t in range(T):
            preds = preds_all[b][t].data.max(1, keepdim=True)[1]
            preds_per_batch.append(preds)
        preds_per_batch = (torch.stack(preds_per_batch)).transpose(1, 0)
        preds_all_by_t.append(preds_per_batch)

    preds_all_by_t = torch.stack(preds_all_by_t)

    return preds_all_by_t


preds_by_t_b = get_predictions(preds_b, bsz=batch_size, T=T).reshape(10000, 20)
preds_by_t_l = get_predictions(preds_l, bsz=batch_size, T=T).reshape(10000, 20)

# %%
# get acc for each time step
acc_t_b = [(preds_by_t_b[:, t].cpu().eq(testdata.targets.data).sum().numpy()) / 10000 * 100 for t in range(T)]
acc_t_l = [(preds_by_t_l[:, t].cpu().eq(testdata.targets.data).sum().numpy()) / 10000 * 100 for t in range(T)]

acc_per_step['time step'] = np.concatenate((np.arange(T), np.arange(T)))
acc_per_step['acc'] = np.concatenate((np.hstack(acc_t_b), np.hstack(acc_t_l)))

# %%
# change in stimulus at t=10

test_loader_split = torch.utils.data.DataLoader(testdata, batch_size=int(10000 / 2),
                                                shuffle=False, num_workers=2)


def change_in_stumuli(trained_model, test_loader_, device, IN_dim, t=10):
    """
    get all analysis data for change in stimuli half way
    :param trained_model:
    :param test_loader_: batch size is half of the data set
    :param device:
    :param IN_dim:
    :param t: time step when change happens
    :return:
    """
    trained_model.eval()
    test_loss = 0
    correct = 0

    hiddens_all_ = []
    preds_all_ = []  # predictions at all timesptes
    data_all_ = []  # get transformed data

    # for data, target in test_loader:
    for i, (data, target) in enumerate(test_loader_):
        data_all_.append(data.data)
        data, target = data.to(device), target.to(device)
        data = data.view(-1, IN_dim)

        with torch.no_grad():
            trained_model.eval()
            if i == 0:
                hidden = trained_model.init_hidden(data.size(0))
            else:
                hidden = tuple(v.detach() for v in hidden[-1])
            log_softmax_outputs, hidden = trained_model.inference(data, hidden, t)
            hiddens_all_.append(hidden)

            test_loss += F.nll_loss(log_softmax_outputs[-1], target, reduction='sum').data.item()

            pred = log_softmax_outputs[-1].data.max(1, keepdim=True)[1]
            preds_all_.append(log_softmax_outputs)

        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        torch.cuda.empty_cache()

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        test_acc))

    data_all_ = torch.stack(data_all_).reshape(10000, 28, 28)

    return hiddens_all_, preds_all_, data_all_


# %%
_, preds_change_b, _ = change_in_stumuli(model_trainable, test_loader_split, device, IN_dim)
_, preds_change_l, _ = change_in_stumuli(model_constant, test_loader_split, device, IN_dim)

preds_by_t_change_b = get_predictions(preds_change_b, 5000, T=10).squeeze()
preds_by_t_change_b = torch.hstack((preds_by_t_change_b[0, :, :], preds_by_t_change_b[1, :, :]))
preds_by_t_change_l = get_predictions(preds_change_l, 5000, T=10).squeeze()
preds_by_t_change_l = torch.hstack((preds_by_t_change_l[0, :, :], preds_by_t_change_l[1, :, :]))

acc_t_b_change = [(preds_by_t_change_b[:, t].cpu().eq(testdata.targets.data[:5000]).sum().numpy()) / 5000 * 100 for t in range(10)] + \
                [(preds_by_t_change_b[:, t].cpu().eq(testdata.targets.data[5000:]).sum().numpy()) / 5000 * 100 for t in range(10, 20)]
acc_t_l_change = [(preds_by_t_change_l[:, t].cpu().eq(testdata.targets.data[:5000]).sum().numpy()) / 5000 * 100 for t in range(10)] + \
                [(preds_by_t_change_l[:, t].cpu().eq(testdata.targets.data[5000:]).sum().numpy()) / 5000 * 100 for t in range(10, 20)]

acc_per_step['time step'] = np.concatenate((acc_per_step['time step'], np.arange(T), np.arange(T)))
acc_per_step['acc'] = np.concatenate((acc_per_step['acc'], np.hstack(acc_t_b_change), np.hstack(acc_t_l_change)))

# condition labelling
acc_per_step['model type'] = np.hstack((['trainable'] * T, ['constant'] * T) * 2)
acc_per_step['condition'] = np.hstack((['constant'] * (T * 2), ['change'] * (T * 2)))

# %%
# plot
fig = plt.figure()
sns.lineplot(acc_per_step, x='time step', y='acc', hue='model type', style='condition')
plt.show()
# %%
