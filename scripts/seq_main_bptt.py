# %%
# this script contains training with sequence data
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
import IPython.display as ipd
from tqdm import tqdm

from network_class import *
from utils import *
from sequence_dataset import *
from test_function import test_seq

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# set seed
torch.manual_seed(999)

# wandb login
wandb.login(key='25f10546ef384a6f1ab9446b42d7513024dea001')
wandb.init(project="spikingPC_onelayer", entity="lucyzmf")
# wandb.init(mode="disabled")

# add wandb.config
config = wandb.config

# network hypers
config.adap_neuron = True  # whether use adaptive neuron or not
config.dp = 0.5
config.onetoone = True
config.num_readout = 10

# loss hypers
config.clf_alpha = 1  # proportion of clf loss
config.energy_alpha = 1  # - config.clf_alpha

# training alg hypers
config.lr = 1e-3
config.alg = 'bptt'
alg = config.alg
config.k_updates = 40

# seq data set config
config.seq_data = True  # whether applies sequence data
seq_data = config.seq_data
config.seq_type = 'pred'  # whether change in digit is predictable (eg acending order) or unpredictable
config.seq_len = 40  # sequence length
config.random_switch = False  # predictable or random switch time
config.switch_time = [config.seq_len / 2]  # if not random switch, provide switch time
config.num_switch = 1  # used when random switch=T

# training parameters
T = config.seq_len
K = config.k_updates  # k_updates is num updates per sequence
omega = int(T / K)  # update frequency
clip = 1.
log_interval = 20
epoch = 10
n_classes = 10

config.exp_name = config.alg + '_ener_fpttalpha02_curr0'

# experiment name 
exp_name = config.exp_name
energy_penalty = True
# checkpoint file name
check_fn = 'onelayer_rec_best.pth.tar'
# experiment date and name 
today = date.today()
# checkpoint file prefix 
prefix = '../results/' + today.strftime("%b-%d-%Y") + '/' + exp_name + '/'

# create exp path 
if os.path.exists(prefix):
    raise Exception('experiment dir already exist')
else:
    os.makedirs(prefix)

# %%
###############################################################
# IMPORT DATASET 
###############################################################

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))])

batch_size = 256

traindata = torchvision.datasets.MNIST(root='./data', train=True,
                                       download=True, transform=transform)

testdata = torchvision.datasets.MNIST(root='./data', train=False,
                                      download=True, transform=transform)

# generate sequence dataset
if config.seq_type == 'pred':
    seq_train = SequenceDatasetPredictable(traindata, traindata.targets, config.seq_len, config.random_switch,
                                           config.switch_time, config.num_switch)
    seq_test = SequenceDatasetPredictable(testdata, testdata.targets, config.seq_len, config.random_switch,
                                          config.switch_time, config.num_switch)
else:
    seq_train = SequenceDataset(traindata, traindata.targets, config.seq_len, config.random_switch,
                                config.switch_time, config.num_switch)
    seq_test = SequenceDataset(testdata, testdata.targets, config.seq_len, config.random_switch,
                               config.switch_time, config.num_switch)

# %%
train_loader = torch.utils.data.DataLoader(seq_train, batch_size=batch_size,
                                           shuffle=False, num_workers=3)
test_loader = torch.utils.data.DataLoader(seq_test, batch_size=batch_size,
                                          shuffle=False, num_workers=3)

for batch_idx, (data, target) in enumerate(train_loader):
    print(data.shape)
    print(target.shape)
    break


# %%
def train_bptt_seq(epoch, batch_size, log_interval,
                   train_loader, model,
                   time_steps, k_updates, omega, optimizer,
                   clf_alpha, energy_alpha, clip, lr):
    train_loss = 0
    total_clf_loss = 0
    total_regularizaton_loss = 0
    total_energy_loss = 0
    correct = 0
    model.train()

    # for each batch
    for batch_idx, (data, target) in enumerate(train_loader):

        # to device and reshape
        data, target = data.to(device), target.to(device)
        data = data.view(-1, time_steps, model.in_dim)

        B = target.size()[0]

        for t in range(time_steps):

            if t == 0:
                h = model.init_hidden(data.size(0))
            elif t % omega == 0:
                h = tuple(v.detach() for v in h)

            o, h = model.forward(data[:, t, :], h)

            pred = o.data.max(1, keepdim=True)[1]
            correct += pred.eq(target[:, t].data.view_as(pred)).cpu().sum()

            optimizer.zero_grad()

            # classification loss
            clf_loss = (t % int(k_updates / 2) + 1) / (int(k_updates / 2)) * F.nll_loss(o, target[:, t])
            # clf_loss = snr*F.cross_entropy(output, target,reduction='none')
            # clf_loss = torch.mean(clf_loss)

            # mem potential loss take l1 norm / num of neurons /batch size
            energy = (torch.norm(h[1], p=1) + torch.norm(h[5], p=1)) / B / sum(model.hidden_dims)

            # overall loss
            loss = clf_alpha * clf_loss + energy_alpha * energy

        loss.backward()

        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        train_loss += loss.item()
        total_clf_loss += clf_loss.item()
        total_energy_loss += energy.item()

        if batch_idx > 0 and batch_idx % log_interval == (log_interval - 1):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tlr: {:.6f}\ttrain acc:{:.4f}\tLoss: {:.6f}\
                \tClf: {:.6f}\tReg: {:.6f}\tFr_p: {:.6f}\tFr_r: {:.6f}'.format(
                epoch, batch_idx * batch_size, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), lr, 100 * correct / (log_interval * B * time_steps),
                       train_loss / log_interval / B,
                       total_clf_loss / log_interval / B, total_regularizaton_loss / log_interval / B,
                       model.fr_p / time_steps / log_interval,
                       model.fr_r / time_steps / log_interval))

            wandb.log({
                'clf_loss': total_clf_loss / log_interval / B,
                'train_acc': 100 * correct / (log_interval * B * time_steps),
                'regularisation_loss': total_regularizaton_loss / log_interval / B,
                'energy_loss': total_energy_loss / log_interval / B,
                'total_loss': train_loss / log_interval / B,
                'pred spiking freq': model.fr_p / time_steps / log_interval,  # firing per time step
                'rep spiking fr': model.fr_r / time_steps / log_interval,
            })

            train_loss = 0
            total_clf_loss = 0
            total_regularizaton_loss = 0
            total_energy_loss = 0
            correct = 0
        # model.network.fr = 0
        model.fr_p = 0
        model.fr_r = 0


# %%
# ##############################################################
# DEFINE NETWORK
# ##############################################################

# set input and t param
IN_dim = 784
hidden_dim = [10 * config.num_readout, 784]

# define network
model = SnnNetworkSeq(IN_dim, hidden_dim, n_classes, is_adapt=config.adap_neuron, one_to_one=config.onetoone,
                      dp_rate=config.dp)
model.to(device)
print(model)

# define new loss and optimiser 
total_params = count_parameters(model)
print('total param count %i' % total_params)

# define optimiser
optimizer = optim.Adamax(model.parameters(), lr=config.lr, weight_decay=0.0001)
# reduce the learning after 20 epochs by a factor of 10
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

# %%
###############################################################################################
##########################          Training and testin         ###############################
###############################################################################################

# untrained network
test_loss, acc1 = test_seq(model, test_loader, T)

# %%

epochs = 20
all_test_losses = []
best_acc1 = 20

wandb.watch(model, log_freq=100)

for epoch in range(epochs):
    train_bptt_seq(epoch, batch_size, log_interval, train_loader,
                   model, T, K, omega, optimizer,
                   config.clf_alpha, config.energy_alpha, clip, config.lr)

    test_loss, acc1 = test_seq(model, test_loader, T)

    scheduler.step()

    # remember best acc@1 and save checkpoint
    is_best = acc1 > best_acc1
    best_acc1 = max(acc1, best_acc1)

    if is_best:
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            # 'oracle_state_dict': oracle.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
            # 'oracle_optimizer' : oracle_optim.state_dict(),
        }, is_best, prefix=prefix, filename=check_fn)

    all_test_losses.append(test_loss)

# %%
