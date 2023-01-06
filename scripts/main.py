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

# wandb login
wandb.login(key='25f10546ef384a6f1ab9446b42d7513024dea001')
wandb.init(project="spikingPC", entity="lucyzmf")
# wandb.init(mode="disabled")

# add wandb.config
config = wandb.config
config.spike_loss = True  # whether use energy penalty on spike or on mem potential 
config.adap_neuron = True  # whether use adaptive neuron or not
config.l1_lambda = 1  # weighting for l1 reg
config.clf_alpha = 1 # proportion of clf loss 
config.energy_alpha = 1 #1-config.clf_alpha 

# experiment name 
exp_name = 'exp_5_adp_spk_loss1_clf1_l11'
energy_penalty = True
spike_loss = config.spike_loss
adap_neuron = config.adap_neuron
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

batch_size = 128

traindata = torchvision.datasets.MNIST(root='./data', train=True,
                                       download=True, transform=transform)

testdata = torchvision.datasets.MNIST(root='./data', train=False,
                                      download=True, transform=transform)

# data loading 
train_loader = torch.utils.data.DataLoader(traindata, batch_size=batch_size,
                                           shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(testdata, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

# check data loading correctness
for batch_idx, (data, target) in enumerate(train_loader):
    print(data.shape)
    break

# %%
# set input and t param
IN_dim = 28 * 28
T = 20  # sequence length, reading from the same image T times 


# if apply first layer drop out, creates sth similar to poisson encoding

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
            hidden = model.init_hidden(data.size(0))

            prob_outputs, log_softmax_outputs, hidden = model(data, hidden, T)

            test_loss += F.nll_loss(log_softmax_outputs[-1], target, reduction='sum').data.item()
            pred = prob_outputs[-1].data.max(1, keepdim=True)[1]

        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        torch.cuda.empty_cache()

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    wandb.log({
        'test_loss': test_loss,
        'test_acc': test_acc
    })
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        test_acc))
    return test_loss, 100. * correct / len(test_loader.dataset)


###############################################################################################
##########################          Train function             ###############################
###############################################################################################
# training parameters
K = T  # K is num updates per sequence
omega = int(T / K)  # update frequency
clip = 1.
log_interval = 100
lr = 1e-3
epoch = 10
n_classes = 10


# train function for one epoch
def train(train_loader, n_classes, model, named_params):
    global steps
    global estimate_class_distribution

    train_loss = 0
    total_clf_loss = 0
    total_regularizaton_loss = 0
    total_energy_loss = 0
    total_l1_loss = 0
    model.train()

    # for each batch 
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(-1, IN_dim)

        B = target.size()[0]

        for p in range(T):

            if p == 0:
                h = model.init_hidden(data.size(0))
            elif p % omega == 0:
                h = tuple(v.detach() for v in h)

            o, h, hs = model.network.forward(data, h)

            # change readout method to implement read out in same layer 
            prob_out = F.softmax(h[1][:, :10], dim=1) # take the first 10 neurons for read out 
            output = F.log_softmax(h[1][:, :10], dim=1)

            if p % omega == 0 and p > 0:
                optimizer.zero_grad()

                # classification loss
                clf_loss = (p + 1) / (K) * F.nll_loss(output, target)
                # clf_loss = snr*F.cross_entropy(output, target,reduction='none')
                # clf_loss = torch.mean(clf_loss)

                # regularizer loss                     
                regularizer = get_regularizer_named_params(named_params, _lambda=1.0)

                if spike_loss:
                    # energy loss: batch mean spiking * weighting param
                    energy = h[1].mean()  # * 0.1
                else:
                    # mem potential loss take l1 norm / num of neurons /batch size
                    energy = torch.norm(h[0], p=1) / B / 784

                # l1 loss on rec weights 
                l1_norm = torch.linalg.norm(model.network.snn_layer.layer1_x.weight)

                # overall loss    
                if energy_penalty:
                    loss = config.clf_alpha*clf_loss + regularizer + config.energy_alpha*energy \
                           + config.l1_lambda * l1_norm
                else:
                    loss = clf_loss + regularizer

                loss.backward()

                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

                optimizer.step()
                post_optimizer_updates(named_params)

                train_loss += loss.item()
                total_clf_loss += clf_loss.item()
                total_regularizaton_loss += regularizer  # .item()
                total_energy_loss += energy.item()
                total_l1_loss += l1_norm.item()

        if batch_idx > 0 and batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tlr: {:.6f}\tLoss: {:.6f}\
                \tClf: {:.6f}\tReg: {:.6f}\tFr: {:.6f}'.format(
                epoch, batch_idx * batch_size, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), lr, train_loss / log_interval,
                       total_clf_loss / log_interval, total_regularizaton_loss / log_interval,
                       model.network.fr / T / log_interval))

            wandb.log({
                'clf_loss': total_clf_loss / log_interval,
                'regularisation_loss': total_regularizaton_loss / log_interval,
                'energy_loss': total_energy_loss / log_interval,
                'l1_loss': config.l1_lambda * total_l1_loss / log_interval,
                'total_loss': train_loss / log_interval,
                'network spiking freq': model.network.fr / T / log_interval  # firing per time step
            })

            train_loss = 0
            total_clf_loss = 0
            total_regularizaton_loss = 0
            total_energy_loss = 0
            total_l1_loss = 0
        model.network.fr = 0


# %%
###############################################################
# DEFINE NETWORK
###############################################################


# define network
model = OneLayerSeqModel(IN_dim, 784, n_classes, is_rec=True, is_LTC=False, is_adapt=adap_neuron)
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
###############################################################################################
##########################          Training and testin         ###############################
###############################################################################################

# untrained network
test_loss, acc1 = test(model, test_loader)

# %%

epochs = 10
named_params = get_stats_named_params(model)
all_test_losses = []
best_acc1 = 20

wandb.watch(model, log_freq=100)

wandb.config = {
    'learning_rate': lr,
    'sequence_len': T,
    'epochs': epochs,
    'update_freq': omega,
}

estimate_class_distribution = torch.zeros(n_classes, T, n_classes, dtype=torch.float)
for epoch in range(epochs):
    train(train_loader, n_classes, model, named_params)

    reset_named_params(named_params)

    test_loss, acc1 = test(model, test_loader)

    scheduler.step()

    # remember best acc@1 and save checkpoint
    is_best = acc1 > best_acc1
    best_acc1 = max(acc1, best_acc1)

    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        # 'oracle_state_dict': oracle.state_dict(),
        'best_acc1': best_acc1,
        'optimizer': optimizer.state_dict(),
        # 'oracle_optimizer' : oracle_optim.state_dict(),
    }, is_best, prefix=prefix, filename=check_fn)

    all_test_losses.append(test_loss)

test_loss, acc1 = test(model, test_loader)
# %%
