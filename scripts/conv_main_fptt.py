# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio

from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision
import numpy as np
import wandb
from datetime import date
import os

from conv_net import *
from conv_local import *
from conv_traintest import *
from utils import *
from FTTP import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# set seed
torch.manual_seed(999)

# %%
# wandb login
wandb.login(key='25f10546ef384a6f1ab9446b42d7513024dea001')
wandb.init(project="spikingPC_conv_voltloss", entity="lucyzmf")
# wandb.init(mode="disabled")

# add wandb.config
config = wandb.config
config.adap_neuron = True  # whether use adaptive conv neuron or not
config.clf_alpha = 1  # proportion of clf loss
config.energy_alpha = 1e-4  # - config.clf_alpha
config.onetoone = True
config.lr = 1e-3
config.alg = 'conv_fptt'
alg = config.alg
config.k_updates = 10
config.dp = 0.2

# training parameters
T = 50
K = config.k_updates  # k_updates is num updates per sequence
omega = int(T / K)  # update frequency
clip = 1.
log_interval = 40
epochs = 30
n_classes = 10

config.exp_name = config.alg + '_ener' + str(config.energy_alpha) + '_2llocalcov_5_10channel'

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
    #  transforms.Resize(16),
     transforms.Normalize((0.5), (0.5))])

batch_size = 256

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
    _, c, h, w = data.shape
    break

# %%
###############################################################
# DEFINE NETWORK
###############################################################

# set input and t param

IN_dim = [c, h, w]
config.hidden_channels = [5, 10]
config.kernel_size = [3, 3]
config.stride = [1, 1]
config.paddings = [0, 0]
config.is_rec = [False, False]
config.pooling = None
config.num_readout = 10
config.conv_adp = False
config.spiking_conv = True
spiking_conv = config.spiking_conv

# define network
model = SnnLocalConvNet(IN_dim, config.hidden_channels, config.kernel_size, config.stride,
                   config.paddings, n_classes, is_adapt_conv=config.conv_adp,
                   dp_rate=config.dp, p_size=config.num_readout,
                   pooling=config.pooling, is_rec=config.is_rec)
model.to(device)
print(model)

config.p_size = model.input_to_pc_sz  # log size of p neurons  
print('neuron count %i' % model.neuron_count)

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
test_loss, acc1 = test_conv(model, test_loader, T)

# %%

named_params = get_stats_named_params(model)
all_test_losses = []
best_acc1 = 20

wandb.watch(model, log_freq=20)

for epoch in range(epochs):
    train_fptt_conv(epoch, batch_size, log_interval, train_loader,
                    model, named_params, T, K, omega, optimizer,
                    config.clf_alpha, config.energy_alpha, clip, config.lr)

    reset_named_params(named_params)

    test_loss, acc1 = test_conv(model, test_loader, T)

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
