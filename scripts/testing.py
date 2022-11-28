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
    return hiddens, test_loss, 100. * correct / len(test_loader.dataset)
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
saved_dict = model_result_dict_load('/home/lucy/spikingPC/results/energy_loss_1_nonadp_memlossNov-28-2022/_onelayer_rec_best.pth.tar')
# %%
model.load_state_dict(saved_dict['state_dict'])
# %%
#####################################################################
# visualisation 
#####################################################################
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
hiddens, test_loss, _ = test(model, test_loader)
# %%
# get spiking pattern along the sequence 
spikes_all = []
for i in range(len(hiddens)):
    spikes_all.append(hiddens[i][0][1].detach().cpu().numpy())

spikes_all = np.stack(spikes_all)
# %%
# here each entry to spike_all is 16*784 (batchsize*num neurons) at each time step 
spikes_one_img = spikes_all[:, 0, :].transpose()
spike_pos = []
for i in range(len(spikes_one_img)):
    pos = np.nonzero(spikes_one_img[0])
    spike_pos.append(pos)

lineoffsets1 = np.arange(-784/2, 784/2)
linelengths1 = np.ones(784)

plt.eventplot(spike_pos, lineoffsets=lineoffsets1, linelengths=linelengths1)
plt.show()
# %%
