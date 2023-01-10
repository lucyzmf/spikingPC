import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import sys
import os
import shutil

from torch.nn.parameter import Parameter
from torch.nn import init
from torch.autograd import Variable
import math
import numpy as np

from network import *
from utils import *

from tqdm import tqdm

# %%
num_readout = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
class OneLayerSnn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, is_rec=True, is_LTC=False, is_adapt=True,
                 one_to_one=False):
        super(OneLayerSnn, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.isAdaptNew = is_adapt
        self.is_rec = is_rec
        self.is_LTC = is_LTC
        self.onetoone = one_to_one

        if not self.onetoone:
            self.input_w = nn.Linear(input_size, hidden_size-50)
            # self.input_w = nn.Linear(input_size, hidden_size)
            nn.init.xavier_uniform_(self.input_w.weight)
            self.weight_mask = torch.ones(hidden_size, input_size).to(device)
            self.weight_mask[:50, :] = 0
            # nn.init.normal_(self.input_w.weight, mean=1, std=0.5)

        self.rnn_name = 'SNN: is_LTC-' + str(is_LTC)

        # one recurrent layer 
        self.snn_layer = SNN_rec_cell(input_size, hidden_size, is_rec, is_LTC, is_adapt, one_to_one)

        self.output_layer = nn.Linear(hidden_size, output_size, bias=True)
        self.output_layer_tauM = nn.Linear(output_size * 2, output_size)
        self.tau_m_o = nn.Parameter(torch.Tensor(output_size))

        nn.init.constant_(self.tau_m_o, 20.)
        # nn.init.constant_(self.tau_m_o, 0.)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer_tauM.weight)
        self.act_o = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.dp1 = nn.Dropout(0.1)  # .1
        self.dp2 = nn.Dropout(0.1)
        self.dp3 = nn.Dropout(0.1)
        self.fr = 0

    def forward(self, inputs, h):

        # outputs = []
        hiddens = []

        b, in_dim = inputs.shape  # b is batch
        # this is just one forward pass
        t = 1

        x_down = inputs.reshape(b, self.input_size).float()

        if self.onetoone:
            x_down = x_down * 0.3
            x_down = torch.cat((torch.zeros(b, 10 * num_readout).to(device), x_down), dim=1)
        else:
            self.input_w.weight.data = self.input_w.weight.data * self.weight_mask
            x_down = self.input_w(x_down)
            # x_down = torch.cat((torch.full((b, 10*num_readout), -1).to(device), x_down), dim=1)

        mem_1, spk_1, b_1 = self.snn_layer(x_down, mem_t=h[0], spk_t=h[1], b_t=h[2])

        dense3_x = self.output_layer(spk_1)
        # tauM2 = self.act3(self.layer3_tauM(torch.cat((dense3_x, h[-2]),dim=-1)))
        tauM2 = torch.exp(-1. / (self.tau_m_o))
        mem_out = output_Neuron(dense3_x, mem=h[-2], tau_m=tauM2)

        self.fr = self.fr + spk_1.detach().cpu().numpy().mean()

        h = (mem_1, spk_1, b_1,
             mem_out)

        f_output = F.log_softmax(mem_out, dim=1)
        hiddens.append(h)

        final_state = h
        return f_output, final_state, hiddens


class OneLayerSeqModelPop(nn.Module):
    def __init__(self, ninp, nhid, nout, is_rec=True, is_LTC=True, is_adapt=True, one_to_one=False):
        super(OneLayerSeqModelPop, self).__init__()
        self.nout = nout  # Should be the number of classes
        self.nhid = nhid
        self.is_rec = is_rec
        self.is_LTC = is_LTC
        self.isAdaptNeu = is_adapt
        self.one_to_one = one_to_one
        self.dp = nn.Dropout(0.4)

        self.network = OneLayerSnn(input_size=ninp, hidden_size=nhid, output_size=nout, is_rec=is_rec, is_LTC=is_LTC,
                                   is_adapt=is_adapt, one_to_one=one_to_one)

    def forward(self, inputs, hidden, T):  # this function is only used during inference not training

        t = T
        # print(inputs.shape) # L,B,d
        B, _ = inputs.size()
        probs_outputs = []  # for pred computation
        log_softmax_outputs = []  # for loss computation
        hiddens_all = []
        spike_sum = torch.zeros(B, 10).to(device)

        inputs = self.dp(inputs)

        for i in range(t):
            f_output, hidden, hiddens = self.network.forward(inputs, hidden)

            # read out from 10 populations
            output_spikes = hidden[1][:, :10 * num_readout].view(-1, 10,
                                                                 num_readout)  # take the first 10*28 neurons for read out
            output_spikes_sum = output_spikes.sum(dim=2)  # mean firing of neurons for each class
            spike_sum += output_spikes_sum

            prob_out = F.softmax(output_spikes_sum, dim=1)
            output = F.log_softmax(output_spikes_sum, dim=1)

            probs_outputs.append(prob_out)
            log_softmax_outputs.append(output)
            hiddens_all.append(hiddens)

        prob_out_sum = F.softmax(spike_sum, dim=1)

        return prob_out_sum, log_softmax_outputs, hiddens_all

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (weight.new(bsz, self.nhid).uniform_(),
                weight.new(bsz, self.nhid).zero_(),
                weight.new(bsz, self.nhid).fill_(b_j0),
                # layer out
                weight.new(bsz, self.nout).zero_(),
                # sum spike
                weight.new(bsz, self.nout).zero_(),
                )


# %%

class FeatureExtractor(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim) -> None:
        super(FeatureExtractor, self).__init__()

        self.linear_layer = nn.Linear(in_dim, hidden_dim, out_dim)
        self.out = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, input):
        h = self.linear_layer(input)
        h = self.relu(h)
        out = self.out(h)

        return out
# %%
