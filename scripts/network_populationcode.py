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

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OneLayerSnnWithOutput(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, readout_size, use_spikes, is_rec=True, is_LTC=False,
                 is_adapt=True,
                 one_to_one=False):
        super(OneLayerSnnWithOutput, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.readout_size = readout_size  # num of neurons for readout per class
        self.isAdaptNew = is_adapt
        self.is_rec = is_rec
        self.is_LTC = is_LTC
        self.use_spikes = use_spikes

        self.rnn_name = 'SNN: is_LTC-' + str(is_LTC)

        # one recurrent layer
        self.snn_layer = SNN_rec_cell(hidden_size, hidden_size, is_rec, is_LTC, is_adapt, one_to_one)

        # TODO change output read out here, instead of fc linear layer, need 10 separate weights
        # self.output_heads = nn.ModuleList([nn.Linear(readout_size, 1) for i in range(readout_size)])
        self.output_head = torch.full((readout_size, 1), 0.5, device=device)

        # two tau_m declarations for different computations
        self.output_layer_tauM = nn.Linear(output_size * 2, output_size)
        self.tau_m_o = nn.Parameter(torch.Tensor(output_size), requires_grad=False)

        # init parameters
        nn.init.constant_(self.tau_m_o, 20.)
        # for i in range(output_size):
        #     nn.init.xavier_uniform_(self.output_heads[i].weight)
        # nn.init.constant_(self.tau_m_o, 0.)
        nn.init.zeros_(self.output_layer_tauM.weight)
        self.act_o = nn.Sigmoid()
        self.relu = nn.ELU()

        # def drop out
        self.dp1 = nn.Dropout(0.1)  # .1
        self.dp2 = nn.Dropout(0.1)
        self.dp3 = nn.Dropout(0.1)
        self.fr = 0

    def forward(self, inputs, h):
        '''
        one forward call of one layer rec, used during training
        :param inputs:
        :param h: hidden from last time step
        :return:
            log softmax output: output mem potentials put through log softmax
            hidden_state: hidden state in network
        '''
        # outputs = []

        b, in_dim = inputs.shape  # b is batch

        x_down = inputs.reshape(b, self.input_size).float()

        mem_1, spk_1, b_1 = self.snn_layer(x_down, mem_t=h[0], spk_t=h[1], b_t=h[2])

        self.fr = self.fr + spk_1.detach().cpu().numpy().mean()

        if self.use_spikes:
            #  read out for population code
            output_spikes = h[1][:, :self.readout_size * 10].view(-1, 10,
                                                                  self.readout_size)  # take the first 40 neurons for read out
            output_spikes_sum = output_spikes.sum(dim=2)  # sum firing of neurons for each class
            log_softmax_out = F.log_softmax(output_spikes_sum, dim=1)
            mem_out = torch.zeros(self.output_size)
        else:
            # input to output layer
            dense_x = torch.zeros(b, self.output_size).to(device)
            # compute for each readout head the inputs to output neurons
            for i in range(self.output_size):
                # dense_x[:, i] = self.output_heads[i](spk_1[:, i * self.readout_size: (i + 1) * self.readout_size]).squeeze()
                dense_x[:, i] = (
                            spk_1[:, i * self.readout_size: (i + 1) * self.readout_size] @ self.output_head).squeeze()

            # TODO clean up tau_m computation
            # tauM2 = self.act3(self.output_layer_tauM(torch.cat((dense3_x, h[-2]),dim=-1)))
            tau_m_outs = torch.exp(-1. / self.tau_m_o)

            # update output neuron mem potential
            mem_out = output_Neuron(dense_x, mem=h[-1], tau_m=tau_m_outs)

            log_softmax_out = F.log_softmax(mem_out, dim=1)

        hidden_state = (mem_1, spk_1, b_1,
                        mem_out)

        return log_softmax_out, hidden_state


class OneLayerSeqModelPop(nn.Module):
    def __init__(self, n_inp, n_hid, n_out, n_readout, use_spikes, is_rec=True, is_LTC=True, is_adapt=True, one_to_one=False):
        super(OneLayerSeqModelPop, self).__init__()
        self.n_inp = n_inp
        self.n_out = n_out  # Should be the number of classes
        self.n_hid = n_hid
        self.n_read = n_readout
        self.is_rec = is_rec
        self.is_LTC = is_LTC
        self.isAdaptNeu = is_adapt
        self.onToOne = one_to_one

        self.network = OneLayerSnnWithOutput(input_size=n_inp, hidden_size=n_hid, output_size=n_out,
                                             readout_size=n_readout, use_spikes=use_spikes, is_rec=is_rec,
                                             is_LTC=is_LTC, is_adapt=is_adapt, one_to_one=one_to_one)

    def forward(self, inputs, hidden, T):  # this function is only used during inference not training

        t = T
        # print(inputs.shape) # L,B,d
        B, _ = inputs.size()
        probs_outputs = []  # for pred computation
        log_softmax_outputs = []  # for loss computation, computed at each time step
        hiddens_all = []  #

        for i in range(t):
            log_softmax_out, hidden = self.network.forward(inputs, hidden)

            log_softmax_outputs.append(log_softmax_out)
            hiddens_all.append(hidden)

        return log_softmax_outputs, hiddens_all

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (weight.new(bsz, self.n_hid).uniform_(),
                weight.new(bsz, self.n_hid).zero_(),
                weight.new(bsz, self.n_hid).fill_(b_j0),
                # layer out
                weight.new(bsz, self.n_out).zero_(),
                # sum spike
                weight.new(bsz, self.n_out).zero_(),
                )
