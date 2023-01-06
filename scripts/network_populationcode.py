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

num_readout = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OneLayerSeqModelPop(nn.Module):
    def __init__(self, ninp, nhid, nout, is_rec=True, is_LTC=True, isAdaptNeu=True, oneToOne=False):
        super(OneLayerSeqModelPop, self).__init__()
        self.nout = nout  # Should be the number of classes
        self.nhid = nhid
        self.is_rec = is_rec
        self.is_LTC = is_LTC
        self.isAdaptNeu = isAdaptNeu
        self.onToOne = oneToOne

        self.network = OneLayerSnn(input_size=ninp, hidden_size=nhid, output_size=nout, is_rec=is_rec, is_LTC=is_LTC,
                                   is_adapt=isAdaptNeu, one_to_one=oneToOne)

    def forward(self, inputs, hidden, T):  # this function is only used during inference not training

        t = T
        # print(inputs.shape) # L,B,d
        B, _ = inputs.size()
        probs_outputs = []  # for pred computation
        log_softmax_outputs = []  # for loss computation
        hiddens_all = []
        spike_sum = torch.zeros(B, 10).to(device)

        for i in range(t):
            f_output, hidden, hiddens = self.network.forward(inputs, hidden)

            # read out from 10 populations, take the first 10*28 neurons for read out
            output_spikes = hidden[1][:, :10 * num_readout].view(-1, 10, num_readout)
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
