# %%
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

import matplotlib.pyplot as plt
import IPython.display as ipd

from tqdm import tqdm

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
###############################################################
# DEFINE NETWORK
###############################################################

"""
Liquid time constant snn
"""

# %%
###############################################################################################
###############################    Define SNN layer   #########################################
###############################################################################################

b_j0 = .1  # neural threshold baseline
R_m = 3  # membrane resistance
dt = 1
gamma = .5  # gradient scale
lens = 0.5


def gaussian(x, mu=0., sigma=.5):
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(math.pi)) / sigma


class ActFun_adp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):  # input = membrane potential- threshold
        ctx.save_for_backward(input)
        return input.gt(0).float()  # is firing ???

    @staticmethod
    def backward(ctx, grad_output):  # approximate the gradients
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # temp = abs(input) < lens
        scale = 6.0
        hight = .15
        # temp = torch.exp(-(input**2)/(2*lens**2))/torch.sqrt(2*torch.tensor(math.pi))/lens
        temp = gaussian(input, mu=0., sigma=lens) * (1. + hight) \
               - gaussian(input, mu=lens, sigma=scale * lens) * hight \
               - gaussian(input, mu=-lens, sigma=scale * lens) * hight
        # temp =  gaussian(input, mu=0., sigma=lens)
        return grad_input * temp.float() * gamma
        # return grad_input


act_fun_adp = ActFun_adp.apply


def mem_update_adp(inputs, mem, spike, tau_adp, tau_m, b, isAdapt=1, dt=1):
    alpha = tau_m

    ro = tau_adp

    if isAdapt:
        beta = 1.8
    else:
        beta = 0.

    b = ro * b + (1 - ro) * spike
    B = b_j0 + beta * b

    d_mem = -mem + inputs
    mem = mem + d_mem * alpha
    inputs_ = mem - B

    spike = act_fun_adp(inputs_)  # act_fun : approximation firing function
    mem = (1 - spike) * mem

    return mem, spike, B, b


def output_Neuron(inputs, mem, tau_m, dt=1):
    """
    The read out neuron is leaky integrator without spike
    """
    d_mem = -mem + inputs
    mem = mem + d_mem * tau_m
    return mem


# %%
###############################################################################################
###############################################################################################
###############################################################################################
class SNN_rec_cell(nn.Module):
    def __init__(self, input_size, hidden_size, readout_size_per_class, is_rec=False, is_LTC=True, isAdaptNeu=True,
                 oneToOne=False):
        super(SNN_rec_cell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.readout_size_per_class = readout_size_per_class
        self.is_rec = is_rec
        self.is_LTC = is_LTC
        self.isAdaptNeu = isAdaptNeu
        self.oneToOne = oneToOne  # whether one to one input or fully connected input

        if is_rec:
            # self.layer1_x = nn.Linear(hidden_size, hidden_size)
            self.r_size = hidden_size - readout_size_per_class * 10
            if not self.oneToOne:
                self.i2r = nn.Linear(input_size, self.r_size)
                nn.init.xavier_uniform_(self.i2r.weight)
            else:
                self.i2r = torch.full((self.r_size,), 0.5).to(device)
            self.r2r = nn.Linear(self.r_size, self.r_size)
            self.p2r = nn.Linear(readout_size_per_class * 10, self.r_size)
            self.p2p = nn.Linear(readout_size_per_class * 10, readout_size_per_class * 10)
            self.r2p = nn.Linear(self.r_size, readout_size_per_class * 10)

            nn.init.xavier_uniform_(self.r2r.weight)
            nn.init.xavier_uniform_(self.p2r.weight)
            nn.init.xavier_uniform_(self.p2p.weight)
            nn.init.xavier_uniform_(self.r2r.weight)

        else:
            self.layer1_x = nn.Linear(input_size, hidden_size)
            nn.init.xavier_uniform_(self.layer1_x.weight)

        # time-constant definiation and initilization
        if is_LTC:
            self.layer1_tauAdp = nn.Linear(2 * hidden_size, hidden_size)
            self.layer1_tauM = nn.Linear(2 * hidden_size, hidden_size)
            nn.init.xavier_uniform_(self.layer1_tauAdp.weight)
            nn.init.xavier_uniform_(self.layer1_tauM.weight)
        else:
            self.tau_adp = nn.Parameter(torch.Tensor(hidden_size))
            self.tau_m = nn.Parameter(torch.Tensor(hidden_size))
            nn.init.normal_(self.tau_adp, 4.6, .1)
            nn.init.normal_(self.tau_m, 3., .1)
        self.act1 = nn.Sigmoid()
        self.act2 = nn.Sigmoid()


    def forward(self, x_t, mem_t, spk_t, b_t):
        if self.is_rec:
            # if not self.one_to_one:
            #     dense_x = self.layer1_x(torch.cat((x_t,spk_t),dim=-1))
            # else:
            # compute input drive, 1 to 1 input
            r_input = self.r2r(spk_t[:, self.readout_size_per_class * 10:]) + \
                      self.p2r(spk_t[:, :self.readout_size_per_class * 10]) + self.i2r(x_t)
            p_input = self.p2p(spk_t[:, :self.readout_size_per_class * 10]) + \
                      self.r2p(spk_t[:, self.readout_size_per_class * 10:])
            dense_x = x_t + torch.cat((p_input, r_input))

        else:
            dense_x = self.layer1_x(x_t)

        if self.is_LTC:
            tauM1 = self.act1(self.layer1_tauM(torch.cat((dense_x, mem_t), dim=-1)))
            tauAdp1 = self.act1(self.layer1_tauAdp(torch.cat((dense_x, b_t), dim=-1)))
        else:
            tauM1 = self.act1(self.tau_m)
            tauAdp1 = self.act2(self.tau_adp)

        mem_1, spk_1, _, b_1 = mem_update_adp(dense_x, mem=mem_t, spike=spk_t,
                                              tau_adp=tauAdp1, tau_m=tauM1, b=b_t, isAdapt=self.isAdaptNeu)

        return mem_1, spk_1, b_1

    def compute_output_size(self):
        return [self.hidden_size]


class one_layer_SeqModel(nn.Module):
    def __init__(self, ninp, nhid, nout, is_rec=True, is_LTC=True, isAdaptNeu=True):
        super(one_layer_SeqModel, self).__init__()
        self.nout = nout  # Should be the number of classes
        self.nhid = nhid
        self.is_rec = is_rec
        self.is_LTC = is_LTC
        self.isAdaptNeu = isAdaptNeu

        self.network = one_layer_SNN(input_size=ninp, hidden_size=nhid, output_size=nout, is_rec=is_rec, is_LTC=is_LTC,
                                     isAdaptNeu=isAdaptNeu)

    def forward(self, inputs, hidden, T):  # this function is only used during inference not training

        t = T
        # print(inputs.shape) # L,B,d
        probs_outputs = []  # for pred computation 
        log_softmax_outputs = []  # for loss computation
        hiddens_all = []
        for i in range(t):
            f_output, hidden, hiddens = self.network.forward(inputs, hidden)

            # read out fron first 10 neuron spikes 
            prob_out = F.softmax(hidden[1][:, :10], dim=1)  # take the first 10 neurons for read out
            output = F.log_softmax(hidden[1][:, :10], dim=1)

            probs_outputs.append(prob_out)
            log_softmax_outputs.append(output)
            hiddens_all.append(hiddens)
        return probs_outputs, log_softmax_outputs, hiddens_all

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
