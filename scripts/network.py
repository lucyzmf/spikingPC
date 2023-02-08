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

    mem = mem*alpha + (1-alpha) * R_m * inputs - B * spike * dt
    inputs_ = mem - B

    # spike = act_fun_adp(inputs_)  # act_fun : approximation firing function
    spike = F.relu(inputs_)
    # mem = (1 - spike) * mem

    return mem, spike, B, b


def output_Neuron(inputs, mem, tau_m, dt=1):
    """
    The read out neuron is leaky integrator without spike
    """
    d_mem = -mem + inputs
    mem = mem + d_mem * tau_m
    return mem

