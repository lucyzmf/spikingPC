import math

import torch
import torch.nn as nn
from network import *


class SnnLayer(nn.Module):
    def __init__(
            self,
            in_dim: int,
            hidden_dim: list,
            is_rec: bool,
            is_adapt: bool,
            one_to_one: bool,
    ):
        super(SnnLayer, self).__init__()
        '''
            hidden_dim contains 
                if rec: (r_in, r_out) where each indicates how many neurons are receiving output or input 
                else: (hidden_dim)
        '''

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.is_rec = is_rec
        self.is_adapt = is_adapt
        self.one_to_one = one_to_one

        # check if dim is correct for one to one set up
        if one_to_one and in_dim != self.r_in:
            raise Exception('input dim and r_in size does not match for one to one set up')

        if is_rec:
            self.r_out = hidden_dim[1]
            self.r_in = hidden_dim[0]
            self.x2in = nn.Linear(in_dim, self.r_in)
            self.rec4in = nn.Linear(self.r_in, self.r_in)
            self.in2out = nn.Linear(self.r_in, self.r_out)
            self.rec4out = nn.Linear(self.r_out, self.r_out)
            self.out2in = nn.Linear(self.r_out, self.r_in)

            # init weights
            nn.init.xavier_uniform_(self.x2in.weight)
            nn.init.xavier_uniform_(self.rec4in.weight)
            nn.init.xavier_uniform_(self.in2out.weight)
            nn.init.xavier_uniform_(self.rec4out.weight)
            nn.init.xavier_uniform_(self.out2in.weight)
        else:
            self.fc_weights = nn.Linear(in_dim, hidden_dim)

        # define param for time constants
        self.tau_adp = nn.Parameter(torch.Tensor(sum(hidden_dim)))
        self.tau_m = nn.Parameter(torch.Tensor(sum(hidden_dim)))

        nn.init.normal_(self.tau_adp, 4.6, .1)
        nn.init.normal_(self.tau_m, 3., .1)

        self.sigmoid = nn.Sigmoid()

    def mem_update(self, inputs, mem, spike, b, is_adapt, dt=1, baseline_thre=0.1, r_m=3):
        tau_m = self.sigmoid(self.tau_m)
        tau_adp = self.sigmoid(self.tau_adp)

        if is_adapt:
            beta = 1.8
        else:
            beta = 0.

        b = tau_adp * b + (1 - tau_adp) * spike  # adaptive contribution
        new_thre = baseline_thre + beta * b  # udpated threshold

        mem = mem * tau_m + (1 - tau_m) * r_m * inputs - new_thre * spike * dt
        inputs_ = mem - new_thre

        spike = act_fun_adp(inputs_)  # act_fun : approximation firing function
        # mem = (1 - spike) * mem

        return mem, spike, new_thre, b

    def forward(self, x_t, mem_t, spk_t, b_t):
        """
        forward function of a single layer. given previous neuron states and current input, update neuron states
        :param x_t: input at time t
        :param mem_t: mem potentials at t
        :param spk_t: spikes at t
        :param b_t: adaptive threshold contribution at t
        :return: new neuron states
        """
        if self.is_rec:
            if self.one_to_one:
                x = x_t
            else:
                x = self.x2in(x_t)
            r_in_ = x + self.rec4in(spk_t[:, self.r_out:]) + self.out2in(spk_t[:, :self.r_out])
            r_out_ = self.rec4out(spk_t[:, :self.r_out]) + self.in2out(spk_t[:, self.r_out:])

            input2hidden = torch.cat((r_out_, r_in_), dim=1)
        else:
            input2hidden = self.fc_weights(x_t)

        mem_t1, spk_t1, _, b_t1 = self.mem_update(input2hidden, mem_t, spk_t, b_t, self.is_adapt)

        return mem_t1, spk_t1, b_t1


class OutputLayer(nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            is_fc: bool,
    ):
        """
        output layer class
        :param is_fc: whether integrator is fc to r_out in rec or not
        """
        super(OutputLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.is_fc = is_fc

        if is_fc:
            self.fc = nn.Linear(in_dim, out_dim)
            nn.init.xavier_uniform_(self.fc.weight)

        # tau_m
        self.tau_m = nn.Parameter(torch.Tensor(out_dim))
        nn.init.constant_(self.tau_m, 20.)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x_t, mem_t, dt=1):
        """
        integrator neuron without spikes
        """
        tau_m = self.sigmoid(self.tau_m)

        if self.is_fc:
            x_t = self.fc(x_t)
        else:
            x_t = x_t.view(-1, 10, self.in_dim / 10).sum(dim=2)  # sum up population spike

        d_mem = -mem_t + x_t
        mem = mem_t + d_mem * tau_m
        return mem


class SnnNetwork(nn.Module):
    def __init__(
            self,
            in_dim: int,
            hidden_dims: list,
            out_dim: int,
            is_adapt: bool,
            one_to_one: bool,
            dp_rate=0.3
    ):
        super(SnnNetwork, self).__init__()

        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.is_adapt = is_adapt
        self.one_to_one = one_to_one

        self.dp = nn.Dropout(dp_rate)

        self.fc_layer = SnnLayer(in_dim, hidden_dims[0], is_rec=False, is_adapt=is_adapt, one_to_one=False)
        self.rec_layer = SnnLayer(hidden_dims[0], hidden_dims[1], is_rec=True, is_adapt=is_adapt, one_to_one=one_to_one)

        self.output_layer = OutputLayer(hidden_dims[1][0], out_dim, is_fc=False)

        self.fr_p = 0
        self.fr_r = 0

    def forward(self, x_t, h):
        batch_dim, input_size = x_t.shape

        x_t = x_t.reshape(batch_dim, self.in_dim).float()
        x_t = self.dp(x_t)

        mem1, spk1, b1 = self.fc_layer(x_t, mem_t=h[0], spk_t=h[1], b_t=h[2])
        mem2, spk2, b2 = self.rec_layer(spk1, mem_t=h[3], spk_t=h[4], b_t=h[5])

        # read out from r_out neurons
        mem_out = self.output_layer(spk2[:, :self.rec_layer.r_out])

        h = (mem1, spk1, b1,
             mem2, spk2, b2,
             mem_out)

        log_softmax = F.log_softmax(mem_out)

        return log_softmax, h

    def inference(self, x_t, h, T):
        """
        only called during inference
        :param x_t: input
        :param h: hidden states
        :param T: sequence length
        :return:
        """

        log_softmax_hist = []
        h_hist = []

        for t in range(T):
            log_softmax, hidden = self.forward(x_t, h)

            log_softmax_hist.append(log_softmax)
            h_hist.append(hidden)

        return log_softmax_hist, h_hist

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (
            # input layer
            weight.new(bsz, self.hidden_dims[0]).uniform_(),  # mem
            weight.new(bsz, self.hidden_dims[0]).zero_(),  # spk
            weight.new(bsz, self.hidden_dims[0]).fill_(b_j0),  # thre
            # rec
            weight.new(bsz, sum(self.hidden_dims[1])).uniform_(),
            weight.new(bsz, sum(self.hidden_dims[1])).zero_(),
            weight.new(bsz, sum(self.hidden_dims[1])).fill_(b_j0),
            # layer out
            weight.new(bsz, self.out_dim).zero_(),
            # sum spike
            weight.new(bsz, self.out_dim).zero_(),
        )
