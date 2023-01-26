# %%
import math

import torch
import torch.nn as nn
from network import *


class SnnLayer(nn.Module):
    def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            is_rec: bool,
            is_adapt: bool,
            one_to_one: bool,
    ):
        super(SnnLayer, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.is_rec = is_rec
        self.is_adapt = is_adapt
        self.one_to_one = one_to_one

        if is_rec:
            self.rec_w = nn.Linear(hidden_dim, hidden_dim)
            # init weights
            nn.init.xavier_uniform_(self.rec_w.weight)
            self.input_fc = nn.Linear(in_dim, hidden_dim)
            nn.init.xavier_uniform_(self.input_fc.weight)

        else:
            self.fc_weights = nn.Linear(in_dim, hidden_dim)
            nn.init.xavier_uniform_(self.fc_weights.weight)

        # define param for time constants
        self.tau_adp = nn.Parameter(torch.Tensor(hidden_dim))
        self.tau_m = nn.Parameter(torch.Tensor(hidden_dim))

        nn.init.normal_(self.tau_adp, 4.6, .1)
        nn.init.normal_(self.tau_m, 3., .1)

        self.sigmoid = nn.Sigmoid()

    def mem_update(self, inputs, mem, spike, b, is_adapt, dt=1, baseline_thre=0.1, r_m=3):
        alpha = self.sigmoid(self.tau_m)
        rho = self.sigmoid(self.tau_adp)

        if is_adapt:
            beta = 1.8
        else:
            beta = 0.

        b = rho * b + (1 - rho) * spike  # adaptive contribution
        new_thre = baseline_thre + beta * b  # udpated threshold

        mem = mem * alpha + (1 - alpha) * r_m * inputs - new_thre * spike 
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
            r_in = self.input_fc(x_t) + self.rec_w(spk_t)
        else:
            r_in = self.fc_weights(x_t)

        mem_t1, spk_t1, _, b_t1 = self.mem_update(r_in, mem_t, spk_t, b_t, self.is_adapt)

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
        nn.init.constant_(self.tau_m, 0.5)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x_t, mem_t, dt=1):
        """
        integrator neuron without spikes
        """
        # alpha = torch.exp(-1/self.tau_m)

        if self.is_fc:
            x_t = self.fc(x_t)
        else:
            x_t = x_t.view(-1, 10, int(self.in_dim / 10)).sum(dim=2)  # sum up population spike

        # d_mem = -mem_t + x_t
        mem = mem_t + x_t * self.tau_m
        return mem


class TwoLayerSnnNetwork(nn.Module):
    def __init__(
            self,
            in_dim: int,
            hidden_dims: list,  # [h1, [r out, r in], [r_out, r in]]
            out_dim: int,
            is_adapt: bool,
            one_to_one: bool,
            dp_rate=0.3
    ):
        super(TwoLayerSnnNetwork, self).__init__()

        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.is_adapt = is_adapt
        self.one_to_one = one_to_one

        self.dp = nn.Dropout(dp_rate)

        self.fc_layer = SnnLayer(in_dim, hidden_dims[0], is_rec=False, is_adapt=is_adapt, one_to_one=False)

        # between layer weights
        self.fc2r_in = nn.Linear(hidden_dims[0], hidden_dims[1][1])
        nn.init.xavier_uniform_(self.fc2r_in.weight)

        self.rec12rec2 = nn.Linear(hidden_dims[1][0], hidden_dims[2][1])
        nn.init.xavier_uniform_(self.rec12rec2.weight)
        self.rec22rec1 = nn.Linear(hidden_dims[2][1], hidden_dims[1][0])
        nn.init.xavier_uniform_(self.rec22rec1.weight)

        # r in rec
        self.r_in_rec1 = SnnLayer((hidden_dims[0] + hidden_dims[1][0]), hidden_dims[1][1], is_rec=True, is_adapt=is_adapt, one_to_one=one_to_one)
        self.r_in_rec2 = SnnLayer((hidden_dims[1][0] + hidden_dims[2][0]), hidden_dims[2][1], is_rec=True, is_adapt=is_adapt, one_to_one=one_to_one)

        # r in to r out
        self.rin2rout1 = nn.Linear(hidden_dims[1][1], hidden_dims[1][0])
        nn.init.xavier_uniform_(self.rin2rout1.weight)

        self.rin2rout2 = nn.Linear(hidden_dims[2][1], hidden_dims[2][0])
        nn.init.xavier_uniform_(self.rin2rout2.weight)

        # r out rec
        self.r_out_rec1 = SnnLayer(hidden_dims[1][0], hidden_dims[1][0], is_rec=True, is_adapt=is_adapt, one_to_one=one_to_one)
        self.r_out_rec2 = SnnLayer(hidden_dims[2][0], hidden_dims[2][0], is_rec=True, is_adapt=is_adapt, one_to_one=one_to_one)

        self.output_layer = OutputLayer(hidden_dims[2][0], out_dim, is_fc=False)

        self.fr_p1 = 0
        self.fr_r1 = 0
        self.fr_p2 = 0
        self.fr_r2 = 0

    def forward(self, x_t, h):
        batch_dim, input_size = x_t.shape

        x_t = x_t.reshape(batch_dim, input_size).float()
        x_t = self.dp(x_t)

        # fc layer
        mem1, spk1, b1 = self.fc_layer(x_t, mem_t=h[0], spk_t=h[1], b_t=h[2])

        # first rec layer
        r_input = torch.cat((spk1, h[7]), dim=1)
        mem_r1, spk_r1, b_r1 = self.r_in_rec1(r_input, mem_t=h[3], spk_t=h[4], b_t=h[5])

        p_input = spk_r1
        mem_p1, spk_p1, b_p1 = self.r_out_rec1(p_input, mem_t=h[6], spk_t=h[7], b_t=h[8])

        # second rec layer
        r_input2 = torch.cat((spk_p1, h[13]), dim=1)
        mem_r2, spk_r2, b_r2 = self.r_in_rec2(r_input2, mem_t=h[9], spk_t=h[10], b_t=h[11])

        p_input2 = spk_r2
        mem_p2, spk_p2, b_p2 = self.r_out_rec2(p_input2, mem_t=h[12], spk_t=h[13], b_t=h[14])

        self.fr_p1 = self.fr_p1 + spk_p1.detach().cpu().numpy().mean()
        self.fr_p2 = self.fr_p2 + spk_p2.detach().cpu().numpy().mean()
        self.fr_r1 = self.fr_r1 + spk_r1.detach().cpu().numpy().mean()
        self.fr_r2 = self.fr_r2 + spk_r2.detach().cpu().numpy().mean()

        # read out from r_out neurons
        mem_out = self.output_layer(spk_p2, h[-1])

        h = (mem1, spk1, b1,
             mem_r1, spk_r1, b_r1,
             mem_p1, spk_p1, b_p1,
             mem_r2, spk_r2, b_r2,
             mem_p2, spk_p2, b_p2,
             mem_out)

        log_softmax = F.log_softmax(mem_out, dim=1)

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
            log_softmax, h = self.forward(x_t, h)

            log_softmax_hist.append(log_softmax)
            h_hist.append(h)

        return log_softmax_hist, h_hist

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (
            # input layer
            weight.new(bsz, self.hidden_dims[0]).uniform_(),  # mem
            weight.new(bsz, self.hidden_dims[0]).zero_(),  # spk
            weight.new(bsz, self.hidden_dims[0]).fill_(b_j0),  # thre
            # r1
            weight.new(bsz, self.hidden_dims[1][1]).uniform_(),
            weight.new(bsz, self.hidden_dims[1][1]).zero_(),
            weight.new(bsz, self.hidden_dims[1][1]).fill_(b_j0),
            # p1
            weight.new(bsz, self.hidden_dims[1][0]).uniform_(),
            weight.new(bsz, self.hidden_dims[1][0]).zero_(),
            weight.new(bsz, self.hidden_dims[1][0]).fill_(b_j0),
            # r2
            weight.new(bsz, self.hidden_dims[2][1]).uniform_(),
            weight.new(bsz, self.hidden_dims[2][1]).zero_(),
            weight.new(bsz, self.hidden_dims[2][1]).fill_(b_j0),
            # p2
            weight.new(bsz, self.hidden_dims[2][0]).uniform_(),
            weight.new(bsz, self.hidden_dims[2][0]).zero_(),
            weight.new(bsz, self.hidden_dims[2][0]).fill_(b_j0),
            # layer out
            weight.new(bsz, self.out_dim).zero_(),
            # sum spike
            weight.new(bsz, self.out_dim).zero_(),
        )

# %%
