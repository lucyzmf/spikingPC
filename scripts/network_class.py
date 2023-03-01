# %%
import math

import torch
import torch.nn as nn
from network import *

b_j0 = 0.1  # neural threshold baseline


class SnnLayer(nn.Module):
    def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            is_rec: bool,
            is_adapt: bool,
            one_to_one: bool,
            tau_m_init=3.,
            tau_adap_init=4.6,
            tau_i_init=0.
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

        else:
            self.fc_weights = nn.Linear(in_dim, hidden_dim)
            nn.init.xavier_uniform_(self.fc_weights.weight)

        # define param for time constants
        self.tau_adp = nn.Parameter(torch.Tensor(hidden_dim))
        self.tau_m = nn.Parameter(torch.Tensor(hidden_dim))
        self.tau_i = nn.Parameter(torch.Tensor(hidden_dim))

        nn.init.normal_(self.tau_adp, tau_adap_init, .1)
        nn.init.normal_(self.tau_m, tau_m_init, .1)
        nn.init.normal_(self.tau_i, tau_i_init, 0.1)

        # self.tau_adp = nn.Parameter(torch.Tensor(1))
        # self.tau_m = nn.Parameter(torch.Tensor(1))
        # self.tau_i = nn.Parameter(torch.Tensor(1))

        # nn.init.constant_(self.tau_adp, tau_adap_init)
        # nn.init.constant_(self.tau_m, tau_m_init)
        # nn.init.constant_(self.tau_i, tau_i_init)

        # nn.init.normal_(self.tau_adp, 200., 20.)
        # nn.init.normal_(self.tau_m, 20., .5)

        self.sigmoid = nn.Sigmoid()

    def mem_update(self, inputs, mem, spike, current, b, is_adapt, dt=1, baseline_thre=b_j0, r_m=3):
        alpha = self.sigmoid(self.tau_m)
        rho = self.sigmoid(self.tau_adp)
        eta = 0  # self.sigmoid(self.tau_i)
        # alpha = torch.exp(-dt/self.tau_m)
        # rho = torch.exp(-dt/self.tau_adp)

        if is_adapt:
            beta = 1.8
        else:
            beta = 0.

        b = rho * b + (1 - rho) * spike  # adaptive contribution
        new_thre = baseline_thre + beta * b  # udpated threshold

        current = eta * current + (1 - eta) * R_m * inputs

        # mem = mem * alpha + (1 - alpha) * r_m * inputs - new_thre * spike
        mem = mem * alpha + current - new_thre * spike  # soft reset
        inputs_ = mem - new_thre

        spike = act_fun_adp(inputs_)  # act_fun : approximation firing function
        # mem = (1 - spike) * mem

        return mem, spike, current, new_thre, b

    def forward(self, x_t, mem_t, spk_t, curr_t, b_t):
        """
        forward function of a single layer. given previous neuron states and current input, update neuron states
        :param curr_t: current
        :param x_t: input at time t
        :param mem_t: mem potentials at t
        :param spk_t: spikes at t
        :param b_t: adaptive threshold contribution at t
        :return: new neuron states
        """
        if self.is_rec:
            r_in = x_t + self.rec_w(spk_t)
        else:
            if self.one_to_one:
                r_in = x_t
            else:
                r_in = self.fc_weights(x_t)

        mem_t1, spk_t1, curr_t1, _, b_t1 = self.mem_update(r_in, mem_t, spk_t, curr_t, b_t, self.is_adapt)

        return mem_t1, spk_t1, curr_t1, b_t1


class OutputLayer(nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            is_fc: bool,
            tau_fixed = None
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
        if tau_fixed is None:
            self.tau_m = nn.Parameter(torch.Tensor(out_dim))
        else:
            self.tau_m = nn.Parameter(torch.Tensor(out_dim), requires_grad=False)

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
            x_t = x_t.view(-1, 10, int(self.in_dim / 10)).mean(dim=2)  # sum up population spike

        # d_mem = -mem_t + x_t
        mem = (mem_t + x_t) * self.tau_m
        return mem


class SnnNetwork(nn.Module):
    def __init__(
            self,
            in_dim: int,
            hidden_dims: list,  # [r out, r in]
            out_dim: int,
            is_adapt: bool,
            one_to_one: bool,
            dp_rate: float
    ):
        super(SnnNetwork, self).__init__()

        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.is_adapt = is_adapt
        self.one_to_one = one_to_one

        self.dp = nn.Dropout(dp_rate)

        self.r_in_rec = SnnLayer(hidden_dims[1], hidden_dims[1], is_rec=True, is_adapt=is_adapt,
                                 one_to_one=one_to_one)

        # r in to r out
        self.rin2rout = nn.Linear(hidden_dims[1], hidden_dims[0])
        nn.init.xavier_uniform_(self.rin2rout.weight)

        # r out to r in
        self.rout2rin = nn.Linear(hidden_dims[0], hidden_dims[1])
        nn.init.xavier_uniform_(self.rout2rin.weight)

        self.r_out_rec = SnnLayer(hidden_dims[0], hidden_dims[0], is_rec=True, is_adapt=is_adapt,
                                  one_to_one=one_to_one)

        self.output_layer = OutputLayer(hidden_dims[0], out_dim, is_fc=False)

        self.fr_p = 0
        self.fr_r = 0

    def forward(self, x_t, h):
        batch_dim, input_size = x_t.shape

        x_t = x_t.reshape(batch_dim, input_size).float()
        x_t = self.dp(x_t)
        # poisson 
        # x_t = x_t.gt(0.5).float()

        r_input = x_t + self.rout2rin(h[5])

        mem_r, spk_r, curr_r, b_r = self.r_in_rec(r_input, mem_t=h[0], spk_t=h[1], curr_t=h[2], b_t=h[3])

        p_input = self.rin2rout(spk_r)

        mem_p, spk_p, curr_p, b_p = self.r_out_rec(p_input, mem_t=h[4], spk_t=h[5], curr_t=h[6], b_t=h[7])

        self.fr_p = self.fr_p + spk_p.detach().cpu().numpy().mean()
        self.fr_r = self.fr_r + spk_r.detach().cpu().numpy().mean()

        # read out from r_out neurons
        mem_out = self.output_layer(spk_p, h[-1])

        h = (mem_r, spk_r, curr_r, b_r,
             mem_p, spk_p, curr_p, b_p,
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
            # r
            weight.new(bsz, self.hidden_dims[1]).uniform_(),
            weight.new(bsz, self.hidden_dims[1]).zero_(),
            weight.new(bsz, self.hidden_dims[1]).zero_(),
            weight.new(bsz, self.hidden_dims[1]).fill_(b_j0),
            # p
            weight.new(bsz, self.hidden_dims[0]).uniform_(),
            weight.new(bsz, self.hidden_dims[0]).zero_(),
            weight.new(bsz, self.hidden_dims[0]).zero_(),
            weight.new(bsz, self.hidden_dims[0]).fill_(b_j0),
            # layer out
            weight.new(bsz, self.out_dim).zero_(),
            # sum spike
            weight.new(bsz, self.out_dim).zero_(),
        )


# %%
class SnnNetworkSeq(SnnNetwork):
    def __init__(
            self,
            in_dim: int,
            hidden_dims: list,  # [r out, r in]
            out_dim: int,
            is_adapt: bool,
            one_to_one: bool,
            dp_rate: float
    ):
        super().__init__(in_dim, hidden_dims, out_dim, is_adapt, one_to_one, dp_rate)

    # override inference function
    def inference(self, x_t, h, time_steps):
        """
        only called during inference
        :param x_t: input, contains all data for one sequence
        :param h: hidden states
        :param time_steps: sequence length
        :return:
        """

        log_softmax_hist = []
        h_hist = []
        pred_hist = []  # log predictions at each time step for evaluation

        for t in range(time_steps):
            # iterate through each seq per time step
            log_softmax, h = self.forward(x_t[:, t, :], h)

            log_softmax_hist.append(log_softmax)
            pred = log_softmax.data.max(1, keepdim=True)[1]

            pred_hist.append(pred)
            h_hist.append(h)

        pred_hist = torch.stack(pred_hist).squeeze()

        return log_softmax_hist, h_hist, pred_hist


class SnnNetwork2Layer(SnnNetwork):
    def __init__(
            self,
            in_dim: int,
            hidden_dims: list,  # [p1, r1, p2, r2]
            out_dim: int,
            is_adapt: bool,
            one_to_one: bool,
            dp_rate: float
    ):
        super().__init__(in_dim, hidden_dims, out_dim, is_adapt, one_to_one, dp_rate)

        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.is_adapt = is_adapt
        self.one_to_one = one_to_one

        self.dp = nn.Dropout(dp_rate)

        self.r_in_rec1 = SnnLayer(hidden_dims[1], hidden_dims[1], is_rec=True, is_adapt=is_adapt,
                                  one_to_one=one_to_one)

        self.r_in_rec2 = SnnLayer(hidden_dims[3], hidden_dims[3], is_rec=True, is_adapt=is_adapt,
                                  one_to_one=one_to_one)

        # r in to r out
        self.rin2rout1 = nn.Linear(hidden_dims[1], hidden_dims[0])
        nn.init.xavier_uniform_(self.rin2rout1.weight)

        self.rin2rout2 = nn.Linear(hidden_dims[3], hidden_dims[2])
        nn.init.xavier_uniform_(self.rin2rout2.weight)

        # r out to r in
        self.rout2rin1 = nn.Linear(hidden_dims[0], hidden_dims[1])
        nn.init.xavier_uniform_(self.rout2rin1.weight)

        self.rout2rin2 = nn.Linear(hidden_dims[2], hidden_dims[3])
        nn.init.xavier_uniform_(self.rout2rin2.weight)

        self.r_out_rec1 = SnnLayer(hidden_dims[0], hidden_dims[0], is_rec=True, is_adapt=is_adapt,
                                   one_to_one=one_to_one)

        self.r_out_rec2 = SnnLayer(hidden_dims[2], hidden_dims[2], is_rec=True, is_adapt=is_adapt,
                                   one_to_one=one_to_one)

        self.top_down = nn.Linear(hidden_dims[3], hidden_dims[0])  # r2 to p1
        self.bottom_up = nn.Linear(hidden_dims[0], hidden_dims[3])  # p1 to r2
        nn.init.xavier_uniform_(self.top_down.weight)
        nn.init.xavier_uniform_(self.bottom_up.weight)

        self.skip = nn.Linear(hidden_dims[1], hidden_dims[3])
        nn.init.xavier_uniform_(self.skip.weight)

        self.output_layer = OutputLayer(hidden_dims[2], out_dim, is_fc=False)

    def forward(self, x_t, h):
        batch_dim, input_size = x_t.shape

        x_t = x_t.reshape(batch_dim, input_size).float()
        x_t = self.dp(x_t)
        # poisson
        # x_t = x_t.gt(0.5).float()

        r_input1 = x_t + self.rout2rin1(h[5])

        mem_r1, spk_r1, curr_r1, b_r1 = self.r_in_rec1(r_input1, mem_t=h[0], spk_t=h[1], curr_t=h[2], b_t=h[3])

        p_input1 = self.rin2rout1(spk_r1) + self.top_down(h[9])

        mem_p1, spk_p1, curr_p1, b_p1 = self.r_out_rec1(p_input1, mem_t=h[4], spk_t=h[5], curr_t=h[6], b_t=h[7])

        r_input2 = self.rout2rin2(h[13]) + self.bottom_up(spk_p1)  # + self.rin2rout1(spk_r1)

        mem_r2, spk_r2, curr_r2, b_r2 = self.r_in_rec2(r_input2, mem_t=h[8], spk_t=h[9], curr_t=h[10], b_t=h[11])

        p_input2 = self.rin2rout2(spk_r2)

        mem_p2, spk_p2, curr_p2, b_p2 = self.r_out_rec2(p_input2, mem_t=h[12], spk_t=h[13], curr_t=h[14], b_t=h[15])

        self.fr_p = self.fr_p + spk_p1.detach().cpu().numpy().mean() / 2 + spk_p2.detach().cpu().numpy().mean() / 2
        self.fr_r = self.fr_r + spk_r1.detach().cpu().numpy().mean() / 2 + spk_r2.detach().cpu().numpy().mean() / 2

        # read out from r_out neurons
        mem_out = self.output_layer(spk_p2, h[-1])

        h = (mem_r1, spk_r1, curr_r1, b_r1,
             mem_p1, spk_p1, curr_p1, b_p1,
             mem_r2, spk_r2, curr_r2, b_r2,
             mem_p2, spk_p2, curr_p2, b_p2,
             mem_out)

        log_softmax = F.log_softmax(mem_out, dim=1)

        return log_softmax, h

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (
            # r
            weight.new(bsz, self.hidden_dims[1]).uniform_(),
            weight.new(bsz, self.hidden_dims[1]).zero_(),
            weight.new(bsz, self.hidden_dims[1]).zero_(),
            weight.new(bsz, self.hidden_dims[1]).fill_(b_j0),
            # p
            weight.new(bsz, self.hidden_dims[0]).uniform_(),
            weight.new(bsz, self.hidden_dims[0]).zero_(),
            weight.new(bsz, self.hidden_dims[0]).zero_(),
            weight.new(bsz, self.hidden_dims[0]).fill_(b_j0),
            # r
            weight.new(bsz, self.hidden_dims[3]).uniform_(),
            weight.new(bsz, self.hidden_dims[3]).zero_(),
            weight.new(bsz, self.hidden_dims[3]).zero_(),
            weight.new(bsz, self.hidden_dims[3]).fill_(b_j0),
            # p
            weight.new(bsz, self.hidden_dims[2]).uniform_(),
            weight.new(bsz, self.hidden_dims[2]).zero_(),
            weight.new(bsz, self.hidden_dims[2]).zero_(),
            weight.new(bsz, self.hidden_dims[2]).fill_(b_j0),
            # layer out
            weight.new(bsz, self.out_dim).zero_(),
            # sum spike
            weight.new(bsz, self.out_dim).zero_(),
        )


class SnnNetwork3MidLayer(SnnNetwork):
    def __init__(
            self,
            in_dim: int,
            hidden_dims: list,  # [p1, r1, p2, r2, r3]
            out_dim: int,
            is_adapt: bool,
            one_to_one: bool,
            dp_rate: float
    ):
        super().__init__(in_dim, hidden_dims, out_dim, is_adapt, one_to_one, dp_rate)

        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.is_adapt = is_adapt
        self.one_to_one = one_to_one

        self.dp = nn.Dropout(dp_rate)

        self.r1 = SnnLayer(hidden_dims[1], hidden_dims[1], is_rec=True, is_adapt=is_adapt,
                           one_to_one=one_to_one)

        self.r2 = SnnLayer(hidden_dims[3], hidden_dims[3], is_rec=True, is_adapt=is_adapt,
                           one_to_one=one_to_one)

        self.r3 = SnnLayer(hidden_dims[4], hidden_dims[4], is_rec=True, is_adapt=is_adapt,
                           one_to_one=one_to_one)

        # r in to r out
        self.bu1 = nn.Linear(hidden_dims[1], hidden_dims[0])
        nn.init.xavier_uniform_(self.bu1.weight)

        self.bu2 = nn.Linear(hidden_dims[0], hidden_dims[4])
        nn.init.xavier_uniform_(self.bu2.weight)

        self.bu3 = nn.Linear(hidden_dims[4], hidden_dims[3])
        nn.init.xavier_uniform_(self.bu3.weight)

        self.bu4 = nn.Linear(hidden_dims[3], hidden_dims[2])
        nn.init.xavier_uniform_(self.bu4.weight)

        # r out to r in
        self.td1 = nn.Linear(hidden_dims[0], hidden_dims[1])
        nn.init.xavier_uniform_(self.td1.weight)

        self.td2 = nn.Linear(hidden_dims[4], hidden_dims[0])
        nn.init.xavier_uniform_(self.td2.weight)

        self.td3 = nn.Linear(hidden_dims[3], hidden_dims[4])
        nn.init.xavier_uniform_(self.td3.weight)

        self.td4 = nn.Linear(hidden_dims[2], hidden_dims[3])
        nn.init.xavier_uniform_(self.td4.weight)

        self.p1 = SnnLayer(hidden_dims[0], hidden_dims[0], is_rec=True, is_adapt=is_adapt,
                           one_to_one=one_to_one)

        self.p2 = SnnLayer(hidden_dims[2], hidden_dims[2], is_rec=True, is_adapt=is_adapt,
                           one_to_one=one_to_one)

        self.output_layer = OutputLayer(hidden_dims[2], out_dim, is_fc=False)

    def forward(self, x_t, h):
        batch_dim, input_size = x_t.shape

        x_t = x_t.reshape(batch_dim, input_size).float()
        x_t = self.dp(x_t)
        # poisson
        # x_t = x_t.gt(0.5).float()

        r_input1 = x_t + self.td1(h[5])

        mem_r1, spk_r1, curr_r1, b_r1 = self.r1(r_input1, mem_t=h[0], spk_t=h[1], curr_t=h[2], b_t=h[3])

        p1_input = self.bu1(spk_r1) + self.td2(h[9])

        mem_p1, spk_p1, curr_p1, b_p1 = self.p1(p1_input, mem_t=h[4], spk_t=h[5], curr_t=h[6], b_t=h[7])

        r3_input = self.bu2(spk_p1) + self.td3(h[13])  # + self.rin2rout1(spk_r1)

        mem_r3, spk_r3, curr_r3, b_r3 = self.r2(r3_input, mem_t=h[8], spk_t=h[9], curr_t=h[10], b_t=h[11])

        r2_input = self.bu3(spk_r3) + self.td4(h[17])

        mem_r2, spk_r2, curr_r2, b_r2 = self.r2(r2_input, mem_t=h[12], spk_t=h[13], curr_t=h[14], b_t=h[15])

        p2_input = self.bu4(spk_r2)

        mem_p2, spk_p2, curr_p2, b_p2 = self.p2(p2_input, mem_t=h[16], spk_t=h[17], curr_t=h[18], b_t=h[19])

        self.fr_p = self.fr_p + spk_p1.detach().cpu().numpy().mean() / 2 + spk_p2.detach().cpu().numpy().mean() / 2
        self.fr_r = self.fr_r + spk_r1.detach().cpu().numpy().mean() / 2 + spk_r2.detach().cpu().numpy().mean() / 2

        # read out from r_out neurons
        mem_out = self.output_layer(spk_p2, h[-1])

        h = (mem_r1, spk_r1, curr_r1, b_r1,
             mem_p1, spk_p1, curr_p1, b_p1,
             mem_r3, spk_r3, curr_r3, b_r3,
             mem_r2, spk_r2, curr_r2, b_r2,
             mem_p2, spk_p2, curr_p2, b_p2,
             mem_out)

        log_softmax = F.log_softmax(mem_out, dim=1)

        return log_softmax, h

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (
            # r1
            weight.new(bsz, self.hidden_dims[1]).uniform_(),
            weight.new(bsz, self.hidden_dims[1]).zero_(),
            weight.new(bsz, self.hidden_dims[1]).zero_(),
            weight.new(bsz, self.hidden_dims[1]).fill_(b_j0),
            # p1
            weight.new(bsz, self.hidden_dims[0]).uniform_(),
            weight.new(bsz, self.hidden_dims[0]).zero_(),
            weight.new(bsz, self.hidden_dims[0]).zero_(),
            weight.new(bsz, self.hidden_dims[0]).fill_(b_j0),
            # r3
            weight.new(bsz, self.hidden_dims[4]).uniform_(),
            weight.new(bsz, self.hidden_dims[4]).zero_(),
            weight.new(bsz, self.hidden_dims[4]).zero_(),
            weight.new(bsz, self.hidden_dims[4]).fill_(b_j0),
            # r2
            weight.new(bsz, self.hidden_dims[3]).uniform_(),
            weight.new(bsz, self.hidden_dims[3]).zero_(),
            weight.new(bsz, self.hidden_dims[3]).zero_(),
            weight.new(bsz, self.hidden_dims[3]).fill_(b_j0),
            # p
            weight.new(bsz, self.hidden_dims[2]).uniform_(),
            weight.new(bsz, self.hidden_dims[2]).zero_(),
            weight.new(bsz, self.hidden_dims[2]).zero_(),
            weight.new(bsz, self.hidden_dims[2]).fill_(b_j0),
            # layer out
            weight.new(bsz, self.out_dim).zero_(),
            # sum spike
            weight.new(bsz, self.out_dim).zero_(),
        )
