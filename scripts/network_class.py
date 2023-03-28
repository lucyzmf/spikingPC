# %%
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

b_j0 = 0.1  # neural threshold baseline

R_m = 3  # membrane resistance
dt = 1
gamma = .5  # gradient scale
lens = 0.5

bias_bool=True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def shifted_sigmoid(currents):
    return (1 / (1 + torch.exp(-currents)) - 0.5)/2


class SnnLayer(nn.Module):
    def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            is_rec: bool,
            is_adapt: bool,
            one_to_one: bool,
            tau_m_init=15.,
            tau_adap_init=20,
            tau_a_init=15.
    ):
        super(SnnLayer, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.is_rec = is_rec
        self.is_adapt = is_adapt
        self.one_to_one = one_to_one

        if is_rec:
            self.rec_w = nn.Linear(hidden_dim, hidden_dim, bias=bias_bool)
            # init weights
            nn.init.xavier_uniform_(self.rec_w.weight)

            p = torch.full(self.rec_w.weight.size(), fill_value=0.5).to(device)
            self.weight_mask = torch.bernoulli(p)

        else:
            self.fc_weights = nn.Linear(in_dim, hidden_dim, bias=bias_bool)
            nn.init.xavier_uniform_(self.fc_weights.weight)

        # define param for time constants
        self.tau_adp = nn.Parameter(torch.Tensor(hidden_dim))
        self.tau_m = nn.Parameter(torch.Tensor(hidden_dim))
        self.tau_a = nn.Parameter(torch.Tensor(hidden_dim))

        nn.init.normal_(self.tau_adp, tau_adap_init, .1)
        nn.init.normal_(self.tau_m, tau_m_init, .1)
        nn.init.normal_(self.tau_a, tau_a_init, 0.1)

        # self.tau_adp = nn.Parameter(torch.Tensor(1))
        # self.tau_m = nn.Parameter(torch.Tensor(1))
        # self.tau_a = nn.Parameter(torch.Tensor(1))

        # nn.init.constant_(self.tau_adp, tau_adap_init)
        # nn.init.constant_(self.tau_m, tau_m_init)
        # nn.init.constant_(self.tau_a, tau_i_init)

        # nn.init.normal_(self.tau_adp, 200., 20.)
        # nn.init.normal_(self.tau_m, 20., .5)

        self.sigmoid = nn.Sigmoid()

    def mem_update(self, ff, fb, soma, spike, a_curr, b, is_adapt, dt=0.5, baseline_thre=b_j0, r_m=3):
        """
        mem update for each layer of neurons
        :param ff: feedforward signal
        :param fb: feedback signal to apical tuft
        :param soma: mem voltage potential at soma
        :param spike: spiking at last time step
        :param a_curr: apical tuft current at last t
        :param b: adaptive threshold
        :return:
        """
        # alpha = self.sigmoid(self.tau_m)
        # rho = self.sigmoid(self.tau_adp)
        # eta = self.sigmoid(self.tau_a)
        alpha = torch.exp(-dt/self.tau_m)
        rho = torch.exp(-dt/self.tau_adp)
        eta = torch.exp(-dt/self.tau_a)

        if is_adapt:
            beta = 1.8
        else:
            beta = 0.

        b = rho * b + (1 - rho) * spike  # adaptive contribution
        new_thre = baseline_thre + beta * b  # udpated threshold

        a_new = eta * a_curr + fb  # fb into apical tuft

        soma_new = alpha * soma + shifted_sigmoid(a_new) + ff - new_thre * spike
        inputs_ = soma_new - new_thre

        spike = act_fun_adp(inputs_)  # act_fun : approximation firing function
        # mem = (1 - spike) * mem

        return soma_new, spike, a_new, new_thre, b

    def forward(self, ff, fb, soma_t, spk_t, a_curr_t, b_t):
        """
        forward function of a single layer. given previous neuron states and current input, update neuron states

        :param ff: ff signal (not counting rec)
        :param fb: fb top down signal
        :param soma_t: soma voltage
        :param a_curr_t: apical tuft voltage
        :return:
        """

        if self.is_rec:
            self.rec_w.weight.data = self.rec_w.weight.data * self.weight_mask
            # self.rec_w.weight.data = (self.rec_w.weight.data < 0).float() * self.rec_w.weight.data
            r_in = ff + self.rec_w(spk_t)
        else:
            if self.one_to_one:
                r_in = ff
            else:
                r_in = self.fc_weights(ff)

        soma_t1, spk_t1, a_curr_t1, _, b_t1 = self.mem_update(r_in, fb, soma_t, spk_t, a_curr_t, b_t, self.is_adapt)

        return soma_t1, spk_t1, a_curr_t1, b_t1


class OutputLayer(nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            is_fc: bool,
            tau_fixed=None
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
            self.fc = nn.Linear(in_dim, out_dim, bias=bias_bool)
            nn.init.xavier_uniform_(self.fc.weight)

        # tau_m
        if tau_fixed is None:
            self.tau_m = nn.Parameter(torch.Tensor(out_dim))
            nn.init.constant_(self.tau_m, 5)
        else:
            self.tau_m = nn.Parameter(torch.Tensor(out_dim), requires_grad=False)
            nn.init.constant_(self.tau_m, tau_fixed)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x_t, mem_t, dt=1):
        """
        integrator neuron without spikes
        """
        alpha = torch.exp(-1/self.tau_m)
        # alpha = self.sigmoid(self.tau_m)

        if self.is_fc:
            x_t = self.fc(x_t)
        else:
            x_t = x_t.view(-1, 10, int(self.in_dim / 10)).mean(dim=2)  # sum up population spike

        # d_mem = -mem_t + x_t
        mem = (mem_t + x_t) * alpha
        # mem = alpha * mem_t + (1 - alpha) * x_t
        return mem


class SnnNetwork(nn.Module):
    def __init__(
            self,
            in_dim: int,
            hidden_dims: list,
            out_dim: int,
            is_adapt: bool,
            one_to_one: bool,
            dp_rate: float, 
            is_rec: bool
    ):
        super(SnnNetwork, self).__init__()

        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.is_adapt = is_adapt
        self.one_to_one = one_to_one
        self.is_rec = is_rec

        self.dp = nn.Dropout(dp_rate)

        self.layer1 = SnnLayer(hidden_dims[0], hidden_dims[0], is_rec=is_rec, is_adapt=is_adapt,
                               one_to_one=one_to_one)

        # r in to r out
        self.layer1to2 = nn.Linear(hidden_dims[0], hidden_dims[1], bias=bias_bool)
        nn.init.xavier_uniform_(self.layer1to2.weight)

        # r out to r in
        self.layer2to1 = nn.Linear(hidden_dims[1], hidden_dims[0], bias=bias_bool)
        nn.init.xavier_uniform_(self.layer2to1.weight)

        self.layer2 = SnnLayer(hidden_dims[1], hidden_dims[1], is_rec=is_rec, is_adapt=is_adapt,
                               one_to_one=one_to_one)

        self.output_layer = OutputLayer(hidden_dims[1], out_dim, is_fc=True)

        self.out2layer2 = nn.Linear(out_dim, hidden_dims[1], bias=bias_bool)
        nn.init.xavier_uniform_(self.out2layer2.weight)

        self.fr_layer2 = 0
        self.fr_layer1 = 0

        self.error1 = 0
        self.error2 = 0

    def forward(self, x_t, h):
        batch_dim, input_size = x_t.shape

        x_t = x_t.reshape(batch_dim, input_size).float()
        x_t = self.dp(x_t*0.5)
        # poisson 
        # x_t = x_t.gt(0.7).float()

        soma_1, spk_1, a_curr_1, b_1 = self.layer1(ff=x_t, fb=self.layer2to1(h[5]), soma_t=h[0], spk_t=h[1],
                                                   a_curr_t=h[2], b_t=h[3])

        self.error1 = a_curr_1 - soma_1

        # use out mem signal as feedback
        soma_2, spk_2, a_curr_2, b_2 = self.layer2(ff=self.layer1to2(spk_1), fb=self.out2layer2(F.normalize(h[-1], dim=1)), soma_t=h[4],
                                                   spk_t=h[5], a_curr_t=h[6], b_t=h[7])

        self.error2 = a_curr_2 - soma_2

        self.fr_layer2 = self.fr_layer2 + spk_2.detach().cpu().numpy().mean()
        self.fr_layer1 = self.fr_layer1 + spk_1.detach().cpu().numpy().mean()

        # read out from r_out neurons
        mem_out = self.output_layer(spk_2, h[-1])

        h = (soma_1, spk_1, a_curr_1, b_1,
             soma_2, spk_2, a_curr_2, b_2,
             mem_out)

        log_softmax = F.log_softmax(mem_out, dim=1)

        return log_softmax, h

    def inference(self, x_t, h, T, bystep=None):
        """
        only called during inference
        :param x_t: input
        :param h: hidden states
        :param T: sequence length
        :param bystep: if true, then x_t is a sequence 
        :return:
        """

        log_softmax_hist = []
        h_hist = []

        for t in range(T): 
            if bystep is None:
                log_softmax, h = self.forward(x_t, h)
            else:
                log_softmax, h = self.forward(x_t[t], h)


            log_softmax_hist.append(log_softmax)
            h_hist.append(h)

        return log_softmax_hist, h_hist
    
    def clamped_generate(self, test_class, zeros, h_clamped, T, clamp_value=0.5, batch=False):
        """
        generate representations with mem of read out clamped 
        :param test_class: which class is clamped 
        :param zeros: input containing zeros, absence of input 
        :param h: hidden states
        :param T: sequence length
        :return:
        """

        log_softmax_hist = []
        h_hist = []

        for t in range(T):
            if not batch:
                h_clamped[-1][0] = -clamp_value
                h_clamped[-1][0, test_class] = clamp_value
            else:
                h_clamped[-1][:, :] = torch.full(h_clamped[-1].size(), -clamp_value).to(device)
                h_clamped[-1][:, test_class] = clamp_value

            # if t==0:
            #     print(h_clamped[-1])

            log_softmax, h_clamped = self.forward(zeros, h_clamped)

            log_softmax_hist.append(log_softmax)
            h_hist.append(h_clamped)

        return log_softmax_hist, h_hist

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (
            # r
            weight.new(bsz, self.hidden_dims[0]).uniform_(),
            weight.new(bsz, self.hidden_dims[0]).zero_(),
            weight.new(bsz, self.hidden_dims[0]).zero_(),
            weight.new(bsz, self.hidden_dims[0]).fill_(b_j0),
            # p
            weight.new(bsz, self.hidden_dims[1]).uniform_(),
            weight.new(bsz, self.hidden_dims[1]).zero_(),
            weight.new(bsz, self.hidden_dims[1]).zero_(),
            weight.new(bsz, self.hidden_dims[1]).fill_(b_j0),
            # layer out
            weight.new(bsz, self.out_dim).zero_(),
            # sum spike
            weight.new(bsz, self.out_dim).zero_(),
        )


# %%

class SnnNetwork2Layer(SnnNetwork):
    def __init__(
            self,
            in_dim: int,
            hidden_dims: list,  
            out_dim: int,
            is_adapt: bool,
            one_to_one: bool,
            dp_rate: float,
            is_rec: bool
    ):
        super().__init__(in_dim, hidden_dims, out_dim, is_adapt, one_to_one, dp_rate, is_rec)

        self.layer3 = SnnLayer(hidden_dims[2], hidden_dims[2], is_rec=is_rec, is_adapt=is_adapt,
                               one_to_one=one_to_one)

        self.layer2to3 = nn.Linear(hidden_dims[1], hidden_dims[2], bias=bias_bool)
        nn.init.xavier_uniform_(self.layer2to3.weight)

        # r out to r in
        self.layer3to2 = nn.Linear(hidden_dims[2], hidden_dims[1], bias=bias_bool)
        nn.init.xavier_uniform_(self.layer3to2.weight)

        self.output_layer = OutputLayer(hidden_dims[2], out_dim, is_fc=True)

        self.out2layer3 = nn.Linear(out_dim, hidden_dims[2], bias=bias_bool)
        nn.init.xavier_uniform_(self.out2layer3.weight)

        self.fr_layer3 = 0

        self.error3 = 0


    def forward(self, x_t, h):
        batch_dim, input_size = x_t.shape

        x_t = x_t.reshape(batch_dim, input_size).float()
        x_t = self.dp(x_t*0.5)
        # poisson 
        # x_t = x_t.gt(0.7).float()

        soma_1, spk_1, a_curr_1, b_1 = self.layer1(ff=x_t, fb=self.layer2to1(h[5]), soma_t=h[0], spk_t=h[1],
                                                   a_curr_t=h[2], b_t=h[3])

        self.error1 = a_curr_1 - soma_1

        # use out mem signal as feedback
        soma_2, spk_2, a_curr_2, b_2 = self.layer2(ff=self.layer1to2(spk_1), fb=self.layer3to2(h[9]), soma_t=h[4],
                                                   spk_t=h[5], a_curr_t=h[6], b_t=h[7])

        self.error2 = a_curr_2 - soma_2

        soma_3, spk_3, a_curr_3, b_3 = self.layer3(ff=self.layer2to3(spk_2), fb=self.out2layer3(F.normalize(h[-1], dim=1)), soma_t=h[8],
                                                   spk_t=h[9], a_curr_t=h[10], b_t=h[11])
        # soma_3, spk_3, a_curr_3, b_3 = self.layer3(ff=self.layer2to3(spk_2), fb=0, soma_t=h[8],
        #                                            spk_t=h[9], a_curr_t=h[10], b_t=h[11])

        self.error3 = a_curr_3 - soma_3

        self.fr_layer3 = self.fr_layer3 + spk_3.detach().cpu().numpy().mean()
        self.fr_layer2 = self.fr_layer2 + spk_2.detach().cpu().numpy().mean()
        self.fr_layer1 = self.fr_layer1 + spk_1.detach().cpu().numpy().mean()

        # read out from r_out neurons
        mem_out = self.output_layer(spk_3, h[-1])

        h = (soma_1, spk_1, a_curr_1, b_1,
             soma_2, spk_2, a_curr_2, b_2,
             soma_3, spk_3, a_curr_3, b_3,
             mem_out)

        log_softmax = F.log_softmax(mem_out, dim=1)

        return log_softmax, h


    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (
            # l1
            weight.new(bsz, self.hidden_dims[0]).uniform_(),
            weight.new(bsz, self.hidden_dims[0]).zero_(),
            weight.new(bsz, self.hidden_dims[0]).zero_(),
            weight.new(bsz, self.hidden_dims[0]).fill_(b_j0),
            # l2
            weight.new(bsz, self.hidden_dims[1]).uniform_(),
            weight.new(bsz, self.hidden_dims[1]).zero_(),
            weight.new(bsz, self.hidden_dims[1]).zero_(),
            weight.new(bsz, self.hidden_dims[1]).fill_(b_j0),
            # l3
            weight.new(bsz, self.hidden_dims[2]).uniform_(),
            weight.new(bsz, self.hidden_dims[2]).zero_(),
            weight.new(bsz, self.hidden_dims[2]).zero_(),
            weight.new(bsz, self.hidden_dims[2]).fill_(b_j0),
            # layer out
            weight.new(bsz, self.out_dim).zero_(),
            # sum spike
            weight.new(bsz, self.out_dim).zero_(),
        )
    
    def init_hidden_allzero(self, bsz):
        weight = next(self.parameters()).data
        return (
            # l1
            weight.new(bsz, self.hidden_dims[0]).zero_(),
            weight.new(bsz, self.hidden_dims[0]).zero_(),
            weight.new(bsz, self.hidden_dims[0]).zero_(),
            weight.new(bsz, self.hidden_dims[0]).fill_(b_j0),
            # l2
            weight.new(bsz, self.hidden_dims[1]).zero_(),
            weight.new(bsz, self.hidden_dims[1]).zero_(),
            weight.new(bsz, self.hidden_dims[1]).zero_(),
            weight.new(bsz, self.hidden_dims[1]).fill_(b_j0),
            # l3
            weight.new(bsz, self.hidden_dims[2]).zero_(),
            weight.new(bsz, self.hidden_dims[2]).zero_(),
            weight.new(bsz, self.hidden_dims[2]).zero_(),
            weight.new(bsz, self.hidden_dims[2]).fill_(b_j0),
            # layer out
            weight.new(bsz, self.out_dim).zero_(),
            # sum spike
            weight.new(bsz, self.out_dim).zero_(),
        )



# %%
