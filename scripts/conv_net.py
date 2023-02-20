import torch
import torch.nn as nn
from network import *
from network_class import SnnNetwork

b_j0 = 0.1  # neural threshold baseline


class SNNConvCell(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 strides: int,
                 padding: int,
                 input_size: list,  # input data size for computing out dim
                 pooling_type=None, pool_size=2, pool_strides=2, bias=True,
                 tau_m_init=3.,
                 tau_adap_init=4.6,
                 tau_i_init=0.,
                 is_adapt=False,
                 synaptic_curr=None
                 ):
        super(SNNConvCell, self).init()

        print('SNN-conv +', pooling_type)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

        self.input_size = input_size
        self.output_size = self.compute_output_size()

        self.is_adapt = is_adapt
        self.synap_curr = synaptic_curr

        self.rnn_name = 'SNN-conv cell'
        if pooling_type is not None:
            if pooling_type == 'max':
                self.pooling = nn.MaxPool2d(kernel_size=pool_size, stride=pool_strides, padding=0)
            elif pooling_type == 'avg':
                self.pooling = nn.AvgPool2d(kernel_size=pool_size, stride=pool_strides, padding=0)
            elif pooling_type == 'up':
                self.pooling = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.pooling = None

        self.conv1_x = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.strides,
                                 padding=self.padding, bias=bias)

        self.sig1 = nn.Sigmoid()

        self.BN = nn.BatchNorm2d(num_features=self.output_size[0])  # num features = channel size

        self.tau_m = nn.Parameter(torch.Tensor(self.output_size))
        self.tau_adp = nn.Parameter(torch.Tensor(self.output_size))
        self.tau_i = nn.Parameter(torch.Tensor(self.output_size))

        # nn.init.kaiming_normal_(self.conv1_x.weight
        nn.init.xavier_normal_(self.conv1_x.weight)
        if bias:
            nn.init.constant_(self.conv1_x.bias, 0)

        nn.init.normal_(self.tau_adp, tau_adap_init, .1)
        nn.init.normal_(self.tau_m, tau_m_init, .1)
        nn.init.normal_(self.tau_i, tau_i_init, 0.1)

    def mem_update(self, inputs, mem, spike, current, b, is_adapt, dt=1, baseline_thre=b_j0, r_m=3):
        alpha = self.sigmoid(self.tau_m)
        rho = self.sigmoid(self.tau_adp)
        if self.synap_curr:
            eta = self.sigmoid(self.tau_i)
        else:
            eta = 0
        # alpha = torch.exp(-dt/self.tau_m)
        # rho = torch.exp(-dt/self.tau_adp)

        if is_adapt:
            beta = 1.8
        else:
            beta = 0.

        b = rho * b + (1 - rho) * spike  # adaptive contribution
        new_thre = baseline_thre + beta * b  # udpated threshold

        current = eta * current + (1 - eta) * inputs

        # mem = mem * alpha + (1 - alpha) * r_m * inputs - new_thre * spike
        mem = mem * alpha + current - new_thre * spike  # soft reset
        inputs_ = mem - new_thre

        spike = act_fun_adp(inputs_)  # act_fun : approximation firing function
        # mem = (1 - spike) * mem

        return mem, spike, current, new_thre, b

    def forward(self, x_t, mem_t, spk_t, curr_t, b_t):
        conv_bnx = self.BN(self.conv1_x(x_t.float()))

        if self.pooling is not None:
            conv_x = self.pooling(conv_bnx)
        else:
            conv_x = conv_bnx

        mem, spk, curr, _, b = self.mem_update(conv_x, mem_t, spk_t, curr_t, b_t, self.is_adapt)

        return mem, spk, curr, b, conv_bnx

    def compute_output_size(self):
        x_emp = torch.randn([1, self.input_size[0], self.input_size[1], self.input_size[2]])
        out = self.conv1_x(x_emp)
        if self.pooling is not None:
            out = self.pooling(out)
        return out.shape[1:]


class SnnConvNet(nn.Module):
    def __init__(
            self,
            input_size,  # data size
            hidden_channels: list,  # number of feature maps/channels per layer
            kernel_size: list,
            stride: list,
            paddings: list,
            out_dim: int,  # num classes
            is_adapt_conv: bool,
            syn_curr_conv: bool,
            dp_rate=0.2,
            p_size=10,  # num prediction neurons
    ):
        super(SnnConvNet, self).__init__()

        self.hidden_channels = hidden_channels
        self.out_dim = out_dim
        self.is_adapt_conv = is_adapt_conv
        self.syn_curr = syn_curr_conv
        self.dp = nn.Dropout2d(dp_rate)
        self.input_size = input_size  # c*h*w

        self.conv1 = SNNConvCell(input_size[0], hidden_channels[0], kernel_size[0], stride[0], paddings[0],
                                 self.input_size, is_adapt=is_adapt_conv)

        self.conv2 = SNNConvCell(hidden_channels[0], hidden_channels[1], kernel_size[1], stride[1], paddings[1],
                                 self.conv1.output_size, is_adapt=is_adapt_conv)

        self.input_to_pc = self.conv2.output_size[0] * self.conv2.output_size[1] * self.conv2.output_size[2]

        # last layer pc inference
        self.pc_layer = SnnNetwork(in_dim=self.input_to_pc,
                                   hidden_dims=[out_dim * p_size, self.input_to_pc], out_dim=self.out_dim,
                                   dp_rate=dp_rate, is_adapt=True, one_to_one=True)

        self.fr_conv1 = 0
        self.fr_conv2 = 0

    def forward(self, x_t, h_conv, h_pc):
        batch_dim, c, h, w = x_t.size()
        x_t = self.dp(x_t)

        mem1, spk1, curr1, b1 = self.conv1(x_t, mem_t=h_conv[0], spk_t=h_conv[1], curr_t=h_conv[2], b_t=h_conv[3])
        mem2, spk2, curr2, b2 = self.conv1(x_t, mem_t=h_conv[4], spk_t=h_conv[5], curr_t=h_conv[6], b_t=h_conv[7])

        log_out, h_pc = self.pc_layer(spk2.flatten(), h_pc)

        h_conv = (mem1, spk1, curr1, b1,
                  mem2, spk2, curr2, b2)

        return log_out, h_conv, h_pc

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (
            # conv1
            weight.new(bsz, self.conv1.output_size).uniform_(),
            weight.new(bsz, self.conv1.output_size).zero_(),
            weight.new(bsz, self.conv1.output_size).zero_(),
            weight.new(bsz, self.conv1.output_size).fill_(b_j0),
            # conv2
            weight.new(bsz, self.conv2.output_size).uniform_(),
            weight.new(bsz, self.conv2.output_size).zero_(),
            weight.new(bsz, self.conv2.output_size).zero_(),
            weight.new(bsz, self.conv2.output_size).fill_(b_j0)
        )
