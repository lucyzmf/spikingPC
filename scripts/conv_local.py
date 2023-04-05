# %%
import torch
import torch.nn as nn
from network_class import *
from network_class import SnnNetwork, SnnLayer, OutputLayer
from torch.nn.modules.utils import _pair


class LocalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride, bias=False):
        super(LocalConv2d, self).__init__()
        output_size = _pair(output_size)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size ** 2)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1])
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)

    def forward(self, x):
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out


class LocalConvTrans2d(nn.Module):
    def __init__(self, in_channels, out_channels, input_size, output_size, kernel_size, stride, bias=False):
        super(LocalConvTrans2d, self).__init__()
        input_size = _pair(input_size)
        output_size = _pair(output_size)
        self.weight = nn.Parameter(
            torch.randn(input_size[0]*input_size[1], in_channels, out_channels * kernel_size**2)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(out_channels, output_size[0], output_size[1])
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.fold = nn.Fold(output_size=output_size, kernel_size=kernel_size)
        self.out_channels = out_channels

    def forward(self, x):
        bs, c, h, w = x.size()
        kh, kw = self.kernel_size

        x = x.flatten(start_dim=2).transpose(2, 1).unsqueeze(2)
        patches = torch.zeros((bs, h*w, self.out_channels*kh*kw))  # n, w_in*h_in, c_out*k**2
        for s in range(bs):
            patches[s] = torch.bmm(x[s], self.weight).squeeze()
        out = self.fold(patches.transpose(2, 1))
        if self.bias is not None:
            out += self.bias
        return out


# %%

b_j0 = 0.1  # neural threshold baseline


class SNNLocalConvCell(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 strides: int,
                 padding: int,
                 input_size: list,  # input data size for computing out dim
                 is_rec: bool,
                 pooling_type=None, pool_size=2, pool_strides=2, bias=True,
                 tau_m_init=15.,
                 tau_adap_init=20.,
                 tau_a_init=15.,
                 is_adapt=False,
                 synaptic_curr=None
                 ):
        super(SNNLocalConvCell, self).__init__()

        print('SNN-conv +', pooling_type)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

        self.input_size = input_size

        self.is_adapt = is_adapt
        self.is_rec = is_rec

        self.rnn_name = 'SNN-conv cell rec ' + str(is_rec)
        if pooling_type is not None:
            if pooling_type == 'max':
                self.pooling = nn.MaxPool2d(kernel_size=pool_size, stride=pool_strides, padding=0)
            elif pooling_type == 'avg':
                self.pooling = nn.AvgPool2d(kernel_size=pool_size, stride=pool_strides, padding=0)
            elif pooling_type == 'up':
                self.pooling = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.pooling = None

        self.local_conv = LocalConv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size,
                                      stride=self.strides, bias=bias)

        self.output_shape = self.compute_output_shape()
        print('output shape ' + str(self.output_shape))
        self.output_size = self.output_shape[0] * self.output_shape[1] * self.output_shape[2]
        print('output size %i' % self.output_size)

        if self.is_rec:
            self.local_rec = nn.Parameter(
                torch.Tensor(self.output_shape[1] * self.output_shape[2], self.out_channels, self.out_channels))
            nn.init.xavier_uniform_(self.local_rec)

        self.sigmoid = nn.Sigmoid()

        self.BN = nn.BatchNorm2d(num_features=self.output_shape[0])  # num features = channel size

        self.tau_m = nn.Parameter(torch.Tensor(self.output_shape))
        self.tau_adp = nn.Parameter(torch.Tensor(self.output_shape))
        self.tau_a = nn.Parameter(torch.Tensor(self.output_shape))
        # self.tau_m = nn.Parameter(torch.Tensor([1]))
        # self.tau_adp = nn.Parameter(torch.Tensor([1]))
        # self.tau_a = nn.Parameter(torch.Tensor([1]))

        # nn.init.kaiming_normal_(self.conv1_x.weight
        nn.init.xavier_normal_(self.local_conv.weight)
        if bias:
            nn.init.constant_(self.local_conv.bias, 0)

        nn.init.normal_(self.tau_adp, tau_adap_init, .1)
        nn.init.normal_(self.tau_m, tau_m_init, .1)
        nn.init.normal_(self.tau_a, tau_a_init, 0.1)
        # nn.init.constant_(self.tau_adp, tau_adap_init)
        # nn.init.constant_(self.tau_m, tau_m_init)
        # nn.init.constant_(self.tau_a, tau_a_init)

    def mem_update(self, ff, fb, soma, spike, a_curr, b, is_adapt, dt=0.5, baseline_thre=b_j0, r_m=3):
        # alpha = self.sigmoid(self.tau_m)
        # rho = self.sigmoid(self.tau_adp)
        # eta = self.sigmoid(self.tau_a)
        alpha = torch.exp(-dt / self.tau_m)
        rho = torch.exp(-dt / self.tau_adp)
        eta = torch.exp(-dt / self.tau_a)

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

    def forward(self, ff, fb, soma_t, spk_t, a_curr_t, b_t, top_down_sig=None):
        ff = self.BN(self.local_conv(ff.float()))
        b, _, _, _ = ff.size()
        # conv_bnx = self.conv1_x(x_t.float())

        if self.pooling is not None:
            conv_x = self.pooling(ff)
        else:
            conv_x = ff

        if self.is_rec:
            reshaped = torch.permute(spk_t.flatten(start_dim=2), (2, 1, 0))
            rec_in = torch.bmm(self.local_rec, reshaped)
            conv_x = conv_x + torch.permute(rec_in, (2, 1, 0)).reshape(b, self.output_shape[0],
                                                                       self.output_shape[1], self.output_shape[2])

        fb = fb.reshape(b, self.output_shape[0], self.output_shape[1], self.output_shape[2])

        # conv_x = F.normalize(torch.flatten(conv_x, start_dim=2), dim=1).reshape(b, self.output_shape[0],
        #                                            self.output_shape[1], self.output_shape[2])

        soma_t1, spk_t1, a_curr_t1, _, b_t1 = self.mem_update(conv_x, fb, soma_t, spk_t, a_curr_t, b_t, self.is_adapt)

        return soma_t1, spk_t1, a_curr_t1, b_t1  # , conv_bnx

    def compute_output_shape(self):
        x_emp = torch.randn([1, self.input_size[0], self.input_size[1], self.input_size[2]])
        out = self.local_conv(x_emp)
        if self.pooling is not None:
            out = self.pooling(out)
        return out.shape[1:]


class SnnLocalConvNet(nn.Module):
    def __init__(
            self,
            input_size,  # data size
            hidden_channels: list,  # number of feature maps/channels per layer
            kernel_size: list,
            stride: list,
            paddings: list,
            out_dim: int,  # num classes
            is_rec: list,
            is_adapt_conv: bool,
            dp_rate=0.3,
            p_size=10,  # num prediction neurons
            num_classes=10,
            pooling=None,
    ):
        super(SnnLocalConvNet, self).__init__()

        self.hidden_channels = hidden_channels
        self.out_dim = out_dim
        self.is_adapt_conv = is_adapt_conv
        self.pooling = pooling
        self.dp = nn.Dropout2d(dp_rate)
        self.input_size = input_size  # c*h*w
        self.classify_population_sz = out_dim * p_size

        self.conv1 = SNNLocalConvCell(input_size[0], hidden_channels[0], kernel_size[0], stride[0], paddings[0],
                                      self.input_size, is_adapt=is_adapt_conv, pooling_type=None, is_rec=is_rec[0])

        self.conv2 = SNNLocalConvCell(hidden_channels[0], hidden_channels[1], kernel_size[1], stride[1], paddings[1],
                                      self.conv1.output_shape, is_adapt=is_adapt_conv, pooling_type=pooling,
                                      is_rec=is_rec[1])

        self.input_to_pc_sz = self.conv2.output_shape[0] * self.conv2.output_shape[1] * self.conv2.output_shape[2]
        # self.input_to_pc_sz = self.conv1.output_shape[0] * self.conv1.output_shape[1] * self.conv1.output_shape[2]

        self.conv_to_pop = nn.Linear(self.input_to_pc_sz, p_size * num_classes)
        nn.init.xavier_uniform_(self.conv_to_pop.weight)

        self.pop_enc = SnnLayer(p_size * num_classes, p_size * num_classes, is_rec=True,
                                is_adapt=True, one_to_one=True)

        # feedback weights
        self.out2pop = nn.Linear(num_classes, p_size * num_classes)
        nn.init.xavier_uniform_(self.out2pop.weight)

        self.pop_to_conv = nn.Linear(p_size * num_classes, self.input_to_pc_sz)
        nn.init.xavier_uniform_(self.pop_to_conv.weight)

        self.deconv1 = LocalConvTrans2d(in_channels=hidden_channels[1], out_channels=hidden_channels[0],
                                        input_size=self.conv2.output_shape[1], output_size=self.conv1.output_shape[1],
                                        kernel_size=kernel_size[0])

        self.output_layer = OutputLayer(p_size * num_classes, out_dim, is_fc=False, tau_fixed=0.2)

        self.neuron_count = self.conv1.output_size + self.pop_enc.hidden_dim + self.conv2.output_size  # + \
        # self.h_layer.hidden_dim

        self.fr_conv1 = 0
        self.fr_conv2 = 0
        self.fr_h = 0
        self.fr_pop = 0

        self.error1 = 0
        self.error2 = 0
        self.error3 = 0
        # self.error_h = 0

    def forward(self, x_t, h):
        batch_dim, c, height, width = x_t.size()
        x_t = self.dp(x_t)
        # poisson 
        # x_t = x_t.gt(0.5).float()

        soma_conv1, spk_conv1, a_curr_conv1, b_conv1 = self.conv1(ff=x_t, fb=self.deconv2(h[5]), soma_t=h[0],
                                                                  spk_t=h[1],
                                                                  a_curr_t=h[2], b_t=h[3])

        self.error1 = a_curr_conv1 - soma_conv1

        soma_conv2, spk_conv2, a_curr_conv2, b_conv2 = self.conv2(ff=spk_conv1, fb=self.pop_to_conv(h[9]), soma_t=h[4],
                                                                  spk_t=h[5], a_curr_t=h[6], b_t=h[7])

        self.error2 = a_curr_conv2 - soma_conv2

        in_to_pop = self.conv_to_pop(spk_conv2.view(batch_dim, -1))
        # in_to_pc = spk2.view(batch_dim, -1)

        soma_pop, spk_pop, a_curr_pop, b_pop = self.pop_enc(ff=in_to_pop, fb=self.out2pop(F.normalize(h[-1], dim=1)),
                                                            soma_t=h[8], spk_t=h[9], a_curr_t=h[10], b_t=h[11])

        self.error3 = a_curr_pop - soma_pop

        mem_out = self.output_layer(spk_pop, h[-1])

        h = (  # soma_h, spk_h.view(batch_dim, -1), a_curr_h, b_h,
            soma_conv1, spk_conv1, a_curr_conv1, b_conv1,
            soma_conv2, spk_conv2, a_curr_conv2, b_conv2,
            soma_pop, spk_pop, a_curr_pop, b_pop,
            mem_out
        )

        self.fr_conv1 = self.fr_conv1 + spk_conv1.detach().cpu().numpy().mean()
        self.fr_conv2 = self.fr_conv2 + spk_conv2.detach().cpu().numpy().mean()
        # self.fr_h = self.fr_h + spk_h.detach().cpu().numpy().mean()
        self.fr_pop = self.fr_pop + spk_pop.detach().cpu().numpy().mean()

        log_softmax = F.log_softmax(mem_out, dim=1)

        return log_softmax, h

    def inference(self, x_t, h, T):
        log_softmax_hist = []
        h_hist = []

        for t in range(T):
            log_softmax, h = self.forward(x_t, h)

            log_softmax_hist.append(log_softmax)
            h_hist.append(h)

        return log_softmax_hist, h_hist

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        sz1 = self.conv1.output_shape  # torch size object
        sz2 = self.conv2.output_shape
        return (
            # conv1
            weight.new(bsz, sz1[0], sz1[1], sz1[2]).uniform_(),
            weight.new(bsz, sz1[0], sz1[1], sz1[2]).zero_(),
            weight.new(bsz, sz1[0], sz1[1], sz1[2]).zero_(),
            weight.new(bsz, sz1[0], sz1[1], sz1[2]).fill_(b_j0),
            # conv2
            weight.new(bsz, sz2[0], sz2[1], sz2[2]).uniform_(),
            weight.new(bsz, sz2[0], sz2[1], sz2[2]).zero_(),
            weight.new(bsz, sz2[0], sz2[1], sz2[2]).zero_(),
            weight.new(bsz, sz2[0], sz2[1], sz2[2]).fill_(b_j0),
            # pop encode
            weight.new(bsz, self.pop_enc.hidden_dim).uniform_(),
            weight.new(bsz, self.pop_enc.hidden_dim).zero_(),
            weight.new(bsz, self.pop_enc.hidden_dim).zero_(),
            weight.new(bsz, self.pop_enc.hidden_dim).fill_(b_j0),
            # layer out
            weight.new(bsz, self.out_dim).zero_()
        )
