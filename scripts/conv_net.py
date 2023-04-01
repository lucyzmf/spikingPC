# %%
import torch
import torch.nn as nn
from network_class import *
from network_class import SnnNetwork, SnnLayer, OutputLayer

b_j0 = 0.1  # neural threshold baseline


class SNNConvCell(nn.Module):
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
        super(SNNConvCell, self).__init__()

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

        self.conv_in = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.strides,
                                 padding=self.padding, bias=bias)

        self.output_shape = self.compute_output_shape()
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

        # nn.init.kaiming_normal_(self.conv1_x.weight
        nn.init.xavier_normal_(self.conv_in.weight)
        if bias:
            nn.init.constant_(self.conv_in.bias, 0)

        nn.init.normal_(self.tau_adp, tau_adap_init, .1)
        nn.init.normal_(self.tau_m, tau_m_init, .1)
        nn.init.normal_(self.tau_a, tau_a_init, 0.1)

    def mem_update(self, ff, fb, soma, spike, a_curr, b, is_adapt, dt=1, baseline_thre=b_j0, r_m=3):
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
        ff = self.BN(self.conv_in(ff.float()))
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
        out = self.conv_in(x_emp)
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
            is_rec: list,
            is_adapt_conv: bool,
            dp_rate=0.3,
            p_size=10,  # num prediction neurons
            num_classes=10,
            pooling=None
    ):
        super(SnnConvNet, self).__init__()

        self.hidden_channels = hidden_channels
        self.out_dim = out_dim
        self.is_adapt_conv = is_adapt_conv
        self.pooling = pooling
        self.dp = nn.Dropout2d(dp_rate)
        self.input_size = input_size  # c*h*w
        self.classify_population_sz = out_dim * p_size

        # ff weights
        self.h_layer = SnnLayer(input_size[1] * input_size[2], input_size[1] * input_size[2], is_rec=False,
                                is_adapt=False,
                                one_to_one=True)

        self.conv1 = SNNConvCell(input_size[0], hidden_channels[0], kernel_size[0], stride[0], paddings[0],
                                 self.input_size, is_adapt=is_adapt_conv, pooling_type=None, is_rec=is_rec[0])

        self.conv2 = SNNConvCell(hidden_channels[0], hidden_channels[1], kernel_size[1], stride[1], paddings[1],
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

        self.deconv2 = nn.ConvTranspose2d(hidden_channels[1], hidden_channels[0], kernel_size=kernel_size[1],
                                          stride=stride[1], padding=paddings[1])

        self.deconv1 = nn.ConvTranspose2d(hidden_channels[0], input_size[0], kernel_size=kernel_size[0],
                                          stride=stride[0], padding=paddings[0])

        self.output_layer = OutputLayer(p_size * num_classes, out_dim, is_fc=False, tau_fixed=0.2)

        self.neuron_count = self.conv1.output_size + self.pop_enc.hidden_dim + self.conv2.output_size + \
                            self.h_layer.hidden_dim

        self.fr_conv1 = 0
        self.fr_conv2 = 0
        self.fr_h = 0
        self.fr_pop = 0

        self.error1 = 0
        self.error2 = 0
        self.error3 = 0
        self.error_h = 0

    def forward(self, x_t, h):
        batch_dim, c, height, width = x_t.size()
        x_t = self.dp(x_t)
        # poisson 
        # x_t = x_t.gt(0.5).float()

        # hidden layer
        # h_input = x_t + self.deconv1(h[5])
        soma_h, spk_h, a_curr_h, b_h = self.h_layer(ff=x_t, fb=self.deconv1(h[5]).view(batch_dim, -1),
                                                    soma_t=h[0], spk_t=h[1], a_curr_t=h[2], b_t=h[3])
        self.error_h = a_curr_h - soma_h

        spk_h = spk_h.reshape(batch_dim, c, height, width)

        soma_conv1, spk_conv1, a_curr_conv1, b_conv1 = self.conv1(ff=spk_h, fb=self.deconv2(h[9]), soma_t=h[4],
                                                                  spk_t=h[5],
                                                                  a_curr_t=h[6], b_t=h[7])

        self.error1 = a_curr_conv1 - soma_conv1

        soma_conv2, spk_conv2, a_curr_conv2, b_conv2 = self.conv2(ff=spk_conv1, fb=self.pop_to_conv(h[13]), soma_t=h[8],
                                                                  spk_t=h[9], a_curr_t=h[10], b_t=h[11])

        self.error2 = a_curr_conv2 - soma_conv2

        in_to_pop = self.conv_to_pop(spk_conv2.view(batch_dim, -1))
        # in_to_pc = spk2.view(batch_dim, -1)

        soma_pop, spk_pop, a_curr_pop, b_pop = self.pop_enc(ff=in_to_pop, fb=self.out2pop(F.normalize(h[-1], dim=1)),
                                                            soma_t=h[12], spk_t=h[13], a_curr_t=h[14], b_t=h[15])

        self.error3 = a_curr_pop - soma_pop

        mem_out = self.output_layer(spk_pop, h[-1])

        h = (soma_h, spk_h.view(batch_dim, -1), a_curr_h, b_h,
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
            # h layer
            # weight.new(bsz, self.h_layer.hidden_dim).uniform_(),
            # weight.new(bsz, self.h_layer.hidden_dim).zero_(),
            # weight.new(bsz, self.h_layer.hidden_dim).zero_(),
            # weight.new(bsz, self.h_layer.hidden_dim).fill_(b_j0),
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


class SnnConvNet1Layer(nn.Module):
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
            syn_curr_conv: bool,
            dp_rate=0.3,
            p_size=10,  # num prediction neurons
            num_classes=10,
            pooling=None
    ):
        super(SnnConvNet1Layer, self).__init__()

        self.hidden_channels = hidden_channels
        self.out_dim = out_dim
        self.is_adapt_conv = is_adapt_conv
        self.syn_curr = syn_curr_conv
        self.pooling = pooling
        self.dp = nn.Dropout2d(dp_rate)
        self.input_size = input_size  # c*h*w
        self.classify_population_sz = out_dim * p_size

        # ff weights
        self.h_layer = SnnLayer(input_size[1] * input_size[2], input_size[1] * input_size[2], is_rec=False,
                                is_adapt=False,
                                one_to_one=True)

        self.conv1 = SNNConvCell(input_size[0], hidden_channels[0], kernel_size[0], stride[0], paddings[0],
                                 self.input_size, is_adapt=is_adapt_conv, pooling_type=None, is_rec=is_rec[0])

        self.input_to_pc_sz = self.conv1.output_shape[0] * self.conv1.output_shape[1] * self.conv1.output_shape[2]

        self.conv_to_pop = nn.Linear(self.input_to_pc_sz, p_size * num_classes)
        nn.init.xavier_uniform_(self.conv_to_pop.weight)

        self.pop_enc = SnnLayer(p_size * num_classes, p_size * num_classes, is_rec=True,
                                is_adapt=True, one_to_one=True)

        # feedback weights
        self.pop_to_conv = nn.Linear(p_size * num_classes, self.input_to_pc_sz)
        nn.init.xavier_uniform_(self.pop_to_conv.weight)

        self.deconv1 = nn.ConvTranspose2d(hidden_channels[0], input_size[0], kernel_size=kernel_size[0],
                                          stride=stride[0], padding=paddings[0])

        self.output_layer = OutputLayer(p_size * num_classes, out_dim, is_fc=False, tau_fixed=True)

        self.neuron_count = self.conv1.output_size + self.h_layer.hidden_dim + self.pop_enc.hidden_dim

        self.fr_conv1 = 0
        self.fr_h = 0
        self.fr_pop = 0

    def forward(self, x_t, h):
        batch_dim, c, height, width = x_t.size()
        x_t = self.dp(x_t)

        # hidden layer
        h_input = x_t + self.deconv1(h[5])
        mem_h, spk_h, curr_h, b_h = self.h_layer(h_input.view(batch_dim, -1), mem_t=h[0], spk_t=h[1], curr_t=h[2],
                                                 b_t=h[3])
        spk_h = spk_h.reshape(batch_dim, c, height, width)

        mem_conv1, spk_conv1, curr_conv1, b_conv1 = self.conv1(spk_h, mem_t=h[4], spk_t=h[5], curr_t=h[6], b_t=h[7],
                                                               top_down_sig=self.pop_to_conv(h[9]))

        in_to_pop = self.conv_to_pop(spk_conv1.view(batch_dim, -1))
        # in_to_pc = spk2.view(batch_dim, -1)

        mem_pop, spk_pop, curr_pop, b_pop = self.pop_enc(in_to_pop, mem_t=h[8], spk_t=h[9], curr_t=h[10],
                                                         b_t=h[11])

        mem_out = self.output_layer(spk_pop, h[-1])

        h = (mem_h, spk_h.view(batch_dim, -1), curr_h, b_h,
             mem_conv1, spk_conv1, curr_conv1, b_conv1,
             mem_pop, spk_pop, curr_pop, b_pop,
             mem_out
             )

        self.fr_conv1 = self.fr_conv1 + spk_conv1.detach().cpu().numpy().mean()
        self.fr_h = self.fr_h + spk_h.detach().cpu().numpy().mean()
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
        return (
            # h layer
            weight.new(bsz, self.h_layer.hidden_dim).uniform_(),
            weight.new(bsz, self.h_layer.hidden_dim).zero_(),
            weight.new(bsz, self.h_layer.hidden_dim).zero_(),
            weight.new(bsz, self.h_layer.hidden_dim).fill_(b_j0),
            # conv1
            weight.new(bsz, sz1[0], sz1[1], sz1[2]).uniform_(),
            weight.new(bsz, sz1[0], sz1[1], sz1[2]).zero_(),
            weight.new(bsz, sz1[0], sz1[1], sz1[2]).zero_(),
            weight.new(bsz, sz1[0], sz1[1], sz1[2]).fill_(b_j0),
            # pop encode
            weight.new(bsz, self.pop_enc.hidden_dim).uniform_(),
            weight.new(bsz, self.pop_enc.hidden_dim).zero_(),
            weight.new(bsz, self.pop_enc.hidden_dim).zero_(),
            weight.new(bsz, self.pop_enc.hidden_dim).fill_(b_j0),
            # layer out
            weight.new(bsz, self.out_dim).zero_()
        )


# %%
class SnnConvNet3Layer(SnnConvNet):
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
            syn_curr_conv: bool,
            dp_rate=0.3,
            p_size=10,  # num prediction neurons
            num_classes=10,
            pooling=None
    ):
        super().__init__(input_size, hidden_channels, kernel_size, stride, paddings, out_dim, is_rec, is_adapt_conv,
                         syn_curr_conv, dp_rate, p_size, num_classes, pooling)

        # ff weights

        self.conv3 = SNNConvCell(hidden_channels[1], hidden_channels[2], kernel_size[2], stride[2], paddings[2],
                                 self.conv2.output_shape, is_adapt=is_adapt_conv, pooling_type=pooling,
                                 is_rec=is_rec[2])

        self.input_to_pc_sz = self.conv3.output_shape[0] * self.conv3.output_shape[1] * self.conv3.output_shape[2]

        self.conv_to_pop = nn.Linear(self.input_to_pc_sz, p_size * num_classes)
        nn.init.xavier_uniform_(self.conv_to_pop.weight)

        # feedback weights
        self.pop_to_conv = nn.Linear(p_size * num_classes, self.input_to_pc_sz)
        nn.init.xavier_uniform_(self.pop_to_conv.weight)

        self.deconv3 = nn.ConvTranspose2d(hidden_channels[2], hidden_channels[1], kernel_size=kernel_size[2],
                                          stride=stride[2], padding=paddings[2])

        self.neuron_count = self.conv1.output_size + self.h_layer.hidden_dim + self.pop_enc.hidden_dim + \
                            self.conv2.output_size + self.conv3.output_size

    def forward(self, x_t, h):
        batch_dim, c, height, width = x_t.size()
        x_t = self.dp(x_t)

        # hidden layer
        h_input = x_t + self.deconv1(h[5])
        mem_h, spk_h, curr_h, b_h = self.h_layer(h_input.view(batch_dim, -1), mem_t=h[0], spk_t=h[1], curr_t=h[2],
                                                 b_t=h[3])
        spk_h = spk_h.reshape(batch_dim, c, height, width)

        mem_conv1, spk_conv1, curr_conv1, b_conv1 = self.conv1(spk_h, mem_t=h[4], spk_t=h[5], curr_t=h[6], b_t=h[7],
                                                               top_down_sig=self.deconv2(h[9]))

        mem_conv2, spk_conv2, curr_conv2, b_conv2 = self.conv2(spk_conv1, mem_t=h[8], spk_t=h[9], curr_t=h[10],
                                                               b_t=h[11],
                                                               top_down_sig=self.deconv3(h[13]))

        mem_conv3, spk_conv3, curr_conv3, b_conv3 = self.conv3(spk_conv2, mem_t=h[12], spk_t=h[13], curr_t=h[14],
                                                               b_t=h[15],
                                                               top_down_sig=self.pop_to_conv(h[17]))

        in_to_pop = self.conv_to_pop(spk_conv3.view(batch_dim, -1))
        # in_to_pc = spk2.view(batch_dim, -1)

        mem_pop, spk_pop, curr_pop, b_pop = self.pop_enc(in_to_pop, mem_t=h[16], spk_t=h[17], curr_t=h[18],
                                                         b_t=h[19])

        mem_out = self.output_layer(spk_pop, h[-1])

        h = (mem_h, spk_h.view(batch_dim, -1), curr_h, b_h,
             mem_conv1, spk_conv1, curr_conv1, b_conv1,
             mem_conv2, spk_conv2, curr_conv2, b_conv2,
             mem_conv3, spk_conv3, curr_conv3, b_conv3,
             mem_pop, spk_pop, curr_pop, b_pop,
             mem_out
             )

        self.fr_conv1 = self.fr_conv1 + spk_conv1.detach().cpu().numpy().mean()
        self.fr_conv2 = self.fr_conv2 + spk_conv2.detach().cpu().numpy().mean()
        self.fr_h = self.fr_h + spk_h.detach().cpu().numpy().mean()
        self.fr_pop = self.fr_pop + spk_pop.detach().cpu().numpy().mean()

        log_softmax = F.log_softmax(mem_out, dim=1)

        return log_softmax, h

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        sz1 = self.conv1.output_shape  # torch size object
        sz2 = self.conv2.output_shape
        sz3 = self.conv3.output_shape
        return (
            # h layer
            weight.new(bsz, self.h_layer.hidden_dim).uniform_(),
            weight.new(bsz, self.h_layer.hidden_dim).zero_(),
            weight.new(bsz, self.h_layer.hidden_dim).zero_(),
            weight.new(bsz, self.h_layer.hidden_dim).fill_(b_j0),
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
            # conv3
            weight.new(bsz, sz3[0], sz3[1], sz3[2]).uniform_(),
            weight.new(bsz, sz3[0], sz3[1], sz3[2]).zero_(),
            weight.new(bsz, sz3[0], sz3[1], sz3[2]).zero_(),
            weight.new(bsz, sz3[0], sz3[1], sz3[2]).fill_(b_j0),
            # pop encode
            weight.new(bsz, self.pop_enc.hidden_dim).uniform_(),
            weight.new(bsz, self.pop_enc.hidden_dim).zero_(),
            weight.new(bsz, self.pop_enc.hidden_dim).zero_(),
            weight.new(bsz, self.pop_enc.hidden_dim).fill_(b_j0),
            # layer out
            weight.new(bsz, self.out_dim).zero_()
        )


class SnnConvNet4Layer(SnnConvNet3Layer):
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
            syn_curr_conv: bool,
            dp_rate=0.3,
            p_size=10,  # num prediction neurons
            num_classes=10,
            pooling=None
    ):
        super().__init__(input_size, hidden_channels, kernel_size, stride, paddings, out_dim, is_rec, is_adapt_conv,
                         syn_curr_conv, dp_rate, p_size, num_classes, pooling)

        # ff weights

        self.conv4 = SNNConvCell(hidden_channels[2], hidden_channels[3], kernel_size[3], stride[3], paddings[3],
                                 self.conv3.output_shape, is_adapt=is_adapt_conv, pooling_type=pooling,
                                 is_rec=is_rec[3])

        self.input_to_pc_sz = self.conv4.output_shape[0] * self.conv4.output_shape[1] * self.conv4.output_shape[2]

        self.conv_to_pop = nn.Linear(self.input_to_pc_sz, p_size * num_classes)
        nn.init.xavier_uniform_(self.conv_to_pop.weight)

        # feedback weights
        self.pop_to_conv = nn.Linear(p_size * num_classes, self.input_to_pc_sz)
        nn.init.xavier_uniform_(self.pop_to_conv.weight)

        self.deconv4 = nn.ConvTranspose2d(hidden_channels[3], hidden_channels[2], kernel_size=kernel_size[3],
                                          stride=stride[3], padding=paddings[3])

        self.neuron_count = self.conv1.output_size + self.h_layer.hidden_dim + self.pop_enc.hidden_dim + \
                            self.conv2.output_size + self.conv3.output_size + self.conv4.output_size

    def forward(self, x_t, h):
        batch_dim, c, height, width = x_t.size()
        x_t = self.dp(x_t)

        # hidden layer
        h_input = x_t + self.deconv1(h[5])
        mem_h, spk_h, curr_h, b_h = self.h_layer(h_input.view(batch_dim, -1), mem_t=h[0], spk_t=h[1], curr_t=h[2],
                                                 b_t=h[3])
        spk_h = spk_h.reshape(batch_dim, c, height, width)

        mem_conv1, spk_conv1, curr_conv1, b_conv1 = self.conv1(spk_h, mem_t=h[4], spk_t=h[5], curr_t=h[6], b_t=h[7],
                                                               top_down_sig=self.deconv2(h[9]))

        mem_conv2, spk_conv2, curr_conv2, b_conv2 = self.conv2(spk_conv1, mem_t=h[8], spk_t=h[9], curr_t=h[10],
                                                               b_t=h[11],
                                                               top_down_sig=self.deconv3(h[13]))

        mem_conv3, spk_conv3, curr_conv3, b_conv3 = self.conv3(spk_conv2, mem_t=h[12], spk_t=h[13], curr_t=h[14],
                                                               b_t=h[15],
                                                               top_down_sig=self.deconv4(h[17]))

        mem_conv4, spk_conv4, curr_conv4, b_conv4 = self.conv4(spk_conv3, mem_t=h[16], spk_t=h[17], curr_t=h[18],
                                                               b_t=h[19],
                                                               top_down_sig=self.pop_to_conv(h[21]))

        in_to_pop = self.conv_to_pop(spk_conv4.view(batch_dim, -1))
        # in_to_pc = spk2.view(batch_dim, -1)

        mem_pop, spk_pop, curr_pop, b_pop = self.pop_enc(in_to_pop, mem_t=h[20], spk_t=h[21], curr_t=h[22],
                                                         b_t=h[23])

        mem_out = self.output_layer(spk_pop, h[-1])

        h = (mem_h, spk_h.view(batch_dim, -1), curr_h, b_h,
             mem_conv1, spk_conv1, curr_conv1, b_conv1,
             mem_conv2, spk_conv2, curr_conv2, b_conv2,
             mem_conv3, spk_conv3, curr_conv3, b_conv3,
             mem_conv4, spk_conv4, curr_conv4, b_conv4,
             mem_pop, spk_pop, curr_pop, b_pop,
             mem_out
             )

        self.fr_conv1 = self.fr_conv1 + spk_conv1.detach().cpu().numpy().mean()
        self.fr_conv2 = self.fr_conv2 + spk_conv2.detach().cpu().numpy().mean()
        self.fr_h = self.fr_h + spk_h.detach().cpu().numpy().mean()
        self.fr_pop = self.fr_pop + spk_pop.detach().cpu().numpy().mean()

        log_softmax = F.log_softmax(mem_out, dim=1)

        return log_softmax, h

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        sz1 = self.conv1.output_shape  # torch size object
        sz2 = self.conv2.output_shape
        sz3 = self.conv3.output_shape
        sz4 = self.conv4.output_shape
        return (
            # h layer
            weight.new(bsz, self.h_layer.hidden_dim).uniform_(),
            weight.new(bsz, self.h_layer.hidden_dim).zero_(),
            weight.new(bsz, self.h_layer.hidden_dim).zero_(),
            weight.new(bsz, self.h_layer.hidden_dim).fill_(b_j0),
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
            # conv3
            weight.new(bsz, sz3[0], sz3[1], sz3[2]).uniform_(),
            weight.new(bsz, sz3[0], sz3[1], sz3[2]).zero_(),
            weight.new(bsz, sz3[0], sz3[1], sz3[2]).zero_(),
            weight.new(bsz, sz3[0], sz3[1], sz3[2]).fill_(b_j0),
            # conv4
            weight.new(bsz, sz4[0], sz4[1], sz4[2]).uniform_(),
            weight.new(bsz, sz4[0], sz4[1], sz4[2]).zero_(),
            weight.new(bsz, sz4[0], sz4[1], sz4[2]).zero_(),
            weight.new(bsz, sz4[0], sz4[1], sz4[2]).fill_(b_j0),
            # pop encode
            weight.new(bsz, self.pop_enc.hidden_dim).uniform_(),
            weight.new(bsz, self.pop_enc.hidden_dim).zero_(),
            weight.new(bsz, self.pop_enc.hidden_dim).zero_(),
            weight.new(bsz, self.pop_enc.hidden_dim).fill_(b_j0),
            # layer out
            weight.new(bsz, self.out_dim).zero_()
        )


class SnnConvNet5Layer(SnnConvNet4Layer):
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
            syn_curr_conv: bool,
            dp_rate=0.3,
            p_size=10,  # num prediction neurons
            num_classes=10,
            pooling=None
    ):
        super().__init__(input_size, hidden_channels, kernel_size, stride, paddings, out_dim, is_rec, is_adapt_conv,
                         syn_curr_conv, dp_rate, p_size, num_classes, pooling)

        # ff weights

        self.conv5 = SNNConvCell(hidden_channels[3], hidden_channels[4], kernel_size[4], stride[4], paddings[4],
                                 self.conv4.output_shape, is_adapt=is_adapt_conv, pooling_type=pooling,
                                 is_rec=is_rec[4])

        self.input_to_pc_sz = self.conv5.output_shape[0] * self.conv5.output_shape[1] * self.conv5.output_shape[2]

        self.conv_to_pop = nn.Linear(self.input_to_pc_sz, p_size * num_classes)
        nn.init.xavier_uniform_(self.conv_to_pop.weight)

        # feedback weights
        self.pop_to_conv = nn.Linear(p_size * num_classes, self.input_to_pc_sz)
        nn.init.xavier_uniform_(self.pop_to_conv.weight)

        self.deconv5 = nn.ConvTranspose2d(hidden_channels[4], hidden_channels[3], kernel_size=kernel_size[4],
                                          stride=stride[4], padding=paddings[4])

        self.neuron_count = self.conv1.output_size + self.h_layer.hidden_dim + self.pop_enc.hidden_dim + \
                            self.conv2.output_size + self.conv3.output_size + self.conv4.output_size + self.conv5.output_size

    def forward(self, x_t, h):
        batch_dim, c, height, width = x_t.size()
        x_t = self.dp(x_t)

        # hidden layer
        h_input = x_t + self.deconv1(h[5])
        mem_h, spk_h, curr_h, b_h = self.h_layer(h_input.view(batch_dim, -1), mem_t=h[0], spk_t=h[1], curr_t=h[2],
                                                 b_t=h[3])
        spk_h = spk_h.reshape(batch_dim, c, height, width)

        mem_conv1, spk_conv1, curr_conv1, b_conv1 = self.conv1(spk_h, mem_t=h[4], spk_t=h[5], curr_t=h[6], b_t=h[7],
                                                               top_down_sig=self.deconv2(h[9]))

        mem_conv2, spk_conv2, curr_conv2, b_conv2 = self.conv2(spk_conv1, mem_t=h[8], spk_t=h[9], curr_t=h[10],
                                                               b_t=h[11],
                                                               top_down_sig=self.deconv3(h[13]))

        mem_conv3, spk_conv3, curr_conv3, b_conv3 = self.conv3(spk_conv2, mem_t=h[12], spk_t=h[13], curr_t=h[14],
                                                               b_t=h[15],
                                                               top_down_sig=self.deconv4(h[17]))

        mem_conv4, spk_conv4, curr_conv4, b_conv4 = self.conv4(spk_conv3, mem_t=h[16], spk_t=h[17], curr_t=h[18],
                                                               b_t=h[19],
                                                               top_down_sig=self.deconv5(h[21]))

        mem_conv5, spk_conv5, curr_conv5, b_conv5 = self.conv5(spk_conv4, mem_t=h[20], spk_t=h[21], curr_t=h[22],
                                                               b_t=h[23],
                                                               top_down_sig=self.pop_to_conv(h[25]))

        in_to_pop = self.conv_to_pop(spk_conv5.view(batch_dim, -1))
        # in_to_pc = spk2.view(batch_dim, -1)

        mem_pop, spk_pop, curr_pop, b_pop = self.pop_enc(in_to_pop, mem_t=h[24], spk_t=h[25], curr_t=h[26],
                                                         b_t=h[27])

        mem_out = self.output_layer(spk_pop, h[-1])

        h = (mem_h, spk_h.view(batch_dim, -1), curr_h, b_h,
             mem_conv1, spk_conv1, curr_conv1, b_conv1,
             mem_conv2, spk_conv2, curr_conv2, b_conv2,
             mem_conv3, spk_conv3, curr_conv3, b_conv3,
             mem_conv4, spk_conv4, curr_conv4, b_conv4,
             mem_conv5, spk_conv5, curr_conv5, b_conv5,
             mem_pop, spk_pop, curr_pop, b_pop,
             mem_out
             )

        self.fr_conv1 = self.fr_conv1 + spk_conv1.detach().cpu().numpy().mean()
        self.fr_conv2 = self.fr_conv2 + spk_conv2.detach().cpu().numpy().mean()
        self.fr_h = self.fr_h + spk_h.detach().cpu().numpy().mean()
        self.fr_pop = self.fr_pop + spk_pop.detach().cpu().numpy().mean()

        log_softmax = F.log_softmax(mem_out, dim=1)

        return log_softmax, h

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        sz1 = self.conv1.output_shape  # torch size object
        sz2 = self.conv2.output_shape
        sz3 = self.conv3.output_shape
        sz4 = self.conv4.output_shape
        sz5 = self.conv5.output_shape

        return (
            # h layer
            weight.new(bsz, self.h_layer.hidden_dim).uniform_(),
            weight.new(bsz, self.h_layer.hidden_dim).zero_(),
            weight.new(bsz, self.h_layer.hidden_dim).zero_(),
            weight.new(bsz, self.h_layer.hidden_dim).fill_(b_j0),
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
            # conv3
            weight.new(bsz, sz3[0], sz3[1], sz3[2]).uniform_(),
            weight.new(bsz, sz3[0], sz3[1], sz3[2]).zero_(),
            weight.new(bsz, sz3[0], sz3[1], sz3[2]).zero_(),
            weight.new(bsz, sz3[0], sz3[1], sz3[2]).fill_(b_j0),
            # conv4
            weight.new(bsz, sz4[0], sz4[1], sz4[2]).uniform_(),
            weight.new(bsz, sz4[0], sz4[1], sz4[2]).zero_(),
            weight.new(bsz, sz4[0], sz4[1], sz4[2]).zero_(),
            weight.new(bsz, sz4[0], sz4[1], sz4[2]).fill_(b_j0),
            # conv5
            weight.new(bsz, sz5[0], sz5[1], sz5[2]).uniform_(),
            weight.new(bsz, sz5[0], sz5[1], sz5[2]).zero_(),
            weight.new(bsz, sz5[0], sz5[1], sz5[2]).zero_(),
            weight.new(bsz, sz5[0], sz5[1], sz5[2]).fill_(b_j0),
            # pop encode
            weight.new(bsz, self.pop_enc.hidden_dim).uniform_(),
            weight.new(bsz, self.pop_enc.hidden_dim).zero_(),
            weight.new(bsz, self.pop_enc.hidden_dim).zero_(),
            weight.new(bsz, self.pop_enc.hidden_dim).fill_(b_j0),
            # layer out
            weight.new(bsz, self.out_dim).zero_()
        )
# %%
