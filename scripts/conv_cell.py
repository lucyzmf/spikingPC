import torch.nn as nn
import torch


class SNN_Conv_cell(nn.Module):
    def init(self, input_size, output_dim, kernel_size=5, strides=1, padding=0,
             pooling_type=None, pool_size=2, pool_strides=2, bias=True):
        super(SNN_Conv_cell, self).init()

        print('SNN-conv +', pooling_type)
        self.input_size = input_size
        self.input_dim = input_size[0]
        self.output_dim = output_dim

        self.input_size = input_size

        self.rnn_name = 'SNN-conv cell'
        if pooling_type is not None:
            if pooling_type == 'max':
                self.pooling = nn.MaxPool2d(kernel_size=pool_size, stride=pool_strides, padding=0)
            elif pooling_type == 'avg':
                self.pooling = nn.AvgPool2d(kernel_size=pool_size, stride=pool_strides, padding=0)
            elif pooling_type == 'maxS1':
                self.pooling = MaxPoolStride1()
            elif pooling_type == 'up':
                self.pooling = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.pooling = None

        self.conv1_x = nn.Conv2d(self.input_dim, output_dim, kernel_size=kernel_size, stride=strides,
                                 padding=kernel_size // 2,
                                 bias=bias)
        self.conv_tauM = nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1)

        self.sig1 = nn.Sigmoid()

        self.BN = nn.BatchNorm2d(output_dim)
        self.BN1 = nn.BatchNorm2d(output_dim)

        self.output_size = self.compute_output_size()

        # nn.init.kaiming_normal_(self.conv1_x.weight)
        # nn.init.kaiming_normal_(self.conv_tauM.weight)
        # nn.init.kaiming_normal_(self.conv_tauAdp.weight)
        nn.init.xavier_normal_(self.conv1_x.weight)
        nn.init.xavier_normal_(self.conv_tauM.weight)
        # nn.init.xavier_normal_(self.conv_tauAdp.weight)
        if bias:
            nn.init.constant_(self.conv1_x.bias, 0)
            nn.init.constant_(self.conv_tauM.bias, 0)

    def forward(self, x_t, mem_t, spk_t, short_cut=None):
        conv_bnx = self.BN(self.conv1_x(x_t.float()))

        if short_cut is not None:
            conv_bnx = conv_bnx + short_cut

        if self.pooling is not None:
            conv_x = self.pooling(conv_bnx)
        else:
            conv_x = conv_bnx

        tauM1 = self.sig1(self.BN1(self.conv_tauM(conv_x + mem_t)))
        # tauAdp1 = self.sig3(self.BN2(self.conv_tauAdp(conv_x+b_t)))
        mem_1, spk_1 = mem_update_adp(conv_x, mem=mem_t, spike=spk_t,
                                      tau_m=tauM1)

        return mem_1, spk_1, conv_bnx

    def compute_output_size(self):
        x_emp = torch.randn([1, self.input_size[0], self.input_size[1], self.input_size[2]])
        out = self.conv1_x(x_emp)
        if self.pooling is not None:
            out = self.pooling(out)
        return out.shape[1:]
