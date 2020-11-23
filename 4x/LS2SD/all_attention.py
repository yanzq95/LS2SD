import torch
import torch.nn as nn
from torch.nn import Module, Sequential, Conv2d, ReLU, AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F



__all__ = ['PAM_Module', 'CAM_Module']


class PAM_Module(Module):

    def __init__(self, in_dim, height, width, kernel_narrow=1):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim
        # self.kernel_narrow = kernel_narrow

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.query_pool = torch.nn.AvgPool2d(kernel_size=(kernel_narrow, width), stride=1, padding=((kernel_narrow-1)//2, 0))
        self.key_pool = torch.nn.AvgPool2d(kernel_size=(height, kernel_narrow), stride=1, padding=(0, (kernel_narrow-1)//2))

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):

        m_batchsize, C, height, width = x.size()# 3 64 256 1216
        proj_query = self.query_conv(x)
        proj_query_pool = self.query_pool(proj_query)# 3 * 64 * 256 * 1

        proj_key = self.key_conv(x)
        proj_key_pool = self.key_pool(proj_key)

        energy = proj_query_pool @ proj_key_pool
        energy_reshape = energy.view(m_batchsize, C, height * width)
        attention = self.softmax(energy_reshape)
        attention_reshape = attention.view(m_batchsize, C, height, width)

        proj_value = self.value_conv(x)
        out = proj_value * attention_reshape
        out = self.gamma*out + x

        return out


class CAM_Module(Module):

    def __init__(self, in_dim, height=None, width=None):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.query_pool = torch.nn.AdaptiveAvgPool2d(1)

        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_pool = torch.nn.AdaptiveAvgPool2d(1)

        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):

        m_batchsize, C, height, width = x.size()

        proj_query = self.query_conv(x)
        proj_query_pool = self.query_pool(proj_query)
        proj_query_pool_reshape = proj_query_pool.view(m_batchsize, C, -1)

        proj_key = self.key_conv(x)
        proj_key_pool = self.key_pool(proj_key)
        proj_key_pool_reshape = proj_key_pool.view(m_batchsize, C, -1).permute(0, 2, 1)

        energy = torch.bmm(proj_query_pool_reshape, proj_key_pool_reshape)
        energy_reshape = energy.view(m_batchsize, -1)

        attention = self.softmax(energy_reshape)
        attention_reshape = attention.view(m_batchsize, C, C)

        proj_value = self.value_conv(x)
        proj_value_reshape = proj_value.view(m_batchsize, C, -1)

        out = torch.bmm(attention_reshape, proj_value_reshape)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma*out + x

        return out



class PAM_and_CAM(Module):
    def __init__(self, in_dim, height, width):
        super(PAM_and_CAM, self).__init__()
        self.chanel_in = in_dim
        self.pa = PAM_Module(in_dim, height, width)
        self.ca = CAM_Module(in_dim, height, width)
        self.conv_pa = conv_bn_relu(in_dim, in_dim, 3, 1, 1, bn=False, relu=False)
        self.conv_ca = conv_bn_relu(in_dim, in_dim, 3, 1, 1, bn=False, relu=False)

    def forward(self, x):

        pa_att = self.pa(x)
        pa_learning = self.conv_pa(pa_att)

        ca_att = self.ca(x)
        ca_learning = self.conv_ca(ca_att)

        pa_sum_ca = pa_learning + ca_learning

        return pa_sum_ca


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 1e-3)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(0, 1e-3)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def conv_bn_relu(in_channels, out_channels, kernel_size, \
        stride=1, padding=0, bn=True, relu=True):
    bias = not bn
    layers = []
    layers.append(
        nn.Conv2d(in_channels,
                  out_channels,
                  kernel_size,
                  stride,
                  padding,
                  bias=bias))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    layers = nn.Sequential(*layers)

    # initialize the weights
    for m in layers.modules():
        init_weights(m)

    return layers