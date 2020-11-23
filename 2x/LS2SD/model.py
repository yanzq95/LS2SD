import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from all_attention import PAM_and_CAM

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

def convt_bn_relu(in_channels, out_channels, kernel_size, \
        stride=1, padding=0, output_padding=0, bn=True, relu=True):
    bias = not bn
    layers = []
    layers.append(
        nn.ConvTranspose2d(in_channels,
                           out_channels,
                           kernel_size,
                           stride,
                           padding,
                           output_padding,
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

class field_transform(nn.Module):
    def __init__(self, c, h, w):
        super(field_transform, self).__init__()

        self.c = c
        self.h = h
        self.w = w

        self.conv_for_dc_add_sr = conv_bn_relu(c, c, 1)
        self.att = PAM_and_CAM(c, h, w)

        self.dc_alpha = nn.Parameter(torch.zeros(1))
        self.sr_beta = nn.Parameter(torch.zeros(1))

    def forward(self, dc_feature, sr_feature):

        dc = dc_feature
        sr = sr_feature
        dc_add_sr = dc_feature + sr_feature
        out = self.conv_for_dc_add_sr(dc_add_sr)
        att = self.att(out)
        dc_out = dc + self.dc_alpha * att
        sr_out = sr + self.sr_beta * att

        return dc_out, sr_out

class DepthCompletionNet(nn.Module):
    def __init__(self, args):
        assert (
            args.layers in [18, 34, 50, 101, 152]
        ), 'Only layers 18, 34, 50, 101, and 152 are defined, but got {}'.format(args.layers)
        super(DepthCompletionNet, self).__init__()
        self.modality = args.input


        channels = 16
        self.conv1_d = conv_bn_relu(1,
                                    # channels,
                                    16,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)

        channels = 48
        self.conv1_img = conv_bn_relu(3,
                                      # channels,
                                      64,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1)
        self.conv1_img2 = conv_bn_relu(64,
                                      # channels,
                                      64,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0)

        self.conv2_img = conv_bn_relu(64,
                                      # channels,
                                      64,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0)
        self.conv2_img2 = conv_bn_relu(64,
                                      # channels,
                                      48,
                                      kernel_size=3,
                                      stride=2,
                                      padding=1)
        self.conv3_img = conv_bn_relu(64,
                                      # channels,
                                      48,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0)
        self.conv3_img2 = conv_bn_relu(48,
                                      # channels,
                                      48,
                                      kernel_size=3,
                                      stride=2,
                                      padding=1)

        pretrained_model = resnet.__dict__['resnet{}'.format(
            args.layers)](pretrained=args.pretrained)
        if not args.pretrained:
            pretrained_model.apply(init_weights)
        #self.maxpool = pretrained_model._modules['maxpool']
        self.conv2 = pretrained_model._modules['layer1']
        self.conv3 = pretrained_model._modules['layer2']
        self.conv4 = pretrained_model._modules['layer3']
        self.conv5 = pretrained_model._modules['layer4']
        del pretrained_model  # clear memory

        # define number of intermediate channels
        num_channels = 512


        # decoding layers
        kernel_size = 3
        stride = 2

        self.convt4 = convt_bn_relu(in_channels=512,
                                    out_channels=256,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=1,
                                    output_padding=1)
        self.srconvt4 = convt_bn_relu(512, 256, kernel_size, stride, 1, 1)

        self.convt3 = convt_bn_relu(in_channels=256,
                                    out_channels=128,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=1,
                                    output_padding=1)
        self.srconvt3 = convt_bn_relu(256, 128, kernel_size, stride, 1, 1)

        self.convt2 = convt_bn_relu(in_channels=128,
                                    out_channels=64,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=1,
                                    output_padding=1)
        self.srconvt2 = convt_bn_relu(128, 64, kernel_size, stride, 1, 1)

        self.convt1 = convt_bn_relu(in_channels=64,
                                    out_channels=64,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=1)
        self.srconvt1 = convt_bn_relu(64, 64, kernel_size, 1, 1)

        self.convt2x = convt_bn_relu(in_channels=64,
                                    out_channels=64,
                                    kernel_size=kernel_size,
                                    stride=2,
                                    padding=1,
                                    output_padding=1)
        self.srconvt2x = convt_bn_relu(112, 64, kernel_size, 2, 1, 1)

        self.convt4x = convt_bn_relu(in_channels=128,
                                    out_channels=64,
                                    kernel_size=kernel_size,
                                    stride=2,
                                    padding=1,
                                    output_padding=1)
        self.srconvt4x = convt_bn_relu(128, 64, kernel_size, 2, 1, 1)

        self.DC_2x_to_1x = convt_bn_relu(64, 64, 3, 2, 1, 1)
        self.SR_1x_to_2x = conv_bn_relu(64, 64, 3, 2, 1)

        self.SR = conv_bn_relu(in_channels=128,
                                   out_channels=64,
                                   kernel_size=1,
                                   stride=1,
                                   bn=True,
                                   relu=True)

        self.DC = conv_bn_relu(in_channels=64,
                                   out_channels=64,
                                   kernel_size=1,
                                   stride=1,
                                   bn=True,
                                   relu=True)

        self.DC_out = conv_bn_relu(in_channels=128,
                                   out_channels=1,
                                   kernel_size=1,
                                   stride=1,
                                   bn=False,
                                   relu=False)
        self.SR_out = conv_bn_relu(in_channels=128,
                                   out_channels=1,
                                   kernel_size=1,
                                   stride=1,
                                   bn=False,
                                   relu=False)
        self.dc_and_sr_field_transform_4 = field_transform(256, 16, 76)
        self.dc_and_sr_field_transform_3 = field_transform(128, 64, 304)
        self.dc_and_sr_field_transform_2 = field_transform(64, 128, 608)
        self.dc_and_sr_field_transform_1 = field_transform(64, 128, 608)

    def forward(self, z):
        # first layer
        sparse = z['d']
        rgb = z['rgb']

        sparse = F.max_pool2d(sparse, 2, 2) # B, 1, 128, 608
        # rgb = F.max_pool2d(rgb, 4, 4)

        # sparse encoding ############################################################
        conv1_d = self.conv1_d(sparse) # B, 16, 128, 608

        # rgb encoding with cnn ######################################################
        conv1_img = self.conv1_img(rgb)       # 1x
        conv1_img = self.conv1_img2(conv1_img) # B, 64, 256, 1216

        conv2_img = self.conv2_img(conv1_img) # 2x
        conv2_img = self.conv2_img2(conv2_img) # B, 48, 128, 608



        # encoder net ################################################################
        conv1 = torch.cat((conv1_d, conv2_img), 1)  # B, 64, 128, 608
        conv2 = self.conv2(conv1)  # B, 64, 128, 608
        conv3 = self.conv3(conv2)  # B, 128, 64, 304
        conv4 = self.conv4(conv3)  # B, 256, 32, 152
        conv5 = self.conv5(conv4)  # B, 512, 16, 76



        # DC and SR decoder ##################################################################
        # 4
        convt4 = self.convt4(conv5)  # B, 256, 32, 152
        y = convt4 + 0.5 * conv4

        sr_convt4 = self.srconvt4(conv5)  # B, 256, 32, 152
        x = sr_convt4 + 0.5 * conv4

        # field transform 4
        # y, x = self.dc_and_sr_field_transform_4(y, x)
        # 4 end

        #************************************************#

        # 3
        convt3 = self.convt3(y)  # B, 128, 64, 304
        y = convt3 + 0.5 * conv3

        sr_convt3 = self.srconvt3(x)  # B, 128, 64, 304
        x = sr_convt3 + 0.5 * conv3

        # field transform 3
        # y, x = self.dc_and_sr_field_transform_3(y, x)
        # 3 end

        # ************************************************#

        # 2
        convt2 = self.convt2(y)  # B, 64, 128, 608
        y = convt2 + 0.5 * conv2

        sr_convt2 = self.srconvt2(x)  # B, 64, 128, 608
        x = sr_convt2 + 0.5 * conv2

        # field transform 2
        y, x = self.dc_and_sr_field_transform_2(y, x)
        # 2 end

        # ************************************************#

        # 1
        convt1 = self.convt1(y)   # B, 64, 128, 608
        y = convt1 + 0.5 * conv1

        sr_convt1 = self.srconvt1(x)  # B, 64, 64, 304
        x = sr_convt1 + 0.5 * conv1

        # field transform 1
        y, x = self.dc_and_sr_field_transform_1(y, x)
        # 1 end

        # ************************************************#

        DC_2x = self.DC(y)  # B, 64, 128, 608

        DC_1x = self.DC_2x_to_1x(DC_2x)   # B, 64, 256, 1216

        x = torch.cat((x, conv2_img), 1)
        x = self.srconvt2x(x)  # B, 64, 128, 608

        x = torch.cat((x, conv1_img), 1)

        x = self.SR(x) # B, 64, 256, 1216

        SR_2x = self.SR_1x_to_2x(x)

        SR_cat_DC = torch.cat((x, DC_1x), 1)
        DC_cat_SR = torch.cat((DC_2x, SR_2x), 1)

        DC_out = self.DC_out(DC_cat_SR)
        SR_out = self.SR_out(SR_cat_DC)

        if self.training:
            return 100 * DC_out, 100 * SR_out
        else:
            min_distance = 0.9
            return F.relu(100 * DC_out - min_distance) + min_distance, F.relu(100 * SR_out - min_distance) + min_distance