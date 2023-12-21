# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn), Jingyi Xie (hsfzxjy@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import functools
from torch.distributions import Normal, Independent, kl
import numpy as np
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
from .bn_helper import BatchNorm2d, BatchNorm2d_class, relu_inplace
from .ResNet_ import resnet18, resnet50, resnet101, resnext50_32x4d, resnext101_32x8d

BatchNorm2d=nn.BatchNorm2d#分布式运行需要删除
ALIGN_CORNERS = True
BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class ResNet_Backbone(nn.Module):
    def __init__(self, config, pretrained=True):
        super(ResNet_Backbone, self).__init__()
        from functools import partial
        phi = config.MODEL.PRETRAINED
        model =  { 'resnet18': resnet18,  'resnet50': resnet50, 'resnet101': resnet101,
            'resnext50_32x4d': resnext50_32x4d, 'resnext101_32x8d': resnext101_32x8d, }[phi](pretrained)
        #model = resnet50(pretrained)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
    def forward(self, x):
        x       = self.conv1(x)
        x       = self.bn1(x)
        feat1   = self.relu(x)
        x       = self.maxpool(feat1)
        feat2   = self.layer1(x)
        feat3   = self.layer2(feat2)
        feat4   = self.layer3(feat3)
        feat5   = self.layer4(feat4)
        return [feat1, feat2, feat3, feat4, feat5]


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs


class Unet(nn.Module):
    def __init__(self, config, num_classes=21, pretrained=True,):
        super(Unet, self).__init__()
        phi = config.MODEL.PRETRAINED
        self.conv_mul1 = BasicConv2d(10, 3, kernel_size=3, padding=1)
        self.conv_VI1 = BasicConv2d(13, 3, kernel_size=3, padding=1)
        self.resnet_mul = ResNet_Backbone(config, pretrained=pretrained)
        self.resnet_rgb = ResNet_Backbone(config, pretrained=pretrained)
        self.resnet_VI = ResNet_Backbone(config, pretrained=pretrained)
        in_filters = [192, 512, 1024, 3072]
        out_filters = [64, 128, 256, 512]

        channel =  { 'resnet18': [64, 64, 128, 256, 512],  'resnet50':  [64, 256, 512, 1024, 2048], 'resnet101': [64, 256, 512, 1024, 2048],
            'resnext50_32x4d': [64, 256, 512, 1024, 2048], 'resnext101_32x8d': [64, 256, 512, 1024, 2048], }[phi]

        in_filters =  { 'resnet18': [128+64, 256+64, 512+128, 768],  'resnet50':  [192, 512, 1024, 3072], 'resnet101': [192, 512, 1024, 3072],
            'resnext50_32x4d': [192, 512, 1024, 3072], 'resnext101_32x8d': [192, 512, 1024, 3072], }[phi]

        self.feat_concat1 = BasicConv2d(channel[0]*3, channel[0], kernel_size=3, padding=1)
        self.feat_concat2 = BasicConv2d(channel[1]*3, channel[1], kernel_size=3, padding=1)
        self.feat_concat3 = BasicConv2d(channel[2]*3, channel[2], kernel_size=3, padding=1)
        self.feat_concat4 = BasicConv2d(channel[3]*3, channel[3], kernel_size=3, padding=1)
        self.feat_concat5 = BasicConv2d(channel[4]*3, channel[4], kernel_size=3, padding=1)
        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])


        self.up_conv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
            nn.ReLU(), )

        self.final = nn.Conv2d(out_filters[0], 1, 1)

        self.classifier = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 1)
        )

    def forward(self, x):
        RGB = x["RGB"]
        MUL = x["MUL"]
        VI = x["VI"]
        H_o, W_o = RGB.size(2), RGB.size(3)#
        MUL = self.conv_mul1(MUL)
        MUL = F.interpolate(MUL, size=(H_o, W_o), mode='bilinear', align_corners=False)
        VI = self.conv_VI1(VI)
        VI = F.interpolate(VI, size=(H_o, W_o), mode='bilinear', align_corners=False)

        [feat1_rgb, feat2_rgb, feat3_rgb, feat4_rgb, feat5_rgb] = self.resnet_rgb.forward(RGB)
        [feat1_mul, feat2_mul, feat3_mul, feat4_mul, feat5_mul] = self.resnet_mul.forward(MUL)
        [feat1_VI, feat2_VI, feat3_VI, feat4_VI, feat5_VI] = self.resnet_VI.forward(VI)
        #torch.Size([16, 64, 128, 128]) torch.Size([16, 256, 64, 64]) torch.Size([16, 512, 32, 32]) torch.Size([16, 1024, 16, 16]) torch.Size([16, 2048, 8, 8])
        #print(feat1_rgb.shape, feat2_rgb.shape, feat3_rgb.shape, feat4_rgb.shape, feat5_rgb.shape)
        feat1 = self.feat_concat1(torch.cat([feat1_rgb, feat1_mul, feat1_VI], dim=1))
        feat2 = self.feat_concat2(torch.cat([feat2_rgb, feat2_mul, feat2_VI], dim=1))
        feat3 = self.feat_concat3(torch.cat([feat3_rgb, feat3_mul, feat3_VI], dim=1))
        feat4 = self.feat_concat4(torch.cat([feat4_rgb, feat4_mul, feat4_VI], dim=1))
        feat5 = self.feat_concat5(torch.cat([feat5_rgb, feat5_mul, feat5_VI], dim=1))
        #print(feat1.shape, feat2.shape, feat3.shape, feat4.shape, feat5.shape, )
        cls_out = self.classifier(feat1)
        cls_out = F.interpolate(cls_out, size=(H_o, W_o), mode='bilinear', align_corners=False)

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)
        up1 = self.up_conv(up1)
        final = self.final(up1)
        final = torch.sigmoid(F.interpolate(final, size=(H_o, W_o), mode='bilinear', align_corners=False))
        return cls_out, final

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def get_seg_model(config, **kwargs):
    model = Unet(config, **kwargs)
    model._init_weight()
    return model
