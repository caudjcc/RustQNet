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
from .hrnet import hrnet48, hrnet32, hrnet18

BatchNorm2d=nn.BatchNorm2d#分布式运行需要删除
CE = torch.nn.BCELoss(reduction='sum')
ALIGN_CORNERS = True
BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

class ModuleHelper:

    @staticmethod
    def BNReLU(num_features, bn_type=None, **kwargs):
        return nn.Sequential(
            BatchNorm2d(num_features, **kwargs),
            nn.ReLU()
        )

    @staticmethod
    def BatchNorm2d(*args, **kwargs):
        return BatchNorm2d

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
    
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class HRNet_multimodal(nn.Module):

    def __init__(self, config, **kwargs):
        global ALIGN_CORNERS
        super(HRNet_multimodal, self).__init__()
        ALIGN_CORNERS = config.MODEL.ALIGN_CORNERS
        last_inp_channels=config.MODEL.last_inp_channels

        self.conv_mul1 = BasicConv2d(10, 3, kernel_size=3, padding=1)
        self.conv_VI1 = BasicConv2d(13, 3, kernel_size=3, padding=1)


        phi = config.MODEL.PRETRAINED
        self.in_channels = {'hrnet48': [48,96,192,384], 'hrnet32': [32,64,128,256], 'hrnet18': [18, 36, 72, 144],}[phi]
        self.conv_concat = BasicConv2d(sum(self.in_channels) * 3, last_inp_channels, kernel_size=3, padding=1)

        self.hrnet_rgb = { 'hrnet48': hrnet48, 'hrnet32': hrnet32, 'hrnet18': hrnet18,}[phi](pretrained=True)
        self.hrnet_mul = {'hrnet48': hrnet48, 'hrnet32': hrnet32, 'hrnet18': hrnet18, }[phi](pretrained=True)
        self.hrnet_VI = {'hrnet48': hrnet48, 'hrnet32': hrnet32, 'hrnet18': hrnet18,}[phi](pretrained=True)

        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=int(last_inp_channels/4),
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm2d(int(last_inp_channels/4), momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace),
            nn.Conv2d(
                in_channels=int(last_inp_channels/4),
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1))

        self.cls_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels[0]*3,
                out_channels=32,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm2d(32, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace),
            nn.Conv2d(
                in_channels=32,
                out_channels=config.DATASET.NUM_CLASSES,
                kernel_size=3,
                stride=1,
                padding=1))

    
    def forward(self, x):
        RGB=x["RGB"]
        MUL=x["MUL"]
        VI=x["VI"]

        H, W = RGB.size(2), RGB.size(3)
        x1_rgb, x2_rgb, x3_rgb, x4_rgb=self.hrnet_rgb(RGB)
        #print(x1_rgb.shape, x2_rgb.shape, x3_rgb.shape, x4_rgb.shape)
        #HRNET18:  torch.Size([8, 18, 64, 64]) torch.Size([8, 36, 64, 64]) torch.Size([8, 72, 64, 64]) torch.Size([8, 144, 64, 64])
        #torch.Size([8, 32, 64, 64])        torch.Size([8, 64, 64, 64])        torch.Size([8, 128, 64, 64])        torch.Size([8, 256, 64, 64])

        MUL = self.conv_mul1(MUL)#10通道变3通道[4, 3, 192, 192]
        MUL = F.interpolate(MUL, size=(H, W), mode='bilinear', align_corners=ALIGN_CORNERS)
        x1_mul, x2_mul, x3_mul, x4_mul=self.hrnet_mul(MUL)

        VI = self.conv_VI1(VI)#13通道变3通道[4, 3, 192, 192]
        VI  = F.interpolate(VI, size=(H, W), mode='bilinear', align_corners=ALIGN_CORNERS)
        x1_VI, x2_VI, x3_VI, x4_VI=self.hrnet_VI(VI)

        low_level_features = torch.cat([x1_rgb, x1_mul, x1_VI], 1)
        cls_out = self.cls_layer(low_level_features)
        cls_out = F.interpolate(cls_out, size=(H, W), mode='bilinear', align_corners=ALIGN_CORNERS)

        high_level_features = torch.cat([x1_rgb, x2_rgb, x3_rgb, x4_rgb,x1_mul, x2_mul, x3_mul, x4_mul,x1_VI, x2_VI, x3_VI, x4_VI], 1)
        high_level_features =self.conv_concat(high_level_features)
        final_out = torch.sigmoid(self.last_layer(high_level_features))
        final_out = F.interpolate(final_out, size=(H, W), mode='bilinear', align_corners=ALIGN_CORNERS)
        return [cls_out, final_out]

    def init_weights(self, pretrained='',):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, BatchNorm2d_class):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            for k, _ in pretrained_dict.items():
                logger.info(
                    '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        else:
            pass

def get_seg_model(cfg, **kwargs):
    model = HRNet_multimodal(cfg, **kwargs)
    model.init_weights(cfg.MODEL.PRETRAINED)
    return model
