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
from .MLPbackbone import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5

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


# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------

class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self, num_classes=20, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1):
        super(SegFormerHead, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            c1=embedding_dim * 4,
            c2=embedding_dim,
            k=1,
        )

        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_ratio)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x


class SegFormer(nn.Module):
    def __init__(self, config, num_classes=2, phi='b5', pretrained=True):
        super(SegFormer, self).__init__()
        phi = config.MODEL.PRETRAINED
        self.in_channels = {
            'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512],
            'b3': [64, 128, 320, 512], 'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
        }[phi]
        self.backbone_RGB = {
            'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
            'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,
        }[phi](pretrained)

        self.backbone_MUL = {
            'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
            'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,
        }[phi](pretrained)

        self.backbone_VI = {
            'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
            'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,
        }[phi](pretrained)

        self.feat_concat1 = BasicConv2d(self.in_channels[0]*3, self.in_channels[0], kernel_size=3, padding=1)
        self.feat_concat2 = BasicConv2d(self.in_channels[1]*3, self.in_channels[1], kernel_size=3, padding=1)
        self.feat_concat3 = BasicConv2d(self.in_channels[2]*3, self.in_channels[2], kernel_size=3, padding=1)
        self.feat_concat4 = BasicConv2d(self.in_channels[3]*3, self.in_channels[3], kernel_size=3, padding=1)

        self.conv_mul1 = BasicConv2d(10, 3, kernel_size=3, padding=1)
        self.conv_VI1 = BasicConv2d(13, 3, kernel_size=3, padding=1)

        self.cls_dim = {
            'b0': 32, 'b1': 64, 'b2': 64,
            'b3': 64, 'b4': 64, 'b5': 64,
        }[phi]
        self.classifier = nn.Sequential(
            nn.Conv2d(self.cls_dim, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 1)
        )

        self.embedding_dim = {
            'b0': 256, 'b1': 256, 'b2': 768,
            'b3': 768, 'b4': 768, 'b5': 768,
        }[phi]
        self.decode_head = SegFormerHead(1, self.in_channels, self.embedding_dim)

    def forward(self, x):
        RGB = x["RGB"]
        MUL = x["MUL"]
        VI = x["VI"]
        H_o, W_o = RGB.size(2), RGB.size(3)#
        MUL = self.conv_mul1(MUL)
        MUL = F.interpolate(MUL, size=(H_o, W_o), mode='bilinear', align_corners=False)
        VI = self.conv_VI1(VI)
        VI = F.interpolate(VI, size=(H_o, W_o), mode='bilinear', align_corners=False)


        [feat1_rgb, feat2_rgb, feat3_rgb, feat4_rgb] =self.backbone_RGB.forward(RGB)
        [feat1_mul, feat2_mul, feat3_mul, feat4_mul] = self.backbone_MUL.forward(MUL)
        [feat1_VI, feat2_VI, feat3_VI, feat4_VI] = self.backbone_VI.forward(VI)

        #print("feat1_rgb, feat2_rgb, feat3_rgb, feat4_rgb", feat1_rgb.shape, feat2_rgb.shape, feat3_rgb.shape, feat4_rgb.shape)
        feat1 = self.feat_concat1(torch.cat([feat1_rgb, feat1_mul, feat1_VI], dim=1))
        feat2 = self.feat_concat2(torch.cat([feat2_rgb, feat2_mul, feat2_VI], dim=1))
        feat3 = self.feat_concat3(torch.cat([feat3_rgb, feat3_mul, feat3_VI], dim=1))
        feat4 = self.feat_concat4(torch.cat([feat4_rgb, feat4_mul, feat4_VI], dim=1))

        cls_out = self.classifier(feat1)
        cls_out = F.interpolate(cls_out, size=(H_o, W_o), mode='bilinear', align_corners=True)

        x = self.decode_head.forward([feat1,feat2, feat3, feat4])
        x =  torch.sigmoid(F.interpolate(x, size=(H_o, W_o), mode='bilinear', align_corners=True))

        return cls_out, x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def get_seg_model(config, **kwargs):
    model = SegFormer(config,  **kwargs)
    model._init_weight()
    return model
