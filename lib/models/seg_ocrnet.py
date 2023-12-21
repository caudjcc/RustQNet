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


class SpatialGather_Module(nn.Module):
    """
        Aggregate the context features according to the initial 
        predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """
    def __init__(self, cls_num=0, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1) # batch x hw x c 
        probs = F.softmax(self.scale * probs, dim=2)# batch x k x hw
        ocr_context = torch.matmul(probs, feats)\
        .permute(0, 2, 1).unsqueeze(3)# batch x k x c
        return ocr_context


class _ObjectAttentionBlock(nn.Module):
    def __init__(self,
                 in_channels, 
                 key_channels, 
                 scale=1, 
                 bn_type=None):
        super(_ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.in_channels, bn_type=bn_type),
        )

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)   

        # add bg context ...
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=ALIGN_CORNERS)

        return context


class ObjectAttentionBlock2D(_ObjectAttentionBlock):
    def __init__(self, 
                 in_channels, 
                 key_channels, 
                 scale=1, 
                 bn_type=None):
        super(ObjectAttentionBlock2D, self).__init__(in_channels,
                                                     key_channels,
                                                     scale, 
                                                     bn_type=bn_type)


class SpatialOCR_Module(nn.Module):
    def __init__(self,
                 in_channels, 
                 key_channels, 
                 out_channels, 
                 scale=1, 
                 dropout=0.1, 
                 bn_type=None):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(in_channels, 
                                                           key_channels, 
                                                           scale, 
                                                           bn_type)
        _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            ModuleHelper.BNReLU(out_channels, bn_type=bn_type),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)

        output = self.conv_bn_dropout(torch.cat([context, feats], 1))

        return output
    
class Ocrnet_multimodal(nn.Module):

    def __init__(self, config, **kwargs):
        global ALIGN_CORNERS
        extra = config.MODEL.EXTRA
        super(Ocrnet_multimodal, self).__init__()       
        ALIGN_CORNERS = config.MODEL.ALIGN_CORNERS
        
        last_inp_channels=config.MODEL.last_inp_channels
        ocr_mid_channels = config.MODEL.OCR.MID_CHANNELS
        ocr_key_channels = config.MODEL.OCR.KEY_CHANNELS


        self.conv_mul1 = BasicConv2d(10, 3, kernel_size=3, padding=1)
        self.conv_VI1 = BasicConv2d(13, 3, kernel_size=3, padding=1)


        phi = config.MODEL.PRETRAINED
        self.in_channels = {'hrnet48': [48,96,192,384], 'hrnet32': [32,64,128,256], 'hrnet18': [18, 36, 72, 144],}[phi]
        self.conv_concat = BasicConv2d(sum(self.in_channels) * 3, last_inp_channels, kernel_size=3, padding=1)

        self.hrnet_rgb = { 'hrnet48': hrnet48, 'hrnet32': hrnet32, 'hrnet18': hrnet18,}[phi](pretrained=True)
        self.hrnet_mul = {'hrnet48': hrnet48, 'hrnet32': hrnet32, 'hrnet18': hrnet18, }[phi](pretrained=True)
        self.hrnet_VI = {'hrnet48': hrnet48, 'hrnet32': hrnet32, 'hrnet18': hrnet18,}[phi](pretrained=True)
        

        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(last_inp_channels, ocr_mid_channels,
                      kernel_size=3, stride=1, padding=1),
            BatchNorm2d(ocr_mid_channels),
            nn.ReLU(inplace=relu_inplace),
        )
        
        self.ocr_gather_head = SpatialGather_Module(config.DATASET.NUM_CLASSES)

        self.ocr_distri_head = SpatialOCR_Module(in_channels=ocr_mid_channels,
                                                 key_channels=ocr_key_channels,
                                                 out_channels=ocr_mid_channels,
                                                 scale=1,
                                                 dropout=0.05,
                                                 )
        self.cls_head = nn.Conv2d(
            ocr_mid_channels, 1, kernel_size=1, stride=1, padding=0, bias=True)

        self.aux_head = nn.Sequential(
            nn.Conv2d(last_inp_channels, ocr_mid_channels,
                      kernel_size=1, stride=1, padding=0),
            BatchNorm2d(ocr_mid_channels),
            nn.ReLU(inplace=relu_inplace),
            nn.Conv2d(ocr_mid_channels, config.DATASET.NUM_CLASSES,
                      kernel_size=1, stride=1, padding=0, bias=True)
        )

    
    
    def forward(self, x):
        RGB=x["RGB"]
        MUL=x["MUL"]
        VI=x["VI"]
        H, W = RGB.size(2), RGB.size(3)
        x1_rgb, x2_rgb, x3_rgb, x4_rgb=self.hrnet_rgb(RGB)

        MUL = self.conv_mul1(MUL)#10通道变3通道[4, 3, 192, 192]
        MUL = F.interpolate(MUL, size=(H, W), mode='bilinear', align_corners=ALIGN_CORNERS)
        x1_mul, x2_mul, x3_mul, x4_mul=self.hrnet_mul(MUL)

        VI = self.conv_VI1(VI)#13通道变3通道[4, 3, 192, 192]
        VI  = F.interpolate(VI, size=(H, W), mode='bilinear', align_corners=ALIGN_CORNERS)
        x1_VI, x2_VI, x3_VI, x4_VI=self.hrnet_VI(VI)

        high_level_features = torch.cat([x1_rgb, x2_rgb, x3_rgb, x4_rgb,x1_mul, x2_mul, x3_mul, x4_mul,x1_VI, x2_VI, x3_VI, x4_VI], 1)
        high_level_features =self.conv_concat(high_level_features)

        out_aux = self.aux_head(high_level_features)
        feats = self.conv3x3_ocr(high_level_features)
        # compute contrast feature
        context = self.ocr_gather_head(feats, out_aux)
        feats = self.ocr_distri_head(feats, context)
        out = torch.sigmoid(self.cls_head(feats))

        out_aux_seg = []
        out_aux_seg.append(F.interpolate(out_aux, size=(H, W), mode='bilinear', align_corners=ALIGN_CORNERS))
        out_aux_seg.append(F.interpolate(out, size=(H, W), mode='bilinear', align_corners=ALIGN_CORNERS))

        return out_aux_seg

    def init_weights(self, pretrained='',):
        logger.info('=> init weights from normal distribution')
        for name, m in self.named_modules():
            if any(part in name for part in {'cls', 'aux', 'ocr'}):
                print('skipped', name)
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, BatchNorm2d_class):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained, map_location={'cuda:0': 'cpu'})
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k.replace('last_layer', 'aux_head').replace('model.', ''): v for k, v in pretrained_dict.items()}  
            #keys_to_skip = {'cls_head', 'aux_head'}
            #pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in keys_to_skip}
            pretrained_dict = {k: v for k, v in pretrained_dict.items()  if k in model_dict.keys() and v.shape == model_dict[k].shape}
            for k, _ in pretrained_dict.items():
                logger.info(
                     '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        else:
            pass


def get_seg_model(cfg, **kwargs):
    model = Ocrnet_multimodal(cfg, **kwargs)
    model.init_weights(cfg.MODEL.PRETRAINED_DIR)
    return model
