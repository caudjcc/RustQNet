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
from .seg_ocrnet import ModuleHelper, BasicConv2d, conv3x3, SpatialGather_Module, _ObjectAttentionBlock, ObjectAttentionBlock2D, SpatialOCR_Module
from .hrnet import hrnet48 as HRnet_Backbone 

BatchNorm2d=nn.BatchNorm2d#分布式运行需要删除
CE = torch.nn.BCELoss(reduction='sum')
ALIGN_CORNERS = True
BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

    
class Mutual_info_reg(nn.Module):
    def __init__(self, input_channels, channels, latent_size, size = 64):
        super(Mutual_info_reg, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = BatchNorm2d(channels)
        self.layer2 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        self.bn2 = BatchNorm2d(channels)
        self.layer3 = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

        self.channel = channels      
        self.size = size
        self.fc1_rgb1 = nn.Linear(channels * 1 * self.size * self.size, latent_size)
        self.fc2_rgb1 = nn.Linear(channels * 1 * self.size * self.size, latent_size)
        self.fc1_depth1 = nn.Linear(channels * 1 * self.size * self.size, latent_size)
        self.fc2_depth1 = nn.Linear(channels * 1 * self.size * self.size, latent_size)

        self.leakyrelu = nn.LeakyReLU() 
        self.tanh = torch.nn.Tanh()

    def kl_divergence(self, posterior_latent_space, prior_latent_space):
        kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
        return kl_div

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, rgb_feat, depth_feat):
        rgb_feat = self.layer3(self.leakyrelu(self.bn1(self.layer1(rgb_feat))))#[4, 64, 12, 12]
        depth_feat = self.layer4(self.leakyrelu(self.bn2(self.layer2(depth_feat))))#[4, 64, 12, 12]


        rgb_feat = rgb_feat.view(-1, self.channel * 1 * self.size * self.size)#[4, 9216]
        depth_feat = depth_feat.view(-1, self.channel * 1 * self.size * self.size)#[4, 9216]
        mu_rgb = self.fc1_rgb1(rgb_feat)
        logvar_rgb = self.fc2_rgb1(rgb_feat)
        mu_depth = self.fc1_depth1(depth_feat)
        logvar_depth = self.fc2_depth1(depth_feat)

        mu_depth = self.tanh(mu_depth)
        mu_rgb = self.tanh(mu_rgb)
        logvar_depth = self.tanh(logvar_depth)
        logvar_rgb = self.tanh(logvar_rgb)
        z_rgb = self.reparametrize(mu_rgb, logvar_rgb)
        dist_rgb = Independent(Normal(loc=mu_rgb, scale=torch.exp(logvar_rgb)), 1)
        z_depth = self.reparametrize(mu_depth, logvar_depth)
        dist_depth = Independent(Normal(loc=mu_depth, scale=torch.exp(logvar_depth)), 1)
        bi_di_kld = torch.mean(self.kl_divergence(dist_rgb, dist_depth)) + torch.mean(
            self.kl_divergence(dist_depth, dist_rgb))
        z_rgb_norm = torch.sigmoid(z_rgb)
        z_depth_norm = torch.sigmoid(z_depth)
        ce_rgb_depth = CE(z_rgb_norm,z_depth_norm.detach())
        ce_depth_rgb = CE(z_depth_norm, z_rgb_norm.detach())
        latent_loss = ce_rgb_depth+ce_depth_rgb-bi_di_kld
        # latent_loss = torch.abs(cos_sim(z_rgb,z_depth)).sum()
        return latent_loss
    


class RustQNet(nn.Module):

    def __init__(self, config, **kwargs):
        global ALIGN_CORNERS
        extra = config.MODEL.EXTRA
        super(RustQNet, self).__init__()       
        ALIGN_CORNERS = config.MODEL.ALIGN_CORNERS
        
        last_inp_channels=config.MODEL.last_inp_channels
        ocr_mid_channels = config.MODEL.OCR.MID_CHANNELS
        ocr_key_channels = config.MODEL.OCR.KEY_CHANNELS
        channel=config.MODEL.MI_channel
        
        self.MI_cul=config.MODEL.MI_cul
        self.RGB_modal = config.DATASET.RGB_modal
        self.MUL_modal = config.DATASET.MUL_modal
        self.VI_modal = config.DATASET.VI_modal
        
        self.count_true = [config.DATASET.RGB_modal, config.DATASET.MUL_modal, config.DATASET.VI_modal].count(True)
        if config.DATASET.RGB_modal:
            self.hrnet_rgb = HRnet_Backbone()
            if self.count_true>1 and self.MI_cul:
                self.convx1_rgb = nn.Conv2d(in_channels=48, out_channels=channel, kernel_size=3, padding=1)        
                self.convx2_rgb = nn.Conv2d(in_channels=96, out_channels=channel, kernel_size=3, padding=1)
                self.convx3_rgb = nn.Conv2d(in_channels=192, out_channels=channel, kernel_size=3, padding=1)
                self.convx4_rgb = nn.Conv2d(in_channels=384, out_channels=channel, kernel_size=3, padding=1)
             
        if config.DATASET.MUL_modal:
            self.upsample2_MUL = nn.ConvTranspose2d(3, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.conv_mul1 = BasicConv2d(10, 3, kernel_size=3, padding=1)
            self.hrnet_mul = HRnet_Backbone()
            if self.count_true>1 and self.MI_cul:
                self.convx1_mul = nn.Conv2d(in_channels=48, out_channels=channel, kernel_size=3, padding=1)        
                self.convx2_mul = nn.Conv2d(in_channels=96, out_channels=channel, kernel_size=3, padding=1)
                self.convx3_mul = nn.Conv2d(in_channels=192, out_channels=channel, kernel_size=3, padding=1)
                self.convx4_mul = nn.Conv2d(in_channels=384, out_channels=channel, kernel_size=3, padding=1)
        if config.DATASET.VI_modal:
            self.upsample2_VI = nn.ConvTranspose2d(3, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.conv_VI1 = BasicConv2d(13, 3, kernel_size=3, padding=1)
            self.hrnet_VI = HRnet_Backbone()
            if self.count_true>1 and self.MI_cul:
                self.convx1_vi = nn.Conv2d(in_channels=48, out_channels=channel, kernel_size=3, padding=1)        
                self.convx2_vi = nn.Conv2d(in_channels=96, out_channels=channel, kernel_size=3, padding=1)
                self.convx3_vi = nn.Conv2d(in_channels=192, out_channels=channel, kernel_size=3, padding=1)
                self.convx4_vi = nn.Conv2d(in_channels=384, out_channels=channel, kernel_size=3, padding=1)


        self.latent_dim = 6

        self.size= config.TRAIN.IMAGE_SIZE[0]//16
        if self.count_true == 1:
            self.conv_concat = BasicConv2d(720, last_inp_channels, kernel_size=3, padding=1)
        if self.count_true == 2:
            self.conv_concat = BasicConv2d(1440, last_inp_channels, kernel_size=3, padding=1)
            if self.MI_cul:
                self.mi_level1 = Mutual_info_reg(channel, channel, self.latent_dim,  self.size)
                self.mi_level2 = Mutual_info_reg(channel, channel, self.latent_dim,  self.size)
                self.mi_level3 = Mutual_info_reg(channel, channel, self.latent_dim,  self.size)
                self.mi_level4 = Mutual_info_reg(channel, channel, self.latent_dim,  self.size)
        if self.count_true == 3:
            self.conv_concat = BasicConv2d(2160, last_inp_channels, kernel_size=3, padding=1)
            if self.MI_cul:
                self.mi_level1 = Mutual_info_reg(channel, channel, self.latent_dim,  self.size)
                self.mi_level2 = Mutual_info_reg(channel, channel, self.latent_dim,  self.size)
                self.mi_level3 = Mutual_info_reg(channel, channel, self.latent_dim,  self.size)
                self.mi_level4 = Mutual_info_reg(channel, channel, self.latent_dim,  self.size)
                
                self.mi_level5 = Mutual_info_reg(channel, channel, self.latent_dim,  self.size)
                self.mi_level6 = Mutual_info_reg(channel, channel, self.latent_dim,  self.size)
                self.mi_level7 = Mutual_info_reg(channel, channel, self.latent_dim,  self.size)
                self.mi_level8 = Mutual_info_reg(channel, channel, self.latent_dim,  self.size)
    
                self.mi_level9 = Mutual_info_reg(channel, channel, self.latent_dim,  self.size)
                self.mi_level10 = Mutual_info_reg(channel, channel, self.latent_dim,  self.size)
                self.mi_level11 = Mutual_info_reg(channel, channel, self.latent_dim,  self.size)
                self.mi_level12 = Mutual_info_reg(channel, channel, self.latent_dim,  self.size) 
                
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
        feature=[]
        if RGB!=None and self.RGB_modal:
            H, W = RGB.size(2), RGB.size(3)
            x1_rgb, x2_rgb, x3_rgb, x4_rgb=self.hrnet_rgb(RGB)
            feature.append(x1_rgb)
            feature.append(x2_rgb)
            feature.append(x3_rgb)
            feature.append(x4_rgb)
                
        if MUL!=None and self.MUL_modal:
            H, W =  2*MUL.size(2),  2*MUL.size(3)
            MUL = self.conv_mul1(MUL)#10通道变3通道[4, 3, 192, 192]
            #MUL=self.upsample2_MUL(MUL)#
            #print("MUL=self.upsample2_MUL(MUL)",MUL.shape)
            MUL = F.interpolate(MUL, size=(H, W), mode='bilinear', align_corners=ALIGN_CORNERS)
            x1_mul, x2_mul, x3_mul, x4_mul=self.hrnet_mul(MUL)
            feature.append(x1_mul)
            feature.append(x2_mul)
            feature.append(x3_mul)
            feature.append(x4_mul)

        if VI!=None and  self.VI_modal:
            H, W =  2*VI.size(2),  2*VI.size(3)
            VI = self.conv_VI1(VI)#13通道变3通道[4, 3, 192, 192]
            #VI=self.upsample2_VI(VI)#[4, 10, 96, 96]→[4, 10, 192, 192]
            #print("VI=self.upsample2_VI(VI)",VI.shape)
            VI  = F.interpolate(VI, size=(H, W), mode='bilinear', align_corners=ALIGN_CORNERS)

            x1_VI, x2_VI, x3_VI, x4_VI=self.hrnet_VI(VI)
            feature.append(x1_VI)
            feature.append(x2_VI)
            feature.append(x3_VI)
            feature.append(x4_VI)


       
        if len(feature)==8 and self.MI_cul:
            #lat_loss1 = self.mi_level1(self.convx1_rgb(feature[0]), self.convx1_mul(feature[4]))
            #lat_loss2 = self.mi_level2(self.convx2_rgb(feature[1]), self.convx2_mul(feature[5]))
            #lat_loss3 = self.mi_level3(self.convx3_rgb(feature[2]), self.convx3_mul(feature[6]))
            if self.RGB_modal and self.MUL_modal:
                lat_loss3 = self.mi_level3(self.convx3_rgb(feature[2]), self.convx3_mul(feature[6]))
                lat_loss4 = self.mi_level4(self.convx4_rgb(feature[3]), self.convx4_mul(feature[7]))
            elif self.RGB_modal and self.VI_modal:
                lat_loss3 = self.mi_level3(self.convx3_rgb(feature[2]), self.convx3_vi(feature[6]))
                lat_loss4 = self.mi_level4(self.convx4_rgb(feature[3]), self.convx4_vi(feature[7]))
            elif self.MUL_modal and self.VI_modal:
                lat_loss3 = self.mi_level3(self.convx3_mul(feature[2]), self.convx3_vi(feature[6]))
                lat_loss4 = self.mi_level4(self.convx4_mul(feature[3]), self.convx4_vi(feature[7]))

            lat_loss = lat_loss3 +lat_loss4
            
           
        elif len(feature)==12 and self.MI_cul:
            #print(feature[0][0].shape)
            lat_loss1 = self.mi_level1(self.convx1_rgb(feature[0]), self.convx1_mul(feature[4]))
            lat_loss2 = self.mi_level2(self.convx2_rgb(feature[1]), self.convx2_mul(feature[5]))
            lat_loss3 = self.mi_level3(self.convx3_rgb(feature[2]), self.convx3_mul(feature[6]))
            lat_loss4 = self.mi_level4(self.convx4_rgb(feature[3]), self.convx4_mul(feature[7]))
            lat_loss5 = self.mi_level5(self.convx1_rgb(feature[0]), self.convx1_vi(feature[8]))
            lat_loss6 = self.mi_level6(self.convx2_rgb(feature[1]), self.convx2_vi(feature[9]))
            lat_loss7 = self.mi_level7(self.convx3_rgb(feature[2]), self.convx3_vi(feature[10]))
            lat_loss8 = self.mi_level8(self.convx4_rgb(feature[3]), self.convx4_vi(feature[11]))
            #lat_loss = lat_loss1 + lat_loss2 + lat_loss3 + lat_loss4 + lat_loss5 + lat_loss6 + lat_loss7 + lat_loss8
            #lat_loss = lat_loss2 + lat_loss3 + lat_loss4 + lat_loss6 + lat_loss7 + lat_loss8
            lat_loss = lat_loss3 + lat_loss4 + lat_loss7 + lat_loss8
            #lat_loss = lat_loss4 + lat_loss8
        else:
            lat_loss = 0
        
        high_level_features = torch.cat(feature, 1)
        
        high_level_features =self.conv_concat(high_level_features)
        #print(high_level_features.shape)
        
        out_aux = self.aux_head(high_level_features)
        feats = self.conv3x3_ocr(high_level_features)
        # compute contrast feature
        context = self.ocr_gather_head(feats, out_aux)
        feats = self.ocr_distri_head(feats, context)
        out = torch.sigmoid(self.cls_head(feats))

        #F.interpolate(VI, size=(2*VI.size(2), 2*VI.size(3)), mode='bilinear', align_corners=ALIGN_CORNERS)

        out_aux_seg = []
        out_aux_seg.append(F.interpolate(out_aux, size=(H, W), mode='bilinear', align_corners=ALIGN_CORNERS))
        out_aux_seg.append(F.interpolate(out, size=(H, W), mode='bilinear', align_corners=ALIGN_CORNERS))
        
        
        
        if self.MI_cul:
            out_aux_seg.append(lat_loss)

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
            print(set(model_dict) - set(pretrained_dict))            
            print(set(pretrained_dict) - set(model_dict))
            #keys_to_skip = {'cls_head', 'aux_head'}
            #pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in keys_to_skip}
            pretrained_dict = {k: v for k, v in pretrained_dict.items()  if k in model_dict.keys() and v.shape == model_dict[k].shape}
            for k, _ in pretrained_dict.items():
                logger.info(
                     '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        elif pretrained:
            raise RuntimeError('No such file {}'.format(pretrained))


def get_seg_model(cfg, **kwargs):
    model = RustQNet(cfg, **kwargs)
    model.init_weights(cfg.MODEL.PRETRAINED)

    return model
