# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn import functional as F
import logging
from config import config
from .regloss import get_laplacian_loss_whole_img
import numpy as np
class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label
        )

    def _forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)

        loss = self.criterion(score, target)

        return loss

    def forward(self, score, target):

        if config.MODEL.NUM_OUTPUTS == 1:
            score = [score]

        weights = config.LOSS.BALANCE_WEIGHTS
        assert len(weights) == len(score)

        return sum([w * self._forward(x, target) for (w, x) in zip(weights, score)])

def dice_loss(  pred,
                        target,
                        weight,
                        eps=1e-3,
                        reduction='mean',
                        naive_dice=False,
                        avg_factor=None,
                    ):

    input = pred.flatten(1)
    target = target.flatten(1).float()

    a = torch.sum(input * target, 1)
    if naive_dice:
        b = torch.sum(input, 1)
        c = torch.sum(target, 1)
        d = (2 * a + eps) / (b + c + eps)
    else:
        b = torch.sum(input * input, 1) + eps
        c = torch.sum(target * target, 1) + eps
        d = (2 * a) / (b + c)

    loss = 1 - d
    if weight is not None:
        assert weight.ndim == loss.ndim
        assert len(weight) == len(pred)
    #loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss

class DiceLoss(nn.Module):
    def __init__(self,
                 use_sigmoid=False,
                 activate=False,
                 reduction='mean',
                 naive_dice=False,
                 loss_weight=1.0,
                 ignore_index=255,
                 eps=1e-3,
                 loss_name='loss_dice'):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.naive_dice = naive_dice
        self.loss_weight = loss_weight
        self.eps = eps
        self.activate = activate
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        pred=pred.argmax(dim=1, keepdim=True)
        #print()
        if self.activate:
            if self.use_sigmoid:
                pred = pred.sigmoid()
            else:
                raise NotImplementedError

        loss = self.loss_weight * dice_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            naive_dice=self.naive_dice,
            avg_factor=avg_factor,
        )

        return loss


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, thres=0.7,
                 min_kept=100000, weight=None, end_epoch=50):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.end_epoch = end_epoch
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label,
            reduction='mean')
        self.DiceLoss = DiceLoss()

    def _ce_forward(self, score, target):
        target = target.squeeze()
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        #print("_ce_forward_score.size, target.size",score.size(), target.size())
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)
        loss = self.criterion(score, target)
        return loss

    def exponential_annealing(self, start_value, end_value, current_epoch, end_epoch):
        decay_rate = math.log(end_value / start_value) / end_epoch
        anneal_reg = start_value * math.exp(decay_rate * current_epoch)
        return anneal_reg

    def _ohem_forward(self, score, target, **kwargs):
        target = target.squeeze()
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

    def forward(self, score, target, epoch):
        if config.MODEL.NUM_OUTPUTS == 1:
            score = [score]
        weights = config.LOSS.BALANCE_WEIGHTS
        if config.MODEL.MI_cul:
            anneal_reg = self.exponential_annealing(0.001, 1, epoch, self.end_epoch)
            Wmi = weights[-1]
            scoreMI = score[-1]
            weights = weights[:-1]
            score = score[:-1]
            loss_MI=Wmi *anneal_reg*scoreMI
            functions_ce = [self._ce_forward] * (len(weights) )
            functions_dice = [self.DiceLoss] * (len(weights) )
            ce_loss_list=[w * func(x, target)   for (w, x, func) in zip(weights, score, functions_ce)]
            dice_loss_list = [w * func(x, target) for (w, x, func) in zip(weights, score, functions_dice)]
            total_loss=sum(ce_loss_list) + sum(dice_loss_list) + loss_MI
            return total_loss
        functions_ce = [self._ce_forward] * (len(weights))
        functions_dice = [self.DiceLoss] * (len(weights))
        ce_loss_list = [w * func(x, target) for (w, x, func) in zip(weights, score, functions_ce)]
        dice_loss_list = [w * func(x, target) for (w, x, func) in zip(weights, score, functions_dice)]
        total_loss = sum(ce_loss_list) + sum(dice_loss_list)
        return total_loss

import math



class RegLoss(nn.Module):
    def __init__(self, ignore_label=-1, weight=None, end_epoch=100):
        super(RegLoss, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.MSELoss(reduction='mean')
        self.ce_criterion = nn.CrossEntropyLoss()
        self.end_epoch=end_epoch
    def _forward(self, score, target):
        loss = self.criterion(score, target)
        return 10*loss
    
    def structure_loss(self, pred, mask):
        pred=pred.argmax(dim=1, keepdim=True)
        #print("pred=pred.argmax(dim=1, keepdim=True)",pred.shape)
        #pred_ = (pred.clone() >= 0.1).int() 
        mask_ = (mask.clone() >= 0.1).int()  
        weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
        wbce  = F.binary_cross_entropy_with_logits(torch.sigmoid(pred), torch.sigmoid(mask_), reduction='none')#F.cross_entropy
        wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))
        pred  = torch.sigmoid(pred)
        mask  = torch.sigmoid(mask_)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou  = 1-(inter+1)/(union-inter+1)#计算加权交并比损失：
        return (wbce+wiou).mean()

    def structure_loss2(self, pred, mask):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return (wbce + wiou).mean()

    def CE_loss(self, pred, mask):
        #pred=pred.argmax(dim=1, keepdim=False)
        mask = torch.argmax(mask, dim=1)
        #mask = torch.nn.functional.one_hot(mask).float()
        #mask=torch.squeeze(mask, 1)
        ce_loss=self.ce_criterion(pred, mask.long())
        return ce_loss

    def exponential_annealing(self, start_value, end_value, current_epoch, end_epoch):
        decay_rate = math.log(end_value / start_value) / end_epoch
        anneal_reg = start_value * math.exp(decay_rate * current_epoch)
        return anneal_reg


    def linear_annealing(self, init, fin, step, annealing_steps):
        """Linear annealing of a parameter."""
        if annealing_steps == 0:
            return fin
        assert fin > init
        delta = fin - init
        annealed = min(init + delta * step / annealing_steps, fin)
        return annealed

    def forward(self, score, target, epoch=1):
        #anneal_reg = self.linear_annealing(0, 10, epoch, self.end_epoch)
        weights = config.LOSS.BALANCE_WEIGHTS
        if config.MODEL.MICCR:
            total_loss = []
            for kk in range(len(score)-1):
                if kk == 0:
                    total_loss.append(10 * self._forward(score[kk], target) + self.structure_loss2(score[kk], target))
                else:
                    total_loss.append(self._forward(score[kk], target) + self.structure_loss2(score[kk], target))
            if config.MODEL.MI_cul:
                anneal_reg = self.exponential_annealing(0.001, 1, epoch, self.end_epoch)
                loss_MI = weights[2] * anneal_reg * score[-1]
                total_loss.append(loss_MI)
            return sum(total_loss)


        loss_cls=weights[0]*self.structure_loss(score[0], target)
        #loss_cls=weights[0]*self.CE_loss(score[0], target[1])
        loss_mse=weights[1]*self._forward(score[1], target)
        #loss_laplacian=0.2*weights[1]*get_laplacian_loss_whole_img(score[1], target[0])
        if config.MODEL.MI_cul:
            anneal_reg = self.exponential_annealing(0.001, 1, epoch, self.end_epoch)
            loss_MI=weights[2]*anneal_reg*score[2]
            return sum([loss_cls, loss_mse, loss_MI])
        return sum([loss_cls, loss_mse])


class RegLossDeeplabv3p(nn.Module):
    def __init__(self, ignore_label=-1, weight=None, end_epoch=100):
        super(RegLossDeeplabv3p, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.MSELoss(reduction='mean')
        self.end_epoch = end_epoch

    def _forward(self, score, target):
        loss = self.criterion(score, target)
        return 10 * loss

    def structure_loss(self, pred, mask):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return (wbce + wiou).mean()

    def forward(self, score, target, epoch=1):
        loss_cls = self.structure_loss(score, target)
        loss_mse = self._forward(score, target)
        return sum([loss_cls, loss_mse])



class IOU(torch.nn.Module):
    def __init__(self, size_average=True):
        super(IOU, self).__init__()
        self.size_average = size_average

    def _iou(self, pred, target, size_average=True):
        b = pred.shape[0]
        IoU = 0.0
        for i in range(0, b):
            # compute the IoU of the foreground
            Iand1 = torch.sum(target[i, :, :, :] * pred[i, :, :, :])
            Ior1 = torch.sum(target[i, :, :, :]) + torch.sum(pred[i, :, :, :]) - Iand1
            IoU1 = Iand1 / Ior1

            # IoU loss is (1-IoU1)
            IoU = IoU + (1 - IoU1)

        return IoU / b

    def forward(self, pred, target):
        return self._iou(pred, target, self.size_average)








