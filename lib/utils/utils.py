# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

class FullModel(nn.Module):
  """
  Distribute the loss on multi-gpu to reduce 
  the memory cost in the main gpu.
  You can check the following discussion.
  https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
  """
  def __init__(self, model, loss):
    super(FullModel, self).__init__()
    self.model = model
    self.loss = loss

  def forward(self, inputs, labels, epoch=1, *args, **kwargs):
    outputs = self.model(inputs, *args, **kwargs)
    loss = self.loss(outputs, labels, epoch)
    return torch.unsqueeze(loss,0), outputs

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg
import shutil
def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    cfg_name_ori = cfg_name
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / model / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg_name_ori , str(final_output_dir))

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = final_output_dir / \
            (cfg_name + '_' + time_str)
    print('=> creating tensorboard_log_dir {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)

def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    #print("pred.shape", pred.shape)
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    #print("output", output.shape)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    #print("seg_pred", seg_pred.shape)
    
    seg_gt = np.squeeze((label.cpu().numpy().transpose(0, 2, 3, 1)), axis=-1)
    
  
    #seg_gt = convert_label_2_cls(seg_gt)
    #print("seg_gt", seg_gt.shape)
    seg_gt = np.asarray(seg_gt, dtype=np.int)
    
    
    if len(seg_gt.shape) != 3:
        raise ValueError(
            'The shape of label is not 3 dimension as (N, H, W), it is {}'.
            format(seg_gt.shape))

    if len(seg_pred.shape) != 3:
        raise ValueError(
            'The shape of logits is not 3 dimension as (N, H, W), it is {}'.
            format(seg_pred.shape))


    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]
    

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))   


    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred] = label_count[cur_index]

    return confusion_matrix

import math
def adjust_learning_rate(optimizer, base_lr, cur_iters, max_iters):
    if cur_iters < int(max_iters*0.1):
        lr = base_lr
    else:
        lr = base_lr * (math.exp(0.1 * (int(max_iters*0.1)-cur_iters)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
"""
def adjust_learning_rate_old(optimizer, base_lr, max_iters,
        cur_iters, power=0.9, nbb_mult=10):
    lr = base_lr*((1-float(cur_iters)/max_iters)**(power))
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) == 2:
        optimizer.param_groups[1]['lr'] = lr * nbb_mult
    return lr
"""
def convert_label_2_cls(label):
    mask = label.copy()
    mask=np.where((mask>=0.82) & (mask<1),8,mask)
    mask=np.where((mask>=0.64) & (mask<1),7,mask)
    mask=np.where((mask>=0.46) & (mask<1),6,mask)
    mask=np.where((mask>=0.28) & (mask<1),5,mask)
    mask=np.where((mask>=0.19) & (mask<1),4,mask)
    mask=np.where((mask>=0.145) & (mask<1),3,mask)
    mask=np.where((mask>0.1) & (mask<1),2,mask)
    mask=np.where(mask==0.1,1,mask)
    return mask

def regression_evaluation(results):

    #results = tuple(zip(*results))
    #assert len(results) == 2
    true_value, pred_value=results [:, 0],results [:, 1]
    #print("true_value",true_value[-1].shape)
    #true_value = np.array([item for sublist in true_value for item in sublist])
    #print("true_value",true_value.shape)
    #pred_value = np.array([item for sublist in pred_value for item in sublist])
    r2 = r2_score(true_value, pred_value)
    mae = mean_absolute_error(true_value, pred_value)
    rmse = np.sqrt(mean_squared_error(true_value, pred_value))
    mse = mean_squared_error(true_value, pred_value)
    print("r2:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mse:{:.4f}".format(r2, mae, rmse, mse))
    return r2, mae, rmse, mse

def reg_prepare(pred, label):
    #N, C, H, W = pred.shape
    #print(pred.shape)
    mask = (label >= 0.1)#忽略土壤的值
    pred_value = pred[mask]
    #print("pred_value = pred[mask]", pred.shape)
    true_value = label[mask]

    return true_value, pred_value



def to_uint8(im, stretch=False):
    from skimage import exposure
    import numpy as np
    # 2% linear stretch
    def _two_percent_linear(image, max_out=255, min_out=0):
        def _gray_process(gray, maxout=max_out, minout=min_out):
            # Get the corresponding gray level at 98% in the histogram.
            high_value = np.percentile(gray, 98)
            low_value = np.percentile(gray, 2)
            truncated_gray = np.clip(gray, a_min=low_value, a_max=high_value)
            processed_gray = ((truncated_gray - low_value) / (high_value - low_value)) * (maxout - minout)
            return np.uint8(processed_gray)

        if len(image.shape) == 3:
            processes = []
            for b in range(image.shape[-1]):
                processes.append(_gray_process(image[:, :, b]))
            result = np.stack(processes, axis=2)
        else:  # if len(image.shape) == 2
            result = _gray_process(image)
        return np.uint8(result)

    # Simple image standardization
    def _sample_norm(image):
        stretches = []
        if len(image.shape) == 3:
            for b in range(image.shape[-1]):
                stretched = exposure.equalize_hist(image[:, :, b])
                stretched /= float(np.max(stretched))
                stretches.append(stretched)
            stretched_img = np.stack(stretches, axis=2)
        else:  # if len(image.shape) == 2
            stretched_img = exposure.equalize_hist(image)
        return np.uint8(stretched_img * 255)

    dtype = im.dtype.name
    if dtype != "uint8":
        im = _sample_norm(im)
    if stretch:
        im = _two_percent_linear(im)
    return im

def get_color_map_list(num_classes):
    num_classes += 1
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = color_map[3:]
    from palettable.tableau import Tableau_10
    #Tableau_10.show_discrete_image()
    tem=Tableau_10.colors
    tem2=[tem[7],tem[2],tem[0],tem[9],tem[8],tem[5],tem[6],tem[4],tem[1],tem[3]]
    j=0
    for i in tem2:
        color_map[j * 3]=i[0]
        color_map[j * 3+1]=i[1]
        color_map[j * 3+2]=i[2]
        j=j+1
    return color_map


def mask_di_level(mask):
    mask[(mask >= 0.82) & (mask < 1)] = 8
    mask[(mask >= 0.64) & (mask < 1)] = 7
    mask[(mask >= 0.46) & (mask < 1)] = 6
    mask[(mask >= 0.28) & (mask < 1)] = 5
    mask[(mask >= 0.19) & (mask < 1)] = 4
    mask[(mask >= 0.145) & (mask < 1)] = 3
    mask[(mask > 0.1) & (mask < 1)] = 2
    mask[mask == 0.1] = 1
    return mask

import cv2
import torch.nn.functional as F
from PIL import Image
color_map = get_color_map_list(256)



def visualize_all_pred(img_list, feature_list):
    RGBs, MULs, VIs=img_list[0], img_list[1], img_list[2]
    MULs = F.interpolate(MULs, size=(RGBs.shape[2], RGBs.shape[3]), mode='bilinear', align_corners=True)
    VIs = F.interpolate(VIs, size=(RGBs.shape[2], RGBs.shape[3]), mode='bilinear', align_corners=True)
    batch_img=[]
    batch_label=[]
    for kk in range(feature_list[1].shape[0]):
        RGB, mul, vi = RGBs[kk, :, :, :], MULs[kk, :, :, :], VIs[kk, :, :, :]
        mul = mul.detach().cpu().numpy().squeeze()
        bands = [5, 3, 1]        
        mul = mul[bands]
        mul = np.transpose(mul, (1, 2, 0))        
        mul = to_uint8(mul, stretch=False)
        mul = cv2.cvtColor(mul, cv2.COLOR_RGB2BGR)
        
        vi = vi.detach().cpu().numpy().squeeze()
        bands = [5, 3, 1]        
        vi = vi[bands]
        vi = np.transpose(vi, (1, 2, 0))        
        vi = to_uint8(vi, stretch=False)
        vi = cv2.cvtColor(vi, cv2.COLOR_RGB2BGR)
        
        RGB = RGB.detach().cpu().numpy().squeeze()
        RGB = np.transpose(RGB, (1, 2, 0))
        RGB = to_uint8(RGB, stretch=False)
        RGB = cv2.cvtColor(RGB, cv2.COLOR_RGB2BGR)        
        cat_img = np.hstack([RGB, mul, vi])
        batch_img.append(cat_img)
        
        pred=[]
        for i in range(len(feature_list)):
            pred_kk=feature_list[i][kk, :, :, :]
            pred_kk = pred_kk.detach().cpu().numpy().squeeze()
            pred.append(pred_kk)
        cat_label = np.hstack(pred)
        batch_label.append(cat_label)
    cat_img =  np.vstack( batch_img )
    cat_label =  np.vstack( batch_label )    
    cat_label =  mask_di_level(cat_label)
    lbl=np.asarray(cat_label)
    lbl_pil = Image.fromarray(lbl.astype(np.uint8), mode='P')
    lbl_pil.putpalette(color_map)
    lbl_pil=np.array(lbl_pil.convert('RGB'))
    return cat_img, lbl_pil
    #img_name = '{:s}_{:02d}_RGB_MUL.jpg'.format(fid, kk)
    #abel_name = '{:s}_{:02d}_gt_.png'.format(fid, kk)
    #cv2.imwrite(save_path + img_name, cat_img)
    #label_name = os.path.join(save_path, label_name)    
    #lbl_pil.save(label_name)


















