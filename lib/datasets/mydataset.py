# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os
import os.path as osp
import cv2
import numpy as np
from PIL import Image
from osgeo import gdal
import torch
from torch.nn import functional as F
import torchvision.transforms as transforms
from .indices import RSIndex
from .base_dataset import BaseDataset
import torchvision.transforms.functional as TF
import random
import logging
class mydataset(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path, 
                 num_samples=None, 
                 num_classes=1,
                 multi_scale=False, 
                 flip=False, 
                 ignore_label=-1, 
                 base_size=2048, 
                 crop_size=(512, 1024), 
                 downsample_rate=1,
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225],
                 RGB_modal=True,
                 MUL_modal=True,
                 VI_modal=True):

        super(mydataset, self).__init__(ignore_label, base_size,
                crop_size, downsample_rate, scale_factor, mean, std,)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes

        
        self.RGB_modal=RGB_modal
        self.MUL_modal=MUL_modal
        self.VI_modal=VI_modal

        self.multi_scale = multi_scale
        self.flip = flip
        self.crop_size=crop_size
        self.img_list = [line.strip().split() for line in open(osp.join(root,list_path))]

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]

        self.rgb_transform = transforms.Compose([transforms.ToTensor(),
            transforms.Resize((self.crop_size[0], self.crop_size[1])),#to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        self.gt_transform = transforms.Compose([transforms.ToTensor(),
            transforms.Resize((self.crop_size[0], self.crop_size[1])),
            ])
        self.mul_transform = transforms.Compose([transforms.ToTensor(),
            transforms.Resize((int(self.crop_size[0]/2), int(self.crop_size[0]/2))),
            transforms.Normalize([0.5] * 10, [0.5] * 10)])
        self.vi_transform = transforms.Compose([transforms.ToTensor(),
            transforms.Resize((int(self.crop_size[0]/2), int(self.crop_size[0]/2))),
            transforms.Normalize([0.5] * 13, [0.5] * 13)])

    
    def read_files(self):
        files = []
        for item in self.img_list:
            #break
            image_rgb_path, image_mul_path, label_path = item
            name = os.path.splitext(os.path.basename(image_rgb_path))[0]
            files.append({
                "img_rgb": image_rgb_path,
                "img_mul": image_mul_path,
                "label": label_path,
                "name": name,
                "weight": 1
            })
        self.size = len(files)
        print(" {} samples in file {}".format(len(files), self.list_path))
        return files

    def read_img(self, path):        
        dataset = gdal.Open(path.replace('\\', '/'))
        if dataset == None:
            raise IOError('Cannot open', path)
        im_data = dataset.ReadAsArray()
        if im_data.ndim == 2:
            im_data = im_data[:, :, np.newaxis]#添加一个额外的维度，将二维图像扩展为三维（H x W x 1）。
        else:
            if im_data.ndim == 3:
                im_data = im_data.transpose((1, 2, 0))#to[H,W,C]
        return im_data 
    
   
    def __len__(self):
        return self.size

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]

        image_rgb = self.read_img(os.path.join(self.root, item["img_rgb"]))
        image_rgb = self.rgb_transform(image_rgb)
        image_mul = self.read_img(os.path.join(self.root, item["img_mul"]))
        image_mul = self.mul_transform(image_mul)
        image_vi = RSIndex(self.read_img(os.path.join(self.root, item["img_mul"])))
        image_vi = self.vi_transform(image_vi)

        size = image_rgb.shape
 
        label = self.read_img(os.path.join(self.root, item["label"]))
        label = self.gt_transform(label)
        if self.flip :
            image_rgb, label, image_mul, image_vi = cv_random_flip(image_rgb, label, image_mul, image_vi) 
            #image_rgb, label, image_mul, image_vi = MULti_scale(image_rgb, label, image_mul, image_vi)
            image_rgb, label, image_mul, image_vi = randomCrop(image_rgb, label, image_mul, image_vi)
            #print(" randomCrop(RGB, gt, MUL)", RGB.shape, gt.shape, VI.shape, MUL.shape,)
            image_rgb, label, image_mul, image_vi = randomRotation(image_rgb, label, image_mul, image_vi)
            #print("randomRotation(RGB, gt, MUL)", RGB.shape, gt.shape, VI.shape, MUL.shape,)
            image_rgb = colorEnhance(image_rgb)
            image_rgb = TF.resize(image_rgb, self.crop_size)
            label = TF.resize(label, self.crop_size)
            image_mul = TF.resize(image_mul, [int(self.crop_size[0]/2),int(self.crop_size[1]/2)])
            image_vi = TF.resize(image_vi, [int(self.crop_size[0]/2),int(self.crop_size[1]/2)])
        
        
        images={"RGB":image_rgb,
                "MUL":image_mul,
                "VI":image_vi,}
        
        return images, label, np.array(size), name
    
def cv_random_flip(RGB, label, MUL, VI):#accept tensor data instead of PIL images
    if torch.rand(1) > 0.5:
        flip_flags = torch.randint(0, 2, (2,)).tolist()    
        if flip_flags[0] == 1:

                RGB = torch.flip(RGB, [-1])#Use torch.flip() to perform the left-right flipping operation on the tensor data. 

                label = torch.flip(label, [-1])#The [-1] argument indicates the last dimension, which corresponds to flipping along the horizontal axis.

                MUL = torch.flip(MUL, [-1])

                VI = torch.flip(VI, [-1])
        if flip_flags[1] == 1:

                RGB = torch.flip(RGB, [-2])#Use torch.flip() to perform the left-right flipping operation on the tensor data. 

                label = torch.flip(label, [-2])#The [-1] argument indicates the last dimension, which corresponds to flipping along the horizontal axis.

                MUL = torch.flip(MUL, [-2])

                VI = torch.flip(VI, [-2])
    return RGB, label, MUL, VI

def randomCrop(RGB, label, MUL, VI):
    if torch.rand(1) > 0.5:
        image_width = MUL.size(2)
        image_height = MUL.size(1)
        border = int(image_width / 5 )
        crop_win_width = torch.randint(image_width - border, image_width, (1,)).item()
        crop_win_height = torch.randint(image_height - border, image_height, (1,)).item()
        random_region = (
            (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
            (image_height + crop_win_height) >> 1)

        RGB = RGB[:, 2*random_region[1]:2*random_region[3], 2*random_region[0]:2*random_region[2]]

        label = label[:, 2*random_region[1]:2*random_region[3], 2*random_region[0]:2*random_region[2]]

        MUL = MUL[:, random_region[1]:random_region[3], random_region[0]:random_region[2]]

        VI = VI[:, random_region[1]:random_region[3], random_region[0]:random_region[2]]
    return RGB, label, MUL, VI

def MULti_scale(RGB, label, MUL, VI):
    scale_list = [0.5, 0.8, 1.5, 2.0]
    target_size = (MUL.size(1), MUL.size(2))
    if torch.rand(1) > 0.5:
        scale = random.choice(scale_list)

        transformations_RGB = []
        transformations_MUL = []
        if scale < 1.0:
            resize_transform = transforms.Resize(size=(int(target_size[0] * scale), int(target_size[1] * scale)), interpolation=Image.BICUBIC)
            pad_transform = transforms.Pad(padding=(int((target_size[1] - int(target_size[1] * scale))/2), int((target_size[0] - int(target_size[0] * scale))/2)), fill=0)
            transformations_MUL.append(transforms.Compose([resize_transform, pad_transform]))

            resize_transform = transforms.Resize(size=(int(2 * target_size[0] * scale), int(2 * target_size[1] * scale)), interpolation=Image.BICUBIC)
            pad_transform = transforms.Pad(padding=(int((2 * target_size[1] - int(2 * target_size[1] * scale))/2), int((2 * target_size[0] - int(2 * target_size[0] * scale))/2)), fill=0)
            transformations_RGB.append(transforms.Compose([resize_transform, pad_transform]))

        else:
            resize_transform = transforms.Resize(size=(int(target_size[0] * scale), int(target_size[1] * scale)), interpolation=Image.BICUBIC)
            crop_transform = transforms.RandomCrop(size=target_size)
            transformations_MUL.append(transforms.Compose([resize_transform, crop_transform]))
            resize_transform = transforms.Resize(size=(int(2 * target_size[0] * scale), int(2 * target_size[1] * scale)), interpolation=Image.BICUBIC)
            crop_transform = transforms.RandomCrop(size=(2 * target_size[0], 2 * target_size[0]))
            transformations_RGB.append(transforms.Compose([resize_transform, crop_transform]))

        # 对每个缩放尺度进行处理
        for transform in transformations_RGB:

            RGB = transform(RGB)
            label = transform(label)
        for transform in transformations_MUL:

            MUL = transform(MUL)

            VI = transform(VI)
    return RGB, label, MUL, VI



def randomRotation(RGB, label, MUL, VI):
    if torch.rand(1) > 0.5:
        random_angle = torch.randint(-30, 30, (1,)).item()

        RGB = TF.rotate(RGB, random_angle)#Set resample=False to disable resampling and maintain the original pixel values.

        label = TF.rotate(label, random_angle)#https://pytorch.org/vision/stable/generated/torchvision.transforms.functional.rotate.html

        MUL = TF.rotate(MUL, random_angle)

        VI = TF.rotate(VI, random_angle)
    return RGB, label, MUL, VI
    
def colorEnhance(image):#expected to be in […, 1 or 3, H, W] format
    if torch.rand(1) > 0.8:
        bright_intensity = torch.randint(5, 15, (1,)).item() / 10.0 #0 gives a black image, 1 gives the original image while 2 increases the brightness by a factor of 2.
        image = TF.adjust_brightness(image, bright_intensity)
        contrast_intensity = torch.randint(5, 15, (1,)).item() / 10.0#0 gives a solid gray image, 1 gives the original image while 2 increases the contrast by a factor of 2
        image = TF.adjust_contrast(image, contrast_intensity)
        color_intensity = torch.randint(5, 15, (1,)).item() / 10.0# 0 will give a black and white image, 1 will give the original image while 2 will enhance the saturation by a factor of 2.
        image = TF.adjust_saturation(image, color_intensity)
        sharp_intensity = torch.randint(5, 15, (1,)).item() / 10.0
        image = TF.adjust_sharpness(image, sharp_intensity)#0 gives a blurred image, 1 gives the original image while 2 increases the sharpness by a factor of 2.
    return image

        
        
