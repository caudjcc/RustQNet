# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import argparse
import os
import pprint
import shutil
import sys
import importlib
import logging
import time
import timeit
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import _init_paths
import models
import datasets
from config import config
from config import update_config
from core.function import testval, test
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModel
from pathlib import Path
from tqdm import tqdm
find_bug=False
os.environ["CUDA_VISIBLE_DEVICES"] = '9'

save_npz = False
save_npz = True
def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default=r"./result/test.yaml",
                        #required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

def get_r2best_pth(folder_path):
    import os
    file_names = os.listdir(folder_path)
    r2_file_names = [name for name in file_names if 'best_R2' in name]
    r2_values = [float(name.split('_')[4]) for name in r2_file_names]
    max_index = r2_values.index(max(r2_values))
    max_r2_file_name = r2_file_names[max_index]
    print("file of best R2:", max_r2_file_name)
    return max_r2_file_name

def main():
    args = parse_args()


    final_output_dir=Path(os.path.join(config.LOG_DIR,"Predict"))
    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)
    final_output_dir=str(final_output_dir)
    print(final_output_dir)
    import shutil
    shutil.copy(args.cfg, final_output_dir)
    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    if torch.__version__.startswith('1'):
        module = eval('models.'+config.MODEL.NAME)
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    module = importlib.import_module('models.' + config.MODEL.NAME)
    model = getattr(module, 'get_seg_model')(config)
    #model = eval('models.'+config.MODEL.NAME + '.get_seg_model')(config)

    """
    dump_input = torch.rand(
        (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    )
    #logger.info(get_model_summary(model.cuda(), dump_input.cuda()))
    """
    if os.path.isfile(config.TEST.MODEL_FILE):
        model_state_file = config.TEST.MODEL_FILE
    else:
        model_state_file = os.path.join(os.path.dirname(args.cfg), get_r2best_pth(os.path.dirname(args.cfg)))
    print('=> loading model from {}'.format(model_state_file))
        
    pretrained_dict = torch.load(model_state_file)

    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    #for k, v in pretrained_dict.items():
        #print(k,k[6:])
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                        if k[6:] in model_dict.keys()}
    #for k, _ in pretrained_dict.items():
    #    print('=> loading {} from pretrained model'.format(k))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    #model.load_pretrain_model(model_state_file)

    #gpus = list(config.GPUS)
    #model = nn.DataParallel(model, device_ids=gpus).cuda()
    model = model.cuda()
    # prepare data
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TEST_SET,
                        num_samples=None,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=False,
                        flip=False,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TEST.BASE_SIZE,
                        crop_size=test_size,
                        downsample_rate=1)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=  config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)
    
    start = timeit.default_timer()

    model.eval()
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    from torchmetrics.classification import BinaryConfusionMatrix
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bcm = BinaryConfusionMatrix().to(device)
    from torchmetrics.regression import R2Score
    r2score = R2Score().to(device)
    # from torchmetrics.regression import RelativeSquaredError
    # relative_squared_error = RelativeSquaredError().to(device)
    from torchmetrics.regression import MeanSquaredError
    mean_squared_error = MeanSquaredError().to(device)
    from torchmetrics.regression import MeanAbsoluteError
    mean_absolute_error = MeanAbsoluteError().to(device)


    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            if find_bug:
                if index > 20: continue

            images, label, _, name, *border_padding = batch
            images = {key: images[key].cuda() for key in images}
            label = label.cuda()
            pred_ = model(images)
            if config.MODEL.MICCR:
                pred_ = pred_[0]
            elif config.MODEL.MI_cul:
                pred_ = pred_[-2]
            else:
                pred_ = pred_[-1]

            #pred_ = pred_[-2]
            #print(final_output_dir)
            #print(name)
            partial_path = final_output_dir[:-16]
            if save_npz:
                for i, NA in enumerate(name):
                    #continue
                    np.savez_compressed(os.path.join(str(final_output_dir), NA+'.npz'), pred_[i,:,:,:].detach().cpu().numpy())
                    #np.savez_compressed(os.path.join("/mnt/sda/dengjie/HRNet-Semantic-Segmentation-HRNet-OCR/result/Compare_RGB_MUL_VI_OCR-MI-Net/LABEL/", NA + '.npz'), label[i, :, :, :].detach().cpu().numpy())
            label_cls = (label.clone() >= 0.1).int()
            pred_cls = (pred_.clone() >= 0.06).int()
            bcm_result = bcm(pred_cls.view(-1), label_cls.view(-1))
            pred_reshaped = pred_.clone().view(-1)
            mask_reshaped = label.clone().view(-1)
            filtered_pred = pred_reshaped[mask_reshaped >= 0.1]
            filtered_mask = mask_reshaped[mask_reshaped >= 0.1]
            if len(filtered_pred) > 2 or len(filtered_mask) > 2:
                filtered_pred = (filtered_pred - 0.1) * 100 / 0.9
                filtered_pred[filtered_pred < 0] = 0
                filtered_mask = (filtered_mask - 0.1) * 100 / 0.9

                r2score_result = r2score(filtered_pred.view(-1), filtered_mask.view(-1))
                mean_squared_error_result = mean_squared_error(filtered_pred.view(-1), filtered_mask.view(-1))
                mean_absolute_error_result = mean_absolute_error(filtered_pred.view(-1), filtered_mask.view(-1))


    bcm_result = bcm.compute().cpu().numpy()
    r2score_result = r2score.compute().cpu().numpy()
    mean_squared_error_result = mean_squared_error.compute().cpu().numpy()
    mean_absolute_error_result = mean_absolute_error.compute().cpu().numpy()
    confusion_matrix = bcm_result
    mean_IoU = 0
    IoU_array = 0
    OA = 0
    r2 = 0
    rmse = 1

    final_class = 2
    EPS = 1e-32
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()
    OA = np.trace(confusion_matrix) / np.sum(confusion_matrix)

    Precision = np.zeros(final_class)
    Recall = np.zeros(final_class)
    F1 = np.zeros(final_class)

    for i in range(final_class):
        TP = confusion_matrix[i, i]
        FP = np.sum(confusion_matrix[:, i]) - TP
        FN = np.sum(confusion_matrix[i, :]) - TP

        Precision[i] = TP / (TP + FP + EPS)
        Recall[i] = TP / (TP + FN + EPS)
        F1[i] = 2 * (Precision[i] * Recall[i]) / (Precision[i] + Recall[i] + EPS)

    r2, mae, rmse, mse = r2score_result, mean_absolute_error_result, np.sqrt(
        mean_squared_error_result), mean_squared_error_result  # regression_evaluation(reg_results)

    results_dict = {}
    results_dict[final_output_dir.split('\\')[-1]] = {
        'class_IoU': str(IoU_array),
        'class_Precision': str(Precision),
        'class_Recall': str(Recall),
        'class_F1': str(F1),
        'reg_results': str({"r2": r2, "mae": mae, "rmse": rmse, "mse": mse}), }
    #final_output_dir
    print(results_dict)
    import json
    results_json = json.dumps(results_dict)
    json_file_path = final_output_dir.split('\\')[-1]+'_final_result.json'
    with open(json_file_path, 'w') as json_file:
        json_file.write(results_json)

    end = timeit.default_timer()
    print('Mins: %d' % int((end-start)/60))
    print('Done')


if __name__ == '__main__':
    main()
