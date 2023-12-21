# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import logging
import os
import time
import numpy as np
import numpy.ma as ma
from tqdm import tqdm
from sklearn.metrics import confusion_matrix as cul_confusion_matrix
from torch.optim.lr_scheduler import LambdaLR
import torch
import torch.nn as nn
from torch.nn import functional as F
import timeit
from utils.utils import AverageMeter
from utils.utils import get_confusion_matrix, reg_prepare, regression_evaluation
from utils.utils import adjust_learning_rate, visualize_all_pred
import torchmetrics
import utils.distributed as dist
EPS = 1e-32
find_bug=False
#find_bug=True
def reduce_tensor(inp, average=True):
    """
    Reduce the loss from all processes so that 
    process with rank 0 has the averaged results.
    """
    world_size = dist.get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp.clone().detach_()
        #torch.distributed.reduce(reduced_inp, dst=0)
        torch.distributed.all_reduce(reduced_inp,
                                     torch.distributed.ReduceOp.SUM)
        if average:
           reduced_inp= reduced_inp / world_size
    return reduced_inp

def gather_tensor(inp):
    world_size = dist.get_world_size()
    if world_size < 2:
        return inp
    tensor_list = [torch.zeros_like(inp) for _ in range(world_size)]
    reduced_inp = inp.clone().detach_()
    torch.distributed.all_gather(tensor_list, reduced_inp)
    return tensor_list


def train(config, epoch, num_epoch, epoch_iters, base_lr,
          num_iters, trainloader, optimizer, model, writer_dict):
    # Training
    model.train()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    if dist.get_rank() == 0:
        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']
        img_list_all=[]
        feature_list_all=[]

    #keep_epoch = int(num_epoch*0.1)
    #lr_lambda = lambda epoch: 1.0 if epoch < keep_epoch else np.math.exp(0.1 * (keep_epoch - epoch))
    #scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)
    for i_iter, batch in enumerate(trainloader, 0):
        if find_bug:
            if i_iter>10:continue
        images, label, _, _ = batch
        images = {key: images[key].cuda() for key in images}
        label=label.cuda()
        label2 = label.clone().detach().cpu().numpy()
        #print("train_np.unique(label)", np.unique(label2))

        losses, pred = model(images, label, epoch)

        loss = losses.mean()

        if dist.is_distributed():
            reduced_loss = reduce_tensor(loss)
        else:
            reduced_loss = loss
        model.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(reduced_loss.item())

    




        if i_iter % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {}, Loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters,
                      batch_time.average(), [x['lr'] for x in optimizer.param_groups], ave_loss.average())
            logging.info(msg)
            """
            if dist.get_rank() == 0:
                img_list, feature_list = visualize_all_pred([images[key] for key in images],
                                                            [label, pred[0].argmax(dim=1, keepdim=True), pred[1],
                                                             (pred[1] >= 0.1).int()])
                img_list_all.append(img_list)
                feature_list_all.append(feature_list)
            """
    #scheduler.step()
    lr = adjust_learning_rate(optimizer, base_lr, epoch, num_epoch)
    if dist.get_rank() == 0:
        #for i in range(len(img_list_all)):
            # print("img_list_all[i]",img_list_all[i].shape)
            #writer.add_image('img_trainl_sample_{}'.format(i), img_list_all[i].transpose(2, 0, 1), global_steps)
            # print("feature_list_all[i]",feature_list_all[i].shape)
            #writer.add_image('feature_train_sample_{}'.format(i), feature_list_all[i].transpose(2, 0, 1), global_steps)
        writer.add_scalar('train_loss', ave_loss.average(), global_steps)
        writer.add_scalar('train_lr', optimizer.param_groups[-1]["lr"], global_steps)
        writer_dict['train_global_steps'] = global_steps + 1





def validate(config, epoch, testloader, model, writer_dict, optimizer, results_dict):
    
    model.eval()
    ave_loss = AverageMeter()
    #nums = config.MODEL.NUM_OUTPUTS
    final_class=config.DATASET.NUM_CLASSES
    #final_class=2
    #confusion_matrix = np.zeros( (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    #confusion_matrix = np.zeros( (final_class, final_class))
    #reg_results = []
    img_list_all=[] 
    feature_list_all=[]

    from torchmetrics.classification import BinaryConfusionMatrix
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bcm = BinaryConfusionMatrix().to(device)
    from torchmetrics.regression import R2Score
    r2score = R2Score().to(device)
    #from torchmetrics.regression import RelativeSquaredError
    #relative_squared_error = RelativeSquaredError().to(device)
    from torchmetrics.regression import MeanSquaredError
    mean_squared_error = MeanSquaredError().to(device)
    from torchmetrics.regression import MeanAbsoluteError
    mean_absolute_error = MeanAbsoluteError().to(device)


    #r2score(preds, target)

    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            if find_bug:
                if idx > 10: continue
            images, label, _, _ = batch
            size = label.size()

            images = {key: images[key].cuda() for key in images}
            label=label.cuda()

            losses, pred = model(images, label)

            if not isinstance(pred, (list, tuple)):
                pred = [pred]

            if config.MODEL.MICCR:
                pred_ = pred[0]
            elif config.MODEL.MI_cul:
                pred_= pred[-2]
                #print("config.MODEL.MI_cul", config.MODEL.MI_cul)
            else:
                pred_= pred[-1]

            if dist.is_distributed():
                logits_gather_list = [torch.zeros_like(label) for _ in range(dist.get_world_size())]
                torch.distributed.all_gather(logits_gather_list, label)
                label = torch.cat(logits_gather_list, dim=0)

                targets_gather_list = [torch.zeros_like(pred_) for _ in range(dist.get_world_size())]
                torch.distributed.all_gather(targets_gather_list, pred_)
                pred_ = torch.cat(targets_gather_list, dim=0)

            if dist.get_rank() == 0:
                #label_cls=labels[0].clone().detach().cpu().numpy()
                label_cls = (label >= 0.1).int()
                pred_cls = (pred_ >= 0.06).int()
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

                #label_cls = label_cls.detach().cpu().numpy()
                #pred_cls=pred_cls.detach().cpu().numpy()
                #tem_confusion_matrix=cul_confusion_matrix(label_cls.reshape(-1), pred_cls.reshape(-1),)
                #confusion_matrix += tem_confusion_matrix

                #pred_res = pred_.clone().detach().cpu().numpy()
                #label_res=label.clone().detach().cpu().numpy()
                #reg_results.append(np.stack(reg_prepare(pred_res, label_res), axis=1).astype(np.float32))


                if idx % config.PRINT_FREQ == 0:
                    msg = "Start to evaluate (total_samples={}, total_steps={})...".format(idx, len(testloader))
                    logging.info(msg)
                """
                if idx % config.PRINT_FREQ*4 == 0:
                    img_list, feature_list=visualize_all_pred([images[key] for key in images], [label, pred[0].argmax(dim=1, keepdim=True) , pred[1], (pred_ >= 0.1).int()])
                    img_list_all.append(img_list)
                    feature_list_all.append(feature_list)
                """

            loss = losses.mean()
            if dist.is_distributed():
                reduced_loss = reduce_tensor(loss)
            else:
                reduced_loss = loss
            ave_loss.update(reduced_loss.item())


            bcm_result = bcm.compute().cpu().numpy()
            r2score_result = r2score.compute().cpu().numpy()
            mean_squared_error_result = mean_squared_error.compute().cpu().numpy()
            mean_absolute_error_result = mean_absolute_error.compute().cpu().numpy()

        confusion_matrix=bcm_result
        mean_IoU=0
        IoU_array=0
        OA=0
        r2=0
        rmse=1
        if dist.get_rank() == 0:
            #reg_results = np.concatenate(reg_results, axis=0)

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
                F1[i] = 2 * (Precision[i] * Recall[i]) / (Precision[i] + Recall[i] + EPS )


            r2, mae, rmse, mse=r2score_result, mean_absolute_error_result, np.sqrt(mean_squared_error_result), mean_squared_error_result#regression_evaluation(reg_results)

            #if dist.get_rank() <= 0:
            logging.info('confusion_matrix{},'.format( confusion_matrix))
            logging.info('IoU_array:{},mean_IoU:{}, OA:{}, '.format(IoU_array, mean_IoU, OA))
            logging.info('class_Precision:{}, class_Recall:{}, class_F1:{}, '.format(Precision, Recall, F1))
            logging.info('r2:{}, mae:{}, rmse:{}, mse:{},'.format( r2, mae, rmse, mse))

            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
            writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
            writer.add_scalar('OA', OA, global_steps)

            for i in range(len(IoU_array)):
                writer.add_scalar('class_{}_IoU'.format(i), IoU_array[i], global_steps)
                writer.add_scalar('class_{}_Precision'.format(i), Precision[i], global_steps)
                writer.add_scalar('class_{}_Recall'.format(i), Recall[i], global_steps)
                writer.add_scalar('class_{}_F1'.format(i), F1[i], global_steps)

                writer.add_scalar('r2', r2, global_steps)
                writer.add_scalar('mae', mae, global_steps)
                writer.add_scalar('rmse', rmse, global_steps)
                writer.add_scalar('mse', mse, global_steps)
            """
            for i in range(len(img_list_all)):
                #print("img_list_all[i]",img_list_all[i].shape)
                writer.add_image('img_val_sample_{}'.format(i), img_list_all[i].transpose(2, 0, 1), global_steps)
                #print("feature_list_all[i]",feature_list_all[i].shape)
                writer.add_image('feature_val_sample_{}'.format(i), feature_list_all[i].transpose(2, 0, 1), global_steps)
            """
            writer_dict['valid_global_steps'] = global_steps + 1


        if dist.get_rank() == 0:
            results_dict[epoch] = {
                'class_IoU': str(IoU_array),
                'class_Precision': str(Precision),
                'class_Recall': str(Recall),
                'class_F1': str(F1),
                'reg_results': str({"r2": r2, "mae": mae, "rmse": rmse, "mse": mse}),
                "lr": str(optimizer.param_groups[-1]["lr"]),
                "val_loss": str(ave_loss.average()),}
    return ave_loss.average(), mean_IoU, IoU_array, OA, r2, rmse



def testval(config, test_dataset, testloader, model, sv_dir='', sv_pred=False):
    model.eval()
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, name, *border_padding = batch
            size = label.size()
            pred = test_dataset.multi_scale_inference(
                config,
                model,
                image,
                scales=config.TEST.SCALE_LIST,
                flip=config.TEST.FLIP_TEST)

            if len(border_padding) > 0:
                border_padding = border_padding[0]
                pred = pred[:, :, 0:pred.size(2) - border_padding[0], 0:pred.size(3) - border_padding[1]]

            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)

            if sv_pred:
                sv_path = os.path.join(sv_dir, 'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)

            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                logging.info('mIoU: %.4f' % (mean_IoU))

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum()/pos.sum()
    mean_acc = (tp/np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    return mean_IoU, IoU_array, pixel_acc, mean_acc


def test(config, test_dataset, testloader, model,
         sv_dir='', sv_pred=True):
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, size, name = batch
            size = size[0]
            pred = test_dataset.multi_scale_inference(
                config,
                model,
                image,
                scales=config.TEST.SCALE_LIST,
                flip=config.TEST.FLIP_TEST)

            if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

            if sv_pred:
                sv_path = os.path.join(sv_dir, 'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)
