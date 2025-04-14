#!/usr/bin/python3
#coding=utf-8

from functools import partial
import sys
import datetime
import os
import time
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from data import dataset 
import logging as logger
from lib.data_prefetcher import DataPrefetcher
import numpy as np
from train_processes import *
from tools import *
import utils.metrics as Measure
import cv2
import numpy as np

# 保存为图片
TAG = "scribblecod"
logger.basicConfig(level=logger.INFO, format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', \
                           filename="train_%s.log"%(TAG), filemode="w")

# import subprocess
# GPU_ID = subprocess.getoutput('nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader | nl -v 0 | sort -nrk 2 | cut -f 1| head -n 1 | xargs')
# os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

""" set lr """
def get_triangle_lr(base_lr, max_lr, total_steps, cur, ratio=1., \
        annealing_decay=1e-2, momentums=[0.95, 0.85]):
    first = int(total_steps*ratio)
    last  = total_steps - first
    min_lr = base_lr * annealing_decay

    cycle = np.floor(1 + cur/total_steps)
    x = np.abs(cur*2.0/total_steps - 2.0*cycle + 1)
    if cur < first:
        lr = base_lr + (max_lr - base_lr) * np.maximum(0., 1.0 - x)
    else:
        lr = ((base_lr - min_lr)*cur + min_lr*first - base_lr*total_steps)/(first - total_steps)
    if isinstance(momentums, int):
        momentum = momentums
    else:
        if cur < first:
            momentum = momentums[0] + (momentums[1] - momentums[0]) * np.maximum(0., 1.-x)
        else:
            momentum = momentums[0]

    return lr, momentum


def get_polylr(base_lr, last_epoch, num_steps, power):
    return base_lr * (1.0 - min(last_epoch, num_steps-1) / num_steps) **power


def validate(model, val_loader):
    model.train(False)
    avg_mae = 0.0
    cnt = 0
    FM = Measure.Fmeasure()
    WFM = Measure.WeightedFmeasure()
    SM = Measure.Smeasure()
    EM = Measure.Emeasure()
    MAE = Measure.MAE()
    with torch.no_grad():
        for image, mask, shape, name in val_loader:
            image, mask = image.cuda().float(), mask.cuda().float()
            out = model(image, image, stage='eval')
            out = F.interpolate(out, size=shape, mode='bilinear', align_corners=False)
            pred = out.sigmoid().data.cpu().numpy().squeeze()
            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)# 标准化处理,把数值范围控制到(0,1)
            folder = '/mnt/e/Mr.Wu/AllModelImage/camomamba/404'
            print(name)
            # cv2.imwrite(os.path.join(folder, name[0]), pred * 255)
            mask = mask.cpu().numpy().astype(np.float32).squeeze()
            mask /= (mask.max() + 1e-8)
            FM.step(pred=pred * 255, gt=mask * 255)
            WFM.step(pred=pred * 255, gt=mask * 255)
            SM.step(pred=pred * 255, gt=mask * 255)
            EM.step(pred=pred * 255, gt=mask * 255)
            MAE.step(pred=pred * 255, gt=mask * 255)
            fm = FM.get_results()["fm"]
            wfm = WFM.get_results()["wfm"]
            sm = SM.get_results()["sm"]
            em = EM.get_results()["em"]
            mae = MAE.get_results()["mae"]
        results = {
            "Smeasure": sm,
            "wFmeasure": wfm,
            "MAE": mae,
            "adpEm": em["adp"],
            "meanEm": em["curve"].mean(),
            "maxEm": em["curve"].max(),
            "adpFm": fm["adp"],
            "meanFm": fm["curve"].mean(),
            "maxFm": fm["curve"].max(),
        }
        print(results)
    model.train(True)
    return mae

def validate_multiloader(model, val_loader):
    maes = []
    for v in val_loader:
        st = time.time()
        mae = validate(model, v)
        maes.append(mae)
        print('Spent %.3fs, %s MAE: %s'%(time.time()-st, v.dataset.data_name, mae))
    return sum(maes)/len(maes)

BASE_LR = 1e-5
MAX_LR = 1e-2
total_epoch = 600
EXP_NAME = '/mnt/e/Mr.Wu/codes/Weakly-Supervised-Camouflaged-Object-Detection-with-Scribble-Annotations-main/out/trained' # change it in main
root = '/mnt/e/Mr.Wu/dataset/Semi-CodDataset'

def train(Dataset, net):
    ## dataset
    val_cfg = [Dataset.Config(datapath=f'{root}/TestDataset/{i}', mode='test') for i in ['CHAMELEON', 'CAMO', 'COD10K', 'N4CK']]
    val_data = [Dataset.Data(v) for v in val_cfg]
    val_loaders = [DataLoader(v, batch_size=1, shuffle=False, num_workers=0) for v in val_data]
    net.train(True)
    net.cuda()
    ## parameter
    base, head = [], []
    for name, param in net.named_parameters():
        if 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)

    # stage 1
    mae = validate_multiloader(net, val_loaders)
    print('VAL MAE:%s' % (mae))


if __name__=='__main__':
    def parse_args():
        parser = argparse.ArgumentParser("FSPNet-Transformer")
        parser.add_argument('--base_lr', default=(1e-4), type=float, help='learning rate')
        parser.add_argument('--batch_size', default=4, type=int, help='batch size per GPU')
        parser.add_argument("--resume", default=None)
        parser.add_argument('--gpu', default=1, type=int)
        parser.add_argument('--path', type=str, default=r'/mnt/e/Mr.Wu/dataset/CodDataset', help='path to train dataset')
        parser.add_argument('--pretrain', type=str,
                            default=r'/mnt/e/Mr.Wu/codes/Semi-Supervised-Camouflaged-Object-Detection-with-Scribble-Annotations-main/mm_ablation/404/camomamba_404.pth',
                            help='path to pretrain model')
        parser.add_argument('--ft_for_MoCA', default=None, type=str, help='path to pretrain model')

        # DDP configs:
        parser.add_argument('--world-size', default=1, type=int,
                            help='number of nodes for distributed training')
        parser.add_argument('--rank', default=0, type=int,
                            help='node rank for distributed training')
        parser.add_argument('--dist-url', default='env://', type=str,
                            help='url used to set up distributed training')
        parser.add_argument('--dist-backend', default='nccl', type=str,
                            help='distributed backend')
        parser.add_argument('--local_rank', default=0, type=int,
                            help='local rank for distributed training')
        args = parser.parse_args()
        return args


    args = parse_args()
    cfg = [.15, 1, 16, 1]
    w_ft, ft_st, topk,w_ftp = cfg
    cfg1 = dataset.Config(datapath=f'{root}', savepath=f'./out/{EXP_NAME}/', mode='stage1', batch=2, lr=1e-3, momen=0.9,
                          decay=5e-4, epoch=total_epoch, label_dir='Scribble')

    # from model.CamoFormer import CamoFormer
    from model.CamoFormer import CamoFormer
    # from model.BlockBaseline import CamoFormer
    net = CamoFormer(cfg=None)
    if args.pretrain:
        encoder = torch.load(args.pretrain)
        net.load_state_dict(encoder, strict=True)

    # net = FSPNet_model.Model(args.pretrain, img_size=384)
    args.distributed = args.world_size > 1
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        os.system("nvidia-smi")
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            net = net.cuda(args.gpu)
        else:
            net.cuda()
            net = torch.nn.parallel.DistributedDataParallel(net)
    else:
        net.cuda()
    encoder_param = []
    decoer_param = []
    for name, param in net.named_parameters():
        if "encoder" in name:
            encoder_param.append(param)
        else:
            decoer_param.append(param)
    tm = partial(train_loss, w_ft=w_ft, ft_st = ft_st, ft_fct=.5, ft_dct = dict(crtl_loss = False, w_ftp=w_ftp, norm=False, topk=topk, step_ratio=2), ft_head=False, mtrsf_prob=1, ops=[0,1,2], w_l2g=0.3, l_me=0.05, me_st=20, multi_sc=0)
    train(dataset, net)