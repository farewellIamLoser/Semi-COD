#!/usr/bin/python3
#coding=utf-8
import os
from functools import partial
import sys
import datetime
import os
import time
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import shutil
import utils.metrics as Measure
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from data import dataset
import logging as logger
from lib.data_prefetcher import DataPrefetcher
from SAMGeneration.image_select_test import samimage_select
from train_processes import *
from tools import *
from SAMGeneration.SAMSSIMGrad import process_image

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
    FM = Measure.Fmeasure()
    WFM = Measure.WeightedFmeasure()
    SM = Measure.Smeasure()
    EM = Measure.Emeasure()
    MAE = Measure.MAE()
    with torch.no_grad():
        for image, mask, shape, name in val_loader:
            image, mask = image.cuda().float(), mask.cuda().float()
            out= model(image, image, stage='eval')

            out = F.interpolate(out, size=shape, mode='bilinear', align_corners=False)
            pred = out.sigmoid().data.cpu().numpy().squeeze()
            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)  # 标准化处理,把数值范围控制到(0,1)
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

def copy_prompt():
    source_folder = r'/mnt/e/Mr.Wu/dataset/Semi-CodDataset/prompt/Prompt'
    destination_folder = '/mnt/e/Mr.Wu/dataset/Semi-CodDataset/stage3/Prompt'
    os.makedirs(destination_folder, exist_ok=True)

    # 获取源文件夹中所有图片文件的路径
    image_files = [f for f in os.listdir(source_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]

    # 遍历图片文件并复制到目标文件夹
    for image_file in image_files:
        source_path = os.path.join(source_folder, image_file)
        destination_path = os.path.join(destination_folder, image_file)
        shutil.copyfile(source_path, destination_path)
        # print(f"Copied {source_path} to {destination_path}")
root = '/mnt/e/Mr.Wu/dataset/Semi-CodDataset'
def prompt_generation(model, val_loader, i):
    pretrain = r'/mnt/e/Mr.Wu/codes/Semi-Supervised-Camouflaged-Object-Detection-with-Scribble-Annotations-main/out/trained/CamoMamba_396.pth'
    data_path = os.path.join(root)
    dict = torch.load(pretrain)
    model.load_state_dict(dict, strict=True)
    model.train(False)
    with torch.no_grad():
        for image, mask, shape, name in val_loader:
            image = image.cuda().float()
            out = model(image, image, stage='eval')
            out = F.interpolate(out, size=shape, mode='bilinear', align_corners=False)
            pred = out.sigmoid().data.cpu().numpy().squeeze()
            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)  # 标准化处理,把数值范围控制到(0,1)
            pred_image = Image.fromarray((pred * 255).astype(np.uint8))
            pred_image.save(name[0].replace('GT', 'Prompt'))
    model.train(True)
    copy_prompt()
    process_image(data_path)
    i = samimage_select(i)
    return i
def validate_multiloader(model, val_loader):
    maes = []
    for v in val_loader:
        st = time.time()
        mae = validate(model, v)
        maes.append(mae)
        print('Spent %.3fs, %s MAE: %s'%(time.time()-st, v.dataset.data_name, mae))
    return sum(maes)/len(maes)

BASE_LR = 1e-5
BASE_LR_SAM = 1e-5
MAX_LR = 5e-3
MAX_LR_SAM = 1e-4
EXP_NAME = '/mnt/e/Mr.Wu/codes/Weakly-Supervised-Camouflaged-Object-Detection-with-Scribble-Annotations-main/out\\trained' # change it in main
root = '/mnt/e/Mr.Wu/dataset/Semi-CodDataset'


def train(Dataset, net, cfg1, cfg2, cfg3, prompt, train_loss, start_from = 0):
    ## dataset
    stage1_data = Dataset.Data(cfg1)
    stage1_loader = DataLoader(stage1_data, batch_size=cfg1.batch, shuffle=True, num_workers=0, drop_last=True)

    stage2_data = Dataset.Data(cfg2)
    stage2_loader = DataLoader(stage2_data, batch_size=cfg2.batch, shuffle=True, num_workers=0, drop_last=True)

    prompt_data = Dataset.Data(prompt)
    prompt_loaders = DataLoader(prompt_data, batch_size=1, shuffle=False, num_workers=0)

    val_cfg1 = [Dataset.Config(datapath=f'{root}/TestDataset/{i}', mode='test') for i in ['CHAMELEON', 'CAMO', 'COD10K', 'N4CK']]
    val_data1 = [Dataset.Data(v) for v in val_cfg1]
    val_loaders1 = [DataLoader(v, batch_size=1, shuffle=False, num_workers=0) for v in val_data1]

    val_cfg2 = [Dataset.Config(datapath=f'{root}/TestDataset/{i}', mode='test') for i in ['CHAMELEON', 'CAMO', 'COD10K', 'N4CK']]
    val_data2 = [Dataset.Data(v) for v in val_cfg2]
    val_loaders2 = [DataLoader(v, batch_size=1, shuffle=False, num_workers=0) for v in val_data2]

    min_mae = 1.0
    best_epoch = 0
    ## network
    # print('model has {} parameters in total'.format(sum(x.numel() for x in net.parameters())))
    net.train(True)
    net.cuda()

    ## log
    sw1 = SummaryWriter(cfg1.savepath)
    db_size1 = len(stage1_loader)
    global_step1 = start_from * db_size1
    et1 = 0

    sw2 = SummaryWriter(cfg2.savepath)
    db_size2 = len(stage2_loader)
    global_step2 = start_from * db_size2
    et2 = 0

    i = 3

    # -------------------------- training ------------------------------------

    # mae = validate_multiloader(net, val_loader)
    # print(mae)
    total_epoch = cfg1.epoch + cfg2.epoch + cfg3.epoch
    for epoch in range(start_from, total_epoch):


        # optimizer setting
        if epoch == 0:
            ## parameter
            base, head = [], []
            for name, param in net.named_parameters():
                if 'encoder' in name:
                    base.append(param)
                else:
                    head.append(param)
            optimizer1 = torch.optim.SGD([{'params': base}, {'params': head}], lr=cfg1.lr, momentum=cfg1.momen,
                                         weight_decay=cfg1.decay, nesterov=True)
        if epoch >= (cfg1.epoch + cfg2.epoch) and epoch % 100 == 0:
            print('strengthen epoch',i)
            torch.cuda.empty_cache()
            i = prompt_generation(net, prompt_loaders, i)
            torch.cuda.empty_cache()
            stage3_data = Dataset.Data(cfg3)
            stage3_loader = DataLoader(stage3_data, batch_size=cfg3.batch, shuffle=True, num_workers=0, drop_last=True)
            sw3 = SummaryWriter(cfg3.savepath)
            db_size3 = len(stage3_loader)
            global_step3 = start_from * db_size3
            et3 = 0

            base, head = [], []
            for name, param in net.named_parameters():
                if 'encoder' in name:
                    base.append(param)
                else:
                    head.append(param)
            optimizer3 = torch.optim.SGD([{'params': base}, {'params': head}], lr=cfg3.lr, momentum=cfg3.momen,
                                         weight_decay=cfg3.decay, nesterov=True)

        # stage 1
        if epoch < cfg1.epoch:
            prefetcher = DataPrefetcher(stage1_loader, cfg1.mode)
            batch_idx1 = -1
            image, mask, label = prefetcher.next(cfg1.mode)
            while image is not None:

                st1 = time.time()
                niter1 = epoch * db_size1 + batch_idx1
                lr1, momentum1 = get_triangle_lr(BASE_LR, MAX_LR, cfg1.epoch * db_size1, niter1, ratio=1.)
                optimizer1.param_groups[0]['lr'] = 0.1 * lr1 # for backbone
                optimizer1.param_groups[1]['lr'] = lr1
                optimizer1.momentum = momentum1
                batch_idx1 += 1
                global_step1 += 1

                loss = train_loss([image, mask, label], cfg1.mode, net, dict(epoch=epoch+1, global_step=global_step1, sw=sw1, t_epo=cfg1.epoch))

                ######  objective function  ######
                optimizer1.zero_grad()
                loss.backward()
                optimizer1.step()
                sw1.add_scalar('lr', optimizer1.param_groups[0]['lr'], global_step=global_step1)
                sw1.add_scalar('loss', loss.item(), global_step=global_step1)

                image, mask, label = prefetcher.next(cfg1.mode)
                ta1 = time.time() - st1
                et1 = 0.9*et1 + 0.1 *ta1 if et1>0 else ta1
                if batch_idx1 % 10 == 0:
                    msg = '%s| %s | eta:%s | step:%d/%d/%d | lr=%.6f | loss=%.6f | bestepoch=%d' % (
                    TAG, datetime.datetime.now(), datetime.timedelta(seconds = int((cfg1.epoch*db_size1-niter1)*et1)), global_step1, epoch+1,
                    cfg1.epoch, optimizer1.param_groups[0]['lr'], loss.item(), best_epoch)
                    print(msg)
                    logger.info(msg)
            if (epoch + 1) % 50 == 0:
                torch.save(net.state_dict(), cfg3.savepath + '/model-new-' + str(epoch + 1))
                mae = validate_multiloader(net, val_loaders1)
                print('VAL MAE:%s' % (mae))
                sw1.add_scalar('val', mae, global_step=global_step1)
                if mae < min_mae:
                    min_mae = mae
                    best_epoch = epoch + 1
                    torch.save(net.state_dict(), cfg1.savepath + '/CamoMamba_396.pth')
                    # torch.save(optimizer1.state_dict(), cfg1.savepath + '/cnn_baseline_optimizer.pth')
                    print('best epoch is:%d, MAE:%s' % (best_epoch, min_mae))

        if epoch >= (cfg1.epoch + cfg2.epoch):
            prefetcher = DataPrefetcher(stage3_loader, cfg3.mode)
            batch_idx3 = -1
            image, mask = prefetcher.next(cfg3.mode)
            while image is not None:
                cfg3_iterate_epoch = 100
                st3 = time.time()
                niter3 = (epoch - (cfg1.epoch + cfg2.epoch)) % cfg3_iterate_epoch * db_size3 + batch_idx3
                lr3, momentum3 = get_triangle_lr(BASE_LR_SAM, MAX_LR_SAM, cfg3_iterate_epoch * db_size3, niter3, ratio=1.)
                optimizer3.param_groups[0]['lr'] = 0.1 * lr3  # for backbone
                optimizer3.param_groups[1]['lr'] = lr3
                optimizer3.momentum = momentum3
                batch_idx3 += 1
                global_step3 += 1

                loss = train_loss([image, mask], cfg3.mode, net,
                                  dict(epoch=epoch + 1, global_step=global_step3, sw=sw3, t_epo=cfg3.epoch))

                ######  objective function  ######
                optimizer3.zero_grad()
                loss.backward()
                optimizer3.step()
                sw3.add_scalar('lr', optimizer3.param_groups[0]['lr'], global_step=global_step3)
                sw3.add_scalar('loss', loss.item(), global_step=global_step3)

                image, mask = prefetcher.next(cfg3.mode)
                ta3 = time.time() - st3
                et3 = 0.9 * et3 + 0.1 * ta3 if et3 > 0 else ta3
                if batch_idx3 % 100 == 0:
                    msg = '%s| %s | eta:%s | step:%d/%d/%d | lr=%.6f | loss=%.6f | bestepoch=%d' % (
                        TAG, datetime.datetime.now(),
                        datetime.timedelta(seconds=int((cfg3.epoch * db_size3 - niter3) * et3)), global_step3,
                        epoch + 1,
                        cfg3.epoch, optimizer3.param_groups[0]['lr'], loss.item(), best_epoch)
                    print(msg)
                    logger.info(msg)
            if (epoch + 1) % 10 == 0:
                torch.save(net.state_dict(), cfg3.savepath + '/model-new-' + str(epoch + 1))
                mae = validate_multiloader(net, val_loaders1)
                print('VAL MAE:%s' % (mae))
                sw3.add_scalar('val', mae, global_step=global_step3)
                if mae < min_mae:
                    min_mae = mae
                    best_epoch = epoch + 1
                    torch.save(net.state_dict(), cfg3.savepath + '/CamoMamba_396.pth')
                    print('best epoch is:%d, MAE:%s' % (best_epoch, min_mae))

        # if epoch == cfg3.epoch-2 or epoch == cfg3.epoch-1 or (epoch+1) % 50 == 0:
        #     # torch.save(net.state_dict(), cfg3.savepath + '/600.pth')
        #     torch.save(net.state_dict(), cfg3.savepath + '/model-new-' + str(epoch + 1))
    print('min val mae for {} is {}'.format(EXP_NAME, min_mae))

if __name__=='__main__':
    def parse_args():
        parser = argparse.ArgumentParser("FSPNet-Transformer")
        parser.add_argument('--base_lr', default=(1e-4), type=float, help='learning rate')
        parser.add_argument('--batch_size', default=4, type=int, help='batch size per GPU')
        parser.add_argument("--resume", default=None)
        parser.add_argument('--gpu', default=1, type=int)
        parser.add_argument('--path', type=str, default=r'/mnt/e/Mr.Wu/dataset/CodDataset', help='path to train dataset')
        parser.add_argument('--pretrain', type=str,
                            default=r'/mnt/e/Mr.Wu/codes/FSPNet_Weak/checkpoint/Backbone/PVTv2/pvt_v2_b4.pth',
                            help='path to pretrain model')
        parser.add_argument('--ft_for_MoCA', default=None, type=str, help='path to pretrain model')

        # Model parameters
        parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                            help='Name of model to train')
        parser.add_argument('--input-size', default=320, type=int, help='images input size')

        parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                            help='Dropout rate (default: 0.)')
        parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                            help='Drop path rate (default: 0.1)')

        parser.add_argument('--model-ema', action='store_true')
        parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
        parser.set_defaults(model_ema=True)
        parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
        parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

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
    EXP_NAME = f'trained'
    stage1_epoch = 0
    stage2_epoch = 0
    stage3_epoch = 100

    cfg1 = dataset.Config(datapath=f'{root}', savepath=f'./out/{EXP_NAME}/', mode='stage1', batch=args.batch_size, lr=1e-3, momen=0.9, decay=5e-4, epoch=stage1_epoch, label_dir = 'Scribble')
    cfg2 = dataset.Config(datapath=f'{root}', savepath=f'./out/{EXP_NAME}/', mode='stage2', batch=args.batch_size, lr=1e-3, momen=0.9, decay=5e-4, epoch=stage2_epoch, label_dir = 'Scribble')
    cfg3 = dataset.Config(datapath=f'{root}', savepath=f'./out/{EXP_NAME}/', mode='stage3', batch=args.batch_size, lr=1e-3, momen=0.9, decay=5e-4, epoch=stage3_epoch, label_dir = 'Scribble')
    prompt = dataset.Config(datapath=f'{root}', mode='prompt')

    from model.CamoFormer import CamoFormer
    # from model.BlockBaseline import CamoFormer
    net = CamoFormer(cfg=None)
    if args.pretrain:
        encoder = torch.load(args.pretrain)
        net.encoder.load_state_dict(encoder, strict=False)
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
    tm = partial(train_loss, w_ft=w_ft, ft_st = ft_st, ft_fct=.5, ft_dct = dict(crtl_loss = False, w_ftp=w_ftp, norm=False, topk=topk, step_ratio=2), ft_head=False, mtrsf_prob=1, ops=[0,1,2], w_l2g=0.3, l_me=0.05, me_st=20, multi_sc=0)
    train(dataset, net, cfg1, cfg2, cfg3, prompt, tm, start_from=0)