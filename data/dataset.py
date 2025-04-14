#!/usr/bin/python3
#coding=utf-8

import os
import os.path as osp
import random

import cv2
import torch
import numpy as np
try:
    from . import transform
except:
    import transform

from torch.utils.data import Dataset, DataLoader
from lib.data_prefetcher import DataPrefetcher

class Config(object):
    def __init__(self, **kwargs):
        if kwargs.get('label_dir') is None:
            kwargs['label_dir'] = 'Scribble'
        self.kwargs    = kwargs
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s'%(k, v))

        if 'ECSSD' in self.kwargs['datapath']:
            self.mean      = np.array([[[117.15, 112.48, 92.86]]])
            self.std       = np.array([[[ 56.36,  53.82, 54.23]]])
        elif 'DUTS' in self.kwargs['datapath']:
            self.mean      = np.array([[[124.55, 118.90, 102.94]]])
            self.std       = np.array([[[ 56.77,  55.97,  57.50]]])
        elif 'DUT-OMRON' in self.kwargs['datapath']:
            self.mean      = np.array([[[120.61, 121.86, 114.92]]])
            self.std       = np.array([[[ 58.10,  57.16,  61.09]]])
        elif 'MSRA-10K' in self.kwargs['datapath']:
            self.mean      = np.array([[[115.57, 110.48, 100.00]]])
            self.std       = np.array([[[ 57.55,  54.89,  55.30]]])
        elif 'MSRA-B' in self.kwargs['datapath']:
            self.mean      = np.array([[[114.87, 110.47,  95.76]]])
            self.std       = np.array([[[ 58.12,  55.30,  55.82]]])
        elif 'SED2' in self.kwargs['datapath']:
            self.mean      = np.array([[[126.34, 133.87, 133.72]]])
            self.std       = np.array([[[ 45.88,  45.59,  48.13]]])
        elif 'PASCAL-S' in self.kwargs['datapath']:
            self.mean      = np.array([[[117.02, 112.75, 102.48]]])
            self.std       = np.array([[[ 59.81,  58.96,  60.44]]])
        elif 'HKU-IS' in self.kwargs['datapath']:
            self.mean      = np.array([[[123.58, 121.69, 104.22]]])
            self.std       = np.array([[[ 55.40,  53.55,  55.19]]])
        elif 'SOD' in self.kwargs['datapath']:
            self.mean      = np.array([[[109.91, 112.13,  93.90]]])
            self.std       = np.array([[[ 53.29,  50.45,  48.06]]])
        elif 'THUR15K' in self.kwargs['datapath']:
            self.mean      = np.array([[[122.60, 120.28, 104.46]]])
            self.std       = np.array([[[ 55.99,  55.39,  56.97]]])
        elif 'SOC' in self.kwargs['datapath']:
            self.mean      = np.array([[[120.48, 111.78, 101.27]]])
            self.std       = np.array([[[ 58.51,  56.73,  56.38]]])
        else:
            #raise ValueError
            self.mean = np.array([[[0.485*256, 0.456*256, 0.406*256]]])
            self.std = np.array([[[0.229*256, 0.224*256, 0.225*256]]])
            # self.std, self.mean = np.array([0.1861761914527739, 0.19748777412623036, 0.2032849354904543])[None,None]*255, np.array([0.3320486163733052, 0.432231354815684, 0.449829585669272])[None,None]*255

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


class Data(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_name = cfg.datapath.split('/')[-1]

        if cfg.mode == 'stage1':
            with open(cfg.datapath + '/' + cfg.mode + '.txt', 'r') as lines:
                self.samples = []
                for line in lines:
                    imagepath = cfg.datapath +'/'+cfg.mode+ '/Image/' + line.strip() + '.jpg'
                    maskpath  = cfg.datapath + '/'+cfg.mode+f'/{cfg.label_dir}/'  + line.strip() + '.png'
                    self.samples.append([imagepath, maskpath])
        elif cfg.mode == 'stage2':
            with open(cfg.datapath + '/' + cfg.mode + '.txt', 'r') as lines:
                self.samples = []
                for line in lines:
                    s_lines = line.split()
                    SimagepathS = []
                    SmaskpathS = []
                    for s_line in s_lines:
                        Simagepath = cfg.datapath + '/' + 'stage2' + '/Image/' + s_line.strip() + '.jpg'
                        Smaskpath = cfg.datapath + '/' + 'stage2' + f'/{cfg.label_dir}/' + s_line.strip() + '.png'
                        SimagepathS.append(Simagepath)
                        SmaskpathS.append(Smaskpath)
                    self.samples.append([SimagepathS, SmaskpathS])
        elif cfg.mode == 'stage2-other':
            with open(cfg.datapath + '/' + 'stage1' + '.txt', 'r') as file1, open(cfg.datapath + '/' + cfg.mode + '.txt', 'r') as file2:
                self.samples = []
                for line1, line2 in zip(file1, file2):
                    Oimagepath = cfg.datapath + '/' + 'stage1' + '/Image/' + line1.strip() + '.jpg'
                    Omaskpath = cfg.datapath + '/' + 'stage1' + f'/{cfg.label_dir}/' + line1.strip() + '.png'
                    Simagepath = cfg.datapath + '/' + 'stage2' + '/Image/' + line2.strip() + '.jpg'
                    Smaskpath = cfg.datapath + '/' + 'stage2' + f'/{cfg.label_dir}/' + line2.strip() + '.png'
                    self.samples.append([Oimagepath, Omaskpath, Simagepath, Smaskpath])
        elif cfg.mode == 'wait_finish':
            with open(cfg.datapath + '/' + 'stage1' + '.txt', 'r') as file1, open(cfg.datapath + '/' + cfg.mode + '.txt', 'r') as file2:
                self.samples = []
                for line1, line2 in zip(file1, file2):
                    Oimagepath = cfg.datapath + '/' + 'stage1' + '/Image/' + line1.strip() + '.jpg'
                    Omaskpath = cfg.datapath + '/' + 'stage1' + f'/{cfg.label_dir}/' + line1.strip() + '.png'
                    Simagepath = cfg.datapath + '/' + 'stage2' + '/Image/' + line2.strip() + '.jpg'
                    Smaskpath = cfg.datapath + '/' + 'stage2' + f'/{cfg.label_dir}/' + line2.strip() + '.png'
                    self.samples.append([Oimagepath, Omaskpath, Simagepath, Smaskpath])
        elif cfg.mode == 'stage3':
            with open(cfg.datapath + '/' + cfg.mode + '.txt', 'r') as lines:
                self.samples = []
                for line in lines:
                    imagepath = cfg.datapath + '/' + cfg.mode + '/Image/' + line.strip() + '.jpg'
                    maskpath = cfg.datapath + '/' + cfg.mode + f'/{cfg.label_dir}/' + line.strip() + '.png'
                    self.samples.append([imagepath, maskpath])
        elif cfg.mode == 'prompt':
            with open(cfg.datapath + '/' + cfg.mode + '.txt', 'r') as lines:
                self.samples = []
                for line in lines:
                    imagepath = cfg.datapath + '/' + cfg.mode + '/Image/' + line.strip() + '.jpg'
                    maskpath = cfg.datapath + '/' + cfg.mode + '/GT/' + line.strip() + '.png'
                    self.samples.append([imagepath, maskpath])
        else:
            with open(cfg.datapath + '/' + cfg.mode + '.txt', 'r') as lines:
                self.samples = []
                for line in lines:
                    index = cfg.datapath.rfind("/")
                    if index != -1:
                      result = cfg.datapath[:index]
                      dir = cfg.datapath[index+1:]
                    else:
                      result = cfg.datapath
                    lindex = line.strip().find('/')
                    if lindex != -1:
                        lresult = line.strip()[:lindex]
                    else:
                        lresult = line.strip()
                    if dir == lresult:
                        imagepath = result + "/" + line.strip() + '.jpg'
                        maskpath  = result + "/" + line.strip() + '.png'
                        maskpath = maskpath.replace("Image", "GT")
                        self.samples.append([imagepath, maskpath])
        image_size = 320
        if cfg.mode == 'train':
            self.Mutitransform = transform.Compose(transform.Resize(image_size, image_size),
                                                   transform.RandomHorizontalFlip(),
                                                   transform.RandomCrop(image_size, image_size))
            self.transform = transform.Compose(transform.Normalize(mean=cfg.mean, std=cfg.std),
                                                    transform.Resize(image_size, image_size),
                                                    transform.RandomHorizontalFlip(),
                                                    transform.RandomCrop(image_size, image_size),
                                                    transform.ToTensor())
        elif cfg.mode == 'pretrain':
            self.Mutitransform = transform.Compose(transform.Resize(image_size, image_size),
                                                   transform.RandomHorizontalFlip(),
                                                   transform.RandomCrop(image_size, image_size))
            self.transform = transform.Compose(transform.Normalize(mean=cfg.mean, std=cfg.std),
                                               transform.Resize(image_size, image_size),
                                               transform.RandomHorizontalFlip(),
                                               transform.RandomCrop(image_size, image_size),
                                               transform.ToTensor())
        elif cfg.mode == 'stage1':
            self.Mutitransform = transform.Compose(transform.Resize(image_size, image_size),
                                                   transform.RandomHorizontalFlip(),
                                                   transform.RandomCrop(image_size, image_size))
            self.transform = transform.Compose(transform.Normalize(mean=cfg.mean, std=cfg.std),
                                               transform.Resize(image_size, image_size),
                                               transform.RandomHorizontalFlip(),
                                               transform.RandomCrop(image_size, image_size),
                                               transform.ToTensor())
        elif cfg.mode == 'stage2':
            self.Mutitransform = transform.Compose(transform.Resize(image_size, image_size),
                                                   transform.RandomHorizontalFlip(),
                                                   transform.RandomCrop(image_size, image_size))
            self.transform = transform.Compose(transform.Normalize(mean=cfg.mean, std=cfg.std),
                                               transform.Resize(image_size, image_size),
                                               transform.ToTensor())
        elif cfg.mode == 'stage3':
            self.Mutitransform = transform.Compose(transform.Resize(image_size, image_size),
                                                   transform.RandomHorizontalFlip(),
                                                   transform.RandomCrop(image_size, image_size))
            self.transform = transform.Compose(transform.Normalize(mean=cfg.mean, std=cfg.std),
                                               transform.Resize(image_size, image_size),
                                               transform.RandomHorizontalFlip(),
                                               transform.RandomCrop(image_size, image_size),
                                               transform.ToTensor())
        elif cfg.mode == 'prompt':
            self.transform = transform.Compose(transform.Normalize(mean=cfg.mean, std=cfg.std),
                                               transform.Resize(image_size, image_size),
                                               transform.ToTensor()
                                               )
        elif cfg.mode == 'test':
            self.transform = transform.Compose(transform.Normalize(mean=cfg.mean, std=cfg.std),
                                                    transform.Resize(image_size, image_size),
                                                    transform.ToTensor()
                                                )
        elif cfg.mode == 'CAMO':
            self.transform = transform.Compose(transform.Normalize(mean=cfg.mean, std=cfg.std),
                                           transform.Resize(image_size, image_size),
                                           transform.ToTensor()
                                           )
        elif cfg.mode == 'CHAMELEON':
            self.transform = transform.Compose(transform.Normalize(mean=cfg.mean, std=cfg.std),
                                                    transform.Resize(image_size, image_size),
                                                    transform.ToTensor()
                                                )
        elif cfg.mode == 'COD10K':
            self.transform = transform.Compose(transform.Normalize(mean=cfg.mean, std=cfg.std),
                                                    transform.Resize(image_size, image_size),
                                                    transform.ToTensor()
                                                )
        elif cfg.mode == 'N4CK':
            self.transform = transform.Compose(transform.Normalize(mean=cfg.mean, std=cfg.std),
                                               transform.Resize(image_size, image_size),
                                               transform.ToTensor()
                                               )
        else:
            raise ValueError

    def __getitem__(self, idx):
        pick_num = 3
        num_classes = 69
        # if self.cfg.mode == 'train':
        #     # in this part 255 is the front image
        #     random_int = random.randint(1, 2)
        #     if random_int == 1:
        #         images = []
        #         masks = []
        #         for i in range(0, 3):
        #             sampleMulti = random.randint(1, 69)
        #             imagepath, maskpath = self.samples[sampleMulti]
        #             image_temp = cv2.imread(imagepath).astype(np.float32)[:, :, ::-1]
        #             mask_temp = cv2.imread(maskpath).astype(np.float32)[:, :, ::-1]
        #             image_temp, mask_temp = self.transform(image_temp, mask_temp)
        #             images.append(image_temp)
        #             masks.append(mask_temp)
        #
        #     image, mask = self.transform(image, mask)
        #     if random_int == 1:
        #         top = torch.cat((image, images[0]), dim=2)
        #         bottom = torch.cat((image[1], images[2]), dim=2)
        #         image = torch.cat((top, bottom), dim=1)
        #
        #         mask_top = torch.cat((mask, masks[0]), dim=2)
        #         mask_bottom = torch.cat((masks[1], masks[2]), dim=2)
        #         mask = torch.cat((mask_top, mask_bottom), dim=1)
        #         image, mask = self.transform(image, mask)
        #     mask[mask == 0.] = 255.
        #     mask[mask == 2.] = 0.
        if self.cfg.mode == 'stage1':

            imagepath, maskpath = self.samples[idx]
            image = cv2.imread(imagepath).astype(np.float32)[:, :, ::-1]
            mask = cv2.imread(maskpath).astype(np.float32)[:, :, ::-1]
            H, W, C = mask.shape

            image, mask = self.Mutitransform(image, mask)
            image, mask  = self.transform(image, mask)
            return image, mask, (H, W), maskpath.split('/')[-1]

        elif self.cfg.mode == 'stage2':
            imagepath, maskpath = self.samples[idx]
            image = cv2.imread(imagepath[0]).astype(np.float32)[:, :, ::-1]
            mask = cv2.imread(maskpath[0]).astype(np.float32)[:, :, ::-1]
            ref_num = random.randint(1, len(imagepath)-1)
            ref_image = cv2.imread(imagepath[ref_num]).astype(np.float32)[:, :, ::-1]
            ref_mask = cv2.imread(maskpath[ref_num]).astype(np.float32)[:, :, ::-1]
            H, W, C = mask.shape

            cates = []
            parts = os.path.basename(imagepath[0]).split('-')
            cate_label = int(parts[4]) - 1
            cates.append(cate_label)

            # in this part 255 is the front image
            random_int = random.randint(1, 2)
            if random_int == 1:
                images = []
                masks = []
                ref_images = []
                ref_masks = []
                for i in range(0, 3):
                    sampleMulti = random.randint(0, len(self.samples) - 1)

                    imagepath, maskpath = self.samples[sampleMulti]

                    parts = os.path.basename(imagepath[0]).split('-')
                    cate_label = int(parts[4]) - 1

                    image_temp = cv2.imread(imagepath[0]).astype(np.float32)[:, :, ::-1]
                    mask_temp = cv2.imread(maskpath[0]).astype(np.float32)[:, :, ::-1]
                    ref_num = random.randint(1, len(imagepath) - 1)
                    ref_image_temp = cv2.imread(imagepath[ref_num]).astype(np.float32)[:, :, ::-1]
                    ref_mask_temp = cv2.imread(maskpath[ref_num]).astype(np.float32)[:, :, ::-1]
                    image_temp, mask_temp = self.Mutitransform(image_temp, mask_temp)
                    ref_image_temp, ref_mask_temp = self.Mutitransform(ref_image_temp, ref_mask_temp)
                    images.append(image_temp)
                    masks.append(mask_temp)
                    ref_images.append(ref_image_temp)
                    ref_masks.append(ref_mask_temp)
                    cates.append(cate_label)

            if len(cates) == 1:
                one_hot_labels = torch.zeros(4, num_classes, dtype=torch.float)
                for i in range(4):
                    one_hot_labels[i, cates[0]] = 1
            else:
                one_hot_labels = torch.zeros(4, num_classes, dtype=torch.float)
                for i, idx in enumerate(cates):
                    one_hot_labels[i, idx] = 1

            image, mask = self.Mutitransform(image, mask)
            ref_image, ref_mask = self.Mutitransform(ref_image, ref_mask)

            if random_int == 1:
                image = np.concatenate([
                    np.concatenate([image, images[0]], axis=1),  # 沿着列（宽度）拼接
                    np.concatenate([images[1], images[2]], axis=1)  # 沿着列（宽度）拼接
                ], axis=0)

                mask = np.concatenate([
                    np.concatenate([mask, masks[0]], axis=1),  # 沿着列（宽度）拼接
                    np.concatenate([masks[1], masks[2]], axis=1)  # 沿着列（宽度）拼接
                ], axis=0)

                ref_image = np.concatenate([
                    np.concatenate([ref_image, ref_images[0]], axis=1),  # 沿着列（宽度）拼接
                    np.concatenate([ref_images[1], ref_images[2]], axis=1)  # 沿着列（宽度）拼接
                ], axis=0)

                ref_mask = np.concatenate([
                    np.concatenate([ref_mask, ref_masks[0]], axis=1),  # 沿着列（宽度）拼接
                    np.concatenate([ref_masks[1], ref_masks[2]], axis=1)  # 沿着列（宽度）拼接
                ], axis=0)
                image, mask = self.transform(image, mask)
                ref_image, ref_mask = self.transform(ref_image, ref_mask)
            else:
                image, mask = self.transform(image, mask)
                ref_image, ref_mask = self.transform(ref_image, ref_mask)
            return image, mask, ref_image, ref_mask, one_hot_labels, (H, W), maskpath[0].split('/')[-1]

        elif self.cfg.mode == 'stage2-other':
            sample_cate = (idx // 3) * 3
            random_int1 = random.randint(0, 2) + sample_cate
            random_int2 = random.randint(0, 2) + sample_cate
            Oimagepath, Omaskpath, _, _ = self.samples[random_int1]
            _, _, Simagepath, Smaskpath = self.samples[random_int2]
            Oimage = cv2.imread(Oimagepath).astype(np.float32)[:, :, ::-1]
            Omask = cv2.imread(Omaskpath).astype(np.float32)[:, :, ::-1]
            Simage = cv2.imread(Simagepath).astype(np.float32)[:, :, ::-1]
            Smask = cv2.imread(Smaskpath).astype(np.float32)[:, :, ::-1]
            H, W, C = Omask.shape
            # in this part 255 is the front image

            cates = []
            parts = os.path.basename(Oimagepath).split('-')
            cate_label = int(parts[4])-1
            cates.append(cate_label)

            random_int = random.randint(1, 2)
            if random_int <= 1:
                Oimages = []
                Omasks = []
                Simages = []
                Smasks = []
                for i in range(0, 3):
                    sampleMulti = random.randint(0, len(self.samples)-1)
                    sample_cate = (sampleMulti // 3) * 3
                    random_int1 = random.randint(0, 2) + sample_cate
                    random_int2 = random.randint(0, 2) + sample_cate
                    Oimagepath, Omaskpath, _, _ = self.samples[random_int1]
                    _, _, Simagepath, Smaskpath = self.samples[random_int2]
                    parts = os.path.basename(Oimagepath).split('-')
                    cate_label = int(parts[4])-1
                    Oimage_temp = cv2.imread(Oimagepath).astype(np.float32)[:, :, ::-1]
                    Omask_temp = cv2.imread(Omaskpath).astype(np.float32)[:, :, ::-1]
                    Simage_temp = cv2.imread(Simagepath).astype(np.float32)[:, :, ::-1]
                    Smask_temp = cv2.imread(Smaskpath).astype(np.float32)[:, :, ::-1]
                    Oimage_temp, Omask_temp = self.Mutitransform(Oimage_temp, Omask_temp)
                    Simage_temp, Smask_temp = self.Mutitransform(Simage_temp, Smask_temp)
                    Oimages.append(Oimage_temp)
                    Omasks.append(Omask_temp)
                    Simages.append(Simage_temp)
                    Smasks.append(Smask_temp)
                    cates.append(cate_label)

            if len(cates) == 1:
                one_hot_labels = torch.zeros(4, num_classes, dtype=torch.float)
                for i in range(4):
                    one_hot_labels[i, cates[0]] = 1
            else:
                one_hot_labels = torch.zeros(4, num_classes, dtype=torch.float)
                for i, idx in enumerate(cates):
                    one_hot_labels[i, idx] = 1

            Oimage, Omask = self.Mutitransform(Oimage, Omask)
            Simage, Smask = self.Mutitransform(Simage, Smask)

            if random_int <= 1:
                Oimage = np.concatenate([
                    np.concatenate([Oimage, Oimages[0]], axis=1),  # 沿着列（宽度）拼接
                    np.concatenate([Oimages[1], Oimages[2]], axis=1)  # 沿着列（宽度）拼接
                ], axis=0)

                Omask = np.concatenate([
                    np.concatenate([Omask, Omasks[0]], axis=1),  # 沿着列（宽度）拼接
                    np.concatenate([Omasks[1], Omasks[2]], axis=1)  # 沿着列（宽度）拼接
                ], axis=0)

                Simage = np.concatenate([
                    np.concatenate([Simage, Simages[0]], axis=1),  # 沿着列（宽度）拼接
                    np.concatenate([Simages[1], Simages[2]], axis=1)  # 沿着列（宽度）拼接
                ], axis=0)

                Smask = np.concatenate([
                    np.concatenate([Smask, Smasks[0]], axis=1),  # 沿着列（宽度）拼接
                    np.concatenate([Smasks[1], Smasks[2]], axis=1)  # 沿着列（宽度）拼接
                ], axis=0)

                Oimage, Omask = self.transform(Oimage, Omask)
                Simage, Smask = self.transform(Simage, Smask)
            else:
                Oimage, Omask = self.transform(Oimage, Omask)
                Simage, Smask = self.transform(Simage, Smask)
            return Oimage, Omask, Simage, Smask, one_hot_labels, (H, W), Omaskpath.split('/')[-1]
        elif self.cfg.mode == 'stage3':
            imagepath, maskpath = self.samples[idx]
            image = cv2.imread(imagepath).astype(np.float32)[:, :, ::-1]
            mask = cv2.imread(maskpath).astype(np.float32)[:, :, ::-1]
            H, W, C = mask.shape

            image, mask = self.Mutitransform(image, mask)
            image, mask = self.transform(image, mask)
            return image, mask, (H, W), maskpath.split('/')[-1]
        elif self.cfg.mode == 'prompt':
            imagepath, maskpath = self.samples[idx]
            image = cv2.imread(imagepath).astype(np.float32)[:, :, ::-1]
            mask = cv2.imread(maskpath).astype(np.float32)[:, :, ::-1]
            H, W, C = mask.shape
            image, _ = self.transform(image, mask)
            mask = torch.from_numpy(mask.copy()).permute(2, 0, 1)
            mask = mask.mean(dim=0, keepdim=True)
            mask /= 255
            return image, mask, (H, W), maskpath
        else:
            imagepath, maskpath = self.samples[idx]
            image = cv2.imread(imagepath).astype(np.float32)[:, :, ::-1]
            mask = cv2.imread(maskpath).astype(np.float32)[:, :, ::-1]
            H, W, C = mask.shape
            image, _ = self.transform(image, mask)
            mask = torch.from_numpy(mask.copy()).permute(2,0,1)
            mask = mask.mean(dim=0, keepdim=True)
            mask /= 255
        # print(image.max(), image.min())
        return image, mask, (H, W), maskpath.split('/')[-1]

    def __len__(self):
        return len(self.samples)


if __name__=='__main__':
    import matplotlib.pyplot as plt
    plt.ion()

    cfg  = Config(mode='test', datapath='/dataC/qhd/cod/CodDataset')
    data = Data(cfg)
    loader = DataLoader(data, batch_size=1, shuffle=True, num_workers=8)
    prefetcher = DataPrefetcher(loader)
    batch_idx = -1
    image, mask = prefetcher.next()
    image = image[0].permute(1, 2, 0).cpu().numpy()*cfg.std + cfg.mean
    mask = mask[0].cpu().numpy()
    plt.subplot(121)
    plt.imshow(np.uint8(image))
    plt.subplot(122)
    plt.imshow(mask)
    input()

