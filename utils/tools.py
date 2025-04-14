# import os
# import random
# from tracemalloc import Snapshot
from typing import Tuple
# import cv2
# import einops
import numpy as np
# import torch
# import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import affine
# from torchvision.utils import draw_segmentation_masks
# from PIL import Image
# import tqdm
import torch
from torch import nn, Tensor

def get_featuremap(h, x):
    w = h.weight
    b = h.bias
    c = w.shape[1]
    c1 = F.conv2d(x, w.transpose(0, 1), padding=(1, 1), groups=c)
    return c1, b


def ToLabel(E):
    fgs = np.argmax(E, axis=1).astype(np.float32)
    return fgs.astype(np.uint8)

def convert_label_to_similarity(normed_feature: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]



class CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, normed_feature: Tensor, label: Tensor) -> Tensor:
        similarity_matrix = normed_feature.transpose(1, 0) @ normed_feature
        total_loss = 0
        for i in range(label.size(0)):
            temp_label = label[i,:]
            label_matrix = temp_label.unsqueeze(1) == temp_label.unsqueeze(0)

            positive_matrix = label_matrix.triu(diagonal=1)
            negative_matrix = label_matrix.logical_not().triu(diagonal=1)

            similarity_matrix = similarity_matrix.view(-1)
            positive_matrix = positive_matrix.view(-1)
            negative_matrix = negative_matrix.view(-1)

            ap = torch.clamp_min(- similarity_matrix[positive_matrix].detach() + 1 + self.m, min=0.)
            an = torch.clamp_min(similarity_matrix[negative_matrix].detach() + self.m, min=0.)

            delta_p = 1 - self.m
            delta_n = self.m

            logit_p = - ap * (similarity_matrix[positive_matrix] - delta_p) * self.gamma
            logit_n = an * (similarity_matrix[negative_matrix] - delta_n) * self.gamma

            loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))
            total_loss += loss
        total_loss /= label.size(0)
        return total_loss


def SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.AvgPool2d(3, 1, 1)(x)
    mu_y = nn.AvgPool2d(3, 1, 1)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(3, 1, 1)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(3, 1, 1)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(3, 1, 1)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d

    return torch.clamp((1 - SSIM) / 2, 0, 1)

def shift_matrix(matrix, shift_direction, stride=1):
    n, m = matrix.shape
    shifted_matrix = torch.zeros_like(matrix)

    # 计算平移的行和列偏移量
    if shift_direction == 0:  # 向上平移
        row_offset = -stride
        col_offset = 0
    elif shift_direction == 1:  # 向左平移
        row_offset = 0
        col_offset = -stride
    elif shift_direction == 2:  # 向下平移
        row_offset = stride
        col_offset = 0
    elif shift_direction == 3:  # 向右平移
        row_offset = 0
        col_offset = stride
    else:
        row_offset = 0
        col_offset = 0

    # 切片操作进行平移
    shifted_matrix[max(0, row_offset):min(n, n + row_offset), max(0, col_offset):min(m, m + col_offset)] = \
        matrix[max(0, -row_offset):min(n, n - row_offset), max(0, -col_offset):min(m, m - col_offset)]

    return shifted_matrix

def MultiDirGrad(img1, img2):
    B, C, H, W = img1.shape
    stride = 1
    Grad_img = torch.zeros((H, W)).to('cuda')
    loss = 0
    for i in range(B):
        for j in range(4):
            shift_img1 = shift_matrix(img1[i, 0], shift_direction=j, stride=stride)
            shift_img1 = (img1[i, 0] - shift_img1) / stride
            shift_img2 = shift_matrix(img2[i, 0], shift_direction=j, stride=stride)
            shift_img2 = (img2[i, 0] - shift_img2) / stride
            Grad_img += torch.abs(shift_img1 - shift_img2)
        loss += torch.mean(Grad_img)

    return loss / B

def SaliencyStructureConsistency(x, y, alpha):
    ssim = torch.mean(SSIM(x, y))
    l1_loss = torch.mean(torch.abs(x - y))
    loss_ssc = alpha * ssim + (1 - alpha) * l1_loss
    return loss_ssc

def GradStructureConsistency(x, y):
    loss_gsc = MultiDirGrad(x, y)
    return loss_gsc

def CateConsistency(x, y):
    feat = nn.functional.normalize(x)
    lbl = y

    criterion = CircleLoss(m=0.25, gamma=256)
    loss_CC = 0.01 * criterion(feat, lbl)

    return loss_CC

def CategoricalConsistency(x,y):
    total_loss = 0
    criterion = nn.BCELoss()
    y = torch.chunk(y, chunks=4, dim=1)
    for cate, label in zip(x,y):
        label = label.squeeze()
        loss = criterion(cate, label)
        total_loss += loss
    return loss

def FeatureConsistency(x, y):
    total_loss = 0
    for o_f, s_f in zip(x,y):
        similarity_matrix = F.cosine_similarity(o_f,s_f)
        loss = torch.mean(torch.clamp(1 - similarity_matrix, min=0))
        total_loss += loss
    return loss

def SaliencyStructureConsistencynossim(x, y):
    l1_loss = torch.mean(torch.abs(x - y))
    return l1_loss


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_coef(iter_percentage=1, method="cos", milestones=(0, 1)):
    min_point, max_point = min(milestones), max(milestones)
    min_coef, max_coef = 0, 1

    ual_coef = 1.0
    if iter_percentage < min_point:
        ual_coef = min_coef
    elif iter_percentage > max_point:
        ual_coef = max_coef
    else:
        if method == "linear":
            ratio = (max_coef - min_coef) / (max_point - min_point)
            ual_coef = ratio * (iter_percentage - min_point)
        elif method == "cos":
            perc = (iter_percentage - min_point) / (max_point - min_point)
            normalized_coef = (1 - np.cos(perc * np.pi)) / 2
            ual_coef = normalized_coef * (max_coef - min_coef) + min_coef
    return ual_coef

def intro_loss(x):
    ual_coef = get_coef(iter_percentage=1, method="cos", milestones=(0, 1))
    ual_loss = ual_coef * (1 - (2 * x - 1).abs().pow(2)).mean()
    return ual_loss

class Flip:
    def __init__(self, flip):
        self.flip = flip

    def __call__(self, img):
        if self.flip == 0:
            return img.flip(-1)
        else:
            return img.flip(-2)


class Translate:
    def __init__(self, fct):
        '''Translate offset factor'''
        drct = np.random.randint(0, 4)
        self.signed_x = drct >= 2 or -1
        self.signed_y = drct % 2 or -1
        self.fct = fct

    def __call__(self, img):
        angle = 0
        scale = 1
        h, w = img.shape[-2:]
        h, w = int(h * self.fct), int(w * self.fct)
        return affine(img, angle, (h * self.signed_y, w * self.signed_x), scale, shear=0,
                      interpolation=InterpolationMode.BILINEAR)


class Crop:
    def __init__(self, H, W):
        '''keep the relative ratio for offset'''
        self.h = H
        self.w = W
        self.xm = np.random.uniform()
        self.ym = np.random.uniform()
        # print(self.xm, self.ym)

    def __call__(self, img):
        H, W = img.shape[-2:]
        sh = int(self.h * H)
        sw = int(self.w * W)
        ymin = int((H - sh + 1) * self.ym)
        xmin = int((W - sw + 1) * self.xm)
        img = img[..., ymin:ymin + sh, xmin:xmin + sw]
        img = F.interpolate(img, size=(H, W), mode='bilinear', align_corners=False)
        return img

if __name__ == "__main__":
    feat = nn.functional.normalize(torch.rand(256, 64, 64, requires_grad=True))
    lbl = torch.randint(high=2, size=(256, 64, 64))

    loss = intro_loss(feat)

    print(loss)