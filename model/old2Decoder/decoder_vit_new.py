import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from einops import rearrange
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from timm.models.layers import DropPath, to_2tuple
def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        # elif isinstance(m, nn.LayerNorm):
        #     nn.init.constant_(m.bias, 0)
        #     nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Sequential, nn.ModuleList)):
            weight_init(m)
        elif isinstance(m, (
        nn.ReLU, nn.GELU, nn.ReLU6, nn.InstanceNorm2d, nn.Sigmoid, nn.Softmax, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool1d, nn.UpsamplingBilinear2d,
        nn.Sigmoid, nn.Identity, nn.Flatten, nn.LayerNorm)):
            pass
        else:
            m.initialize()


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class PyramidPooling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PyramidPooling, self).__init__()
        hidden_channel = int(in_channel / 4)
        self.conv1 = basicConv(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv2 = basicConv(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv3 = basicConv(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv4 = basicConv(in_channel, hidden_channel, k=1, s=1, p=0)
        self.out = basicConv(in_channel * 2, out_channel, k=1, s=1, p=0)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = F.interpolate(self.conv1(F.adaptive_avg_pool2d(x, 1)), size)
        feat2 = F.interpolate(self.conv2(F.adaptive_avg_pool2d(x, 2)), size)
        feat3 = F.interpolate(self.conv3(F.adaptive_avg_pool2d(x, 3)), size)
        feat4 = F.interpolate(self.conv4(F.adaptive_avg_pool2d(x, 4)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)

        return x

    def initialize(self):
        weight_init(self)

class Deep_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Deep_Block, self).__init__()
        hidden_channel = int(in_channel / 4)
        self.conv1 = basicConv(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv2 = basicConv(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv3 = basicConv(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv4 = basicConv(in_channel, hidden_channel, k=1, s=1, p=0)
        self.out = basicConv(in_channel * 2, out_channel, k=1, s=1, p=0)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = F.interpolate(self.conv1(F.adaptive_avg_pool2d(x, 1)), size)
        feat2 = F.interpolate(self.conv2(F.adaptive_avg_pool2d(x, 2)), size)
        feat3 = F.interpolate(self.conv3(F.adaptive_avg_pool2d(x, 3)), size)
        feat4 = F.interpolate(self.conv4(F.adaptive_avg_pool2d(x, 4)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)

        return x

    def initialize(self):
        weight_init(self)

class basicConv(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(basicConv, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
            # conv.append(nn.LayerNorm(out_channel, eps=1e-6))
        if relu:
            conv.append(nn.GELU())
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)

    def initialize(self):
        weight_init(self)

########################################### CoordAttention #########################################
# Revised from: Coordinate Attention for Efficient Mobile Network Design, CVPR21
# https://github.com/houqb/CoordAttention
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

    def initialize(self):
        weight_init(self)

class SFA(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SFA, self).__init__()
        self.branch0 = nn.Sequential(
            basicConv(in_channel, out_channel, 1, relu=False),
        )
        self.branch1 = nn.Sequential(
            basicConv(in_channel, out_channel, 1),
            basicConv(out_channel, out_channel, k=7, p=3),
            basicConv(out_channel, out_channel, 3, p=7, d=7, relu=False)
        )
        self.branch2 = nn.Sequential(
            basicConv(in_channel, out_channel, 1),
            basicConv(out_channel, out_channel, k=7, p=3),
            basicConv(out_channel, out_channel, k=7, p=3),
            basicConv(out_channel, out_channel, 3, p=7, d=7, relu=False)
        )
        self.css = CrossStrengthen(out_channel)

    def forward(self, x, y):

        N, C, H, W = x.size()
        x0 = F.interpolate(self.branch0(x), H)
        x1 = F.interpolate(self.branch1(x), H)
        x2 = F.interpolate(self.branch2(x), H)
        if x.size() != y.size():
            y = F.interpolate(y, H)
        out = self.css(x0, y)
        out = self.css(out, x1)
        out = self.css(out, x2)
        return out

    def initialize(self):
        weight_init(self)

class MaskAttention(nn.Module):
    def __init__(self, channel):
        super(MaskAttention, self).__init__()
        LayerNorm_type = 'WithBias'
        bias = False
        ffn_expansion_factor = 4
        num_heads = 8
        mode = 'dilation'
        self.conv_1 = conv3x3(channel, channel)
        self.bn_1 = nn.BatchNorm2d(channel)
        self.conv_2 = conv3x3(channel, channel)
        self.bn_2 = nn.BatchNorm2d(channel)
        self.norm1 = LayerNorm(channel, LayerNorm_type)
        self.attn = Attention(channel, num_heads, bias, mode)
        self.norm2 = LayerNorm(channel, LayerNorm_type)
        self.ffn = FeedForward(channel, ffn_expansion_factor, bias)
        self.fuse = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=1), nn.Conv2d(channel, channel, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(channel), nn.ReLU(inplace=True))

    def forward(self, x, y):
        x = x * y
        out = F.relu(self.bn_1(self.conv_1(x)), inplace=True)
        out = F.relu(self.bn_2(self.conv_2(out)), inplace=True)
        out = out + self.attn(self.norm1(out))
        out = out + self.ffn(self.norm2(out))
        out = self.fuse(x + x * out)
        return out

    def initialize(self):
        weight_init(self)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

    def initialize(self):
        weight_init(self)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def initialize(self):
        weight_init(self)

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

    def initialize(self):
        weight_init(self)

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape


    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

    def initialize(self):
        weight_init(self)

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, stride=16, in_chans=3, embed_dim=768, norm_layer=None,
                 flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = ((img_size[0] - patch_size[0]) // stride + 1, (img_size[1] - patch_size[1]) // stride + 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class UnpatchEmbed(nn.Module):
    """Patch Embedding to 2D Image"""

    def __init__(self, img_size=224, patch_size=16, stride=16, embed_dim=768, in_chans=3):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        self.grid_size = ((img_size - patch_size) // stride + 1, (img_size - patch_size) // stride + 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.unproj = nn.ConvTranspose2d(embed_dim, in_chans, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        B, N, C = x.shape
        assert N == self.num_patches, \
            f"Input patch size ({N}) doesn't match model ({self.num_patches})."
        x = x.transpose(1, 2).reshape(B, C, self.grid_size[0], self.grid_size[1])
        x = self.unproj(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
def _get_act_fn(act_name, inplace=True):
    if act_name == "relu":
        return nn.ReLU(inplace=inplace)
    elif act_name == "leaklyrelu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
    elif act_name == "gelu":
        return nn.GELU()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        raise NotImplementedError
class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        act_name="relu",
        is_transposed=False,
    ):
        """
        Convolution-BatchNormalization-ActivationLayer

        :param in_planes:
        :param out_planes:
        :param kernel_size:
        :param stride:
        :param padding:
        :param dilation:
        :param groups:
        :param bias:
        :param act_name: None denote it doesn't use the activation layer.
        :param is_transposed: True -> nn.ConvTranspose2d, False -> nn.Conv2d
        """
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes

        if is_transposed:
            conv_module = nn.ConvTranspose2d
        else:
            conv_module = nn.Conv2d
        self.add_module(
            name="conv",
            module=conv_module(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=to_2tuple(stride),
                padding=to_2tuple(padding),
                dilation=to_2tuple(dilation),
                groups=groups,
                bias=bias,
            ),
        )
        self.add_module(name="bn", module=nn.BatchNorm2d(out_planes))
        if act_name is not None:
            self.add_module(name=act_name, module=_get_act_fn(act_name=act_name))

class SFTA(nn.Module):
    def __init__(self, in_channel, out_channel, img_size, stride, is_fisrt_cls=True):
        super(SFTA, self).__init__()
        patch_size = 4
        embed_dim = 276
        depth = 3
        num_heads = 12
        attn_drop_rate = 0
        drop_path_rate = 0.1
        mlp_ratio = 4.
        qkv_bias = True
        act_layer = nn.GELU
        drop_rate = 0.
        self.is_cls = is_fisrt_cls
        self.down_channel = nn.Sequential(
            conv3x3(in_channel, out_channel, stride=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(out_channel, out_channel, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.fuse = nn.Sequential(
            conv3x3(out_channel * 2, out_channel, stride=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        if self.is_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        channels = out_channel
        norm_layer = nn.LayerNorm
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, stride=stride, in_chans=channels, embed_dim=embed_dim)
        self.unpatch_embed = UnpatchEmbed(
            img_size=img_size, patch_size=patch_size, stride=stride, embed_dim=embed_dim, in_chans=channels)
        self.final_relu = nn.ReLU(True)
        self.temp_strengthen = nn.Sequential(
            ConvBNReLU(channels, channels, 3, 1, 1, act_name=None)
        )
    def forward(self, x, y, cls_token=None):
        _, _, H1, _ = x.size()
        _, _, H2, _ = y.size()
        if H1 != H2:
            y = self.up(y)
        x = self.down_channel(x)
        first_out = self.fuse(torch.cat((x, y), dim=1))
        hidden_states = self.patch_embed(first_out)
        if self.is_cls:
            cls_token = self.cls_token.expand(hidden_states.shape[0], -1,
                                              -1)  # stole cls_tokens impl from Phil Wang, thanks
        hidden_states = torch.cat((cls_token, hidden_states), dim=1)
        for blk in self.blocks:
            x = blk(hidden_states)
        x = self.norm(x)
        cls_token = x[:, 0, :].unsqueeze(1)
        out = x[:, 1:, :]
        out = self.unpatch_embed(out)
        out = self.temp_strengthen(out + first_out)
        out = self.final_relu(out)
        return out, cls_token

    def initialize(self):
        weight_init(self)
class MSA_head(nn.Module):
    def __init__(self, mode='dilation',dim=128, num_heads=8, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias'):
        super(MSA_head, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias, mode)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

    def initialize(self):
        weight_init(self)
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

    def initialize(self):
        weight_init(self)


class CrossStrengthen(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False, LayerNorm_type='WithBias'):
        super(CrossStrengthen, self).__init__()
        self.num_heads = num_heads
        self.temperature1 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature3 = nn.Parameter(torch.ones(num_heads, 1, 1))
        ffn_expansion_factor = 4
        # x
        self.qkv_x_0 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_x_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_x_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.qkvx1conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkvx2conv = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1, groups=dim, bias=bias)
        self.qkvx3conv = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1, groups=dim, bias=bias)

        # y
        self.qkv_y_0 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_y_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_y_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.qkvy1conv = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1, groups=dim, bias=bias)
        self.qkvy2conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkvy3conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.norm = LayerNorm(dim, LayerNorm_type)
        self.norm_x = LayerNorm(dim, LayerNorm_type)
        self.norm_y = LayerNorm(dim, LayerNorm_type)
        self.fuse = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1), nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(dim), nn.ReLU(inplace=True))

    def forward(self, x, y):
        input = x
        # Attention part
        B, C, H, W = x.shape
        x = self.norm_x(x)
        y = self.norm_y(y)
        qx = self.qkvx1conv(self.qkv_x_0(x))
        kx = self.qkvx2conv(self.qkv_x_1(x))
        vx = self.qkvx3conv(self.qkv_x_2(x))
        qx = rearrange(qx, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        kx = rearrange(kx, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        vx = rearrange(vx, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        qx = torch.nn.functional.normalize(qx, dim=-1)
        kx = torch.nn.functional.normalize(kx, dim=-1)

        qy = self.qkvy1conv(self.qkv_y_0(y))
        ky = self.qkvy2conv(self.qkv_y_1(y))
        vy = self.qkvy3conv(self.qkv_y_2(y))
        qy = rearrange(qy, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        ky = rearrange(ky, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        vy = rearrange(vy, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        qy = torch.nn.functional.normalize(qy, dim=-1)
        ky = torch.nn.functional.normalize(ky, dim=-1)

        attnx = ((qx @ ky.transpose(-2, -1)) * self.temperature1).softmax(dim=-1)
        attny = ((qy @ kx.transpose(-2, -1)) * self.temperature2).softmax(dim=-1)

        out = (attnx @ attny @ vx) @ (vx.transpose(-2, -1) @ vy) * self.temperature3
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)
        out = self.project_out(out)

        out = input + out
        out = out + self.ffn(self.norm(out))
        out = self.fuse(input + input * out)
        return out
    def initialize(self):
        weight_init(self)


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1, bias=False):
    "3x3 convolution with padding"

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=bias)

class OutPut(nn.Module):
    def __init__(self, in_chs, scale=1):
        super(OutPut, self).__init__()
        self.out = nn.Sequential(nn.Conv2d(in_chs, in_chs, 1, bias=False),
                                 nn.BatchNorm2d(in_chs),
                                 nn.ReLU(inplace=True),
                                 nn.UpsamplingBilinear2d(scale_factor=scale),
                                 nn.Conv2d(in_chs, 1, 1),
                                 nn.Sigmoid())

    def forward(self, feat):
        return self.out(feat)

    def initialize(self):
        weight_init(self)

class SemiOutPut(nn.Module):
    def __init__(self, in_chs):
        super(SemiOutPut, self).__init__()
        self.out = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                 nn.BatchNorm2d(in_chs),
                                 nn.ReLU(inplace=True),
                                 nn.Flatten(),
                                 nn.Sigmoid())

    def forward(self, feat):
        return self.out(feat)

    def initialize(self):
        weight_init(self)

class Decoder(nn.Module):
    def __init__(self, channels):
        super(Decoder, self).__init__()
        img_sizes = [10, 20, 40, 80, 320]
        self.pyramid_pooling = PyramidPooling(512, channels)
        self.sfa1 = SFTA(512, channels, img_size=img_sizes[0], stride=2, is_fisrt_cls=True)
        self.sfa2 = SFTA(320, channels, img_size=img_sizes[1], stride=4, is_fisrt_cls=False)
        self.sft1 = SFTA(128, channels, img_size=img_sizes[2], stride=4, is_fisrt_cls=False)
        self.sft2 = SFTA(64, channels, img_size=img_sizes[3], stride=4, is_fisrt_cls=False)
        self.up = nn.UpsamplingBilinear2d(scale_factor=4)
        self.single_class_cate = nn.Sequential(
            nn.Conv2d(69, 69 * 4, 1),
            nn.ReLU(True),
            nn.Conv2d(69 * 4, 69, 1),
            nn.Flatten(),
            nn.Softmax(dim=1)
        )
        self.down = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        # self.initialize()
    def forward(self, E1, E2, E3, E4):
        # E1 512 12 E2 320 24 E3 128 48 E4 64 96 96
        S5 = self.pyramid_pooling(E1)
        S4, cls = self.sfa1(E1, S5)
        S3, cls = self.sfa2(E2, S4, cls)
        S2, cls = self.sft1(E3, S3, cls)
        S1, cate_pred = self.sft2(E4, S2, cls)
        out = self.down(self.up(S1))
        B, H, D = cate_pred.size()
        cate_preds = cate_pred.view(B, D, 1, 1).chunk(4, dim=1)
        cate_pred1 = self.single_class_cate(cate_preds[0])
        cate_pred2 = self.single_class_cate(cate_preds[1])
        cate_pred3 = self.single_class_cate(cate_preds[2])
        cate_pred4 = self.single_class_cate(cate_preds[3])
        cate_pred = [cate_pred1, cate_pred2, cate_pred3, cate_pred4]
        return out

    def initialize(self):
        weight_init(self)
class DifferenceAwareOps(nn.Module):
    def __init__(self, num_frames):
        super(DifferenceAwareOps, self).__init__()
        self.num_frames = num_frames

        self.temperal_proj_norm = nn.LayerNorm(num_frames, elementwise_affine=False)
        self.temperal_proj_kv = nn.Linear(num_frames, 2 * num_frames, bias=False)
        self.temperal_proj = nn.Sequential(
            nn.Conv2d(num_frames, num_frames, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(num_frames, num_frames, 3, 1, 1, bias=False),
        )
        for t in self.parameters():
            nn.init.zeros_(t)

        self.initialize()
    def forward(self, x):
        B, C, H, W = x.shape

        unshifted_x_tmp = rearrange(x, "(b t) c h w -> b c h w t", t=self.num_frames)
        B, C, H, W, T = unshifted_x_tmp.shape
        shifted_x_tmp = torch.roll(unshifted_x_tmp, shifts=1, dims=-1)
        diff_q = shifted_x_tmp - unshifted_x_tmp  # B,C,H,W,T
        diff_q = self.temperal_proj_norm(diff_q)  # normalization along the time

        # merge all channels
        diff_k, diff_v = self.temperal_proj_kv(diff_q).chunk(2, dim=-1)
        diff_qk = torch.einsum("bxhwt, byhwt -> bxyt", diff_q, diff_k) * (H * W) ** -0.5
        temperal_diff = torch.einsum("bxyt, byhwt -> bxhwt", diff_qk.softmax(dim=2), diff_v)

        temperal_diff = rearrange(temperal_diff, "b c h w t -> (b c) t h w")
        shifted_x_tmp = self.temperal_proj(temperal_diff)  # combine different time step
        shifted_x_tmp = rearrange(shifted_x_tmp, "(b c) t h w -> (b t) c h w", c=x.shape[1])
        return x + shifted_x_tmp

    def initialize(self):
        weight_init(self)

class DifferenceFuseAwareOps(nn.Module):
    def __init__(self, num_frames):
        super(DifferenceFuseAwareOps, self).__init__()
        self.num_frames = num_frames

        self.temperal_proj_norm = nn.LayerNorm(num_frames, elementwise_affine=False)
        self.temperal_proj_kv = nn.Linear(num_frames, 2 * num_frames, bias=False)
        self.temperal_proj = nn.Sequential(
            nn.Conv2d(num_frames, num_frames, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(num_frames, num_frames, 3, 1, 1, bias=False),
        )
        for t in self.parameters():
            nn.init.zeros_(t)

        self.initialize()
    def forward(self, x, y):
        B, C, H, W = x.shape
        # if B == 1:
        #     return x
        y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=False)
        # x diff_qk generation
        unshifted_x_tmp = rearrange(x, "(b t) c h w -> b c h w t", t=self.num_frames)
        B, C, H, W, T = unshifted_x_tmp.shape
        shifted_x_tmp = torch.roll(unshifted_x_tmp, shifts=1, dims=-1)
        diff_xq = shifted_x_tmp - unshifted_x_tmp  # B,C,H,W,T
        diff_xq = self.temperal_proj_norm(diff_xq)  # normalization along the time

        # merge all channels
        diff_xk, diff_xv = self.temperal_proj_kv(diff_xq).chunk(2, dim=-1)
        diff_xqk = torch.einsum("bxhwt, byhwt -> bxyt", diff_xq, diff_xk) * (H * W) ** -0.5

        # y diff_qk generation
        unshifted_y_tmp = rearrange(y, "(b t) c h w -> b c h w t", t=self.num_frames)
        B, C, H, W, T = unshifted_y_tmp.shape
        shifted_y_tmp = torch.roll(unshifted_y_tmp, shifts=1, dims=-1)
        diff_yq = shifted_y_tmp - unshifted_y_tmp  # B,C,H,W,T
        diff_yq = self.temperal_proj_norm(diff_yq)  # normalization along the time

        # merge all channels
        diff_yk, diff_yv = self.temperal_proj_kv(diff_yq).chunk(2, dim=-1)
        diff_yqk = torch.einsum("bxhwt, byhwt -> bxyt", diff_yq, diff_yk) * (H * W) ** -0.5

        diff_qk = torch.einsum("bxyt, bxyt -> bxyt", diff_xqk, diff_yqk)
        temperal_diff = torch.einsum("bxyt, byhwt -> bxhwt", diff_qk.softmax(dim=2), diff_xv)

        temperal_diff = rearrange(temperal_diff, "b c h w t -> (b c) t h w")
        shifted_x_tmp = self.temperal_proj(temperal_diff)  # combine different time step
        shifted_x_tmp = rearrange(shifted_x_tmp, "(b c) t h w -> (b t) c h w", c=x.shape[1])
        return x + shifted_x_tmp

    def initialize(self):
        weight_init(self)

class FeatureDecoder(nn.Module):
    def __init__(self, in_chs):
        super(FeatureDecoder, self).__init__()
        self.Diff1 = DifferenceFuseAwareOps(1)
        self.Diff2 = DifferenceFuseAwareOps(1)
        self.Diff3 = DifferenceFuseAwareOps(1)
        self.Diff4 = DifferenceFuseAwareOps(1)
        self.Diff = DifferenceAwareOps(1)
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=8)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=1)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_chs, in_chs, 1),
            nn.ReLU(True),
            nn.Conv2d(in_chs, in_chs, 1),
            nn.Softmax(dim=1),
        )
        self.CBR = nn.Sequential(
            nn.Conv2d(in_chs * 4, in_chs, 1),
            nn.BatchNorm2d(in_chs),
            nn.ReLU(True)
        )
        self.softmax = nn.Softmax(dim=1)
        self.class_cate = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_chs, in_chs*64, 1),
            nn.ReLU(True),
            nn.Conv2d(in_chs*64, 69 * 4, 1)
        )
        self.single_class_cate = nn.Sequential(
            nn.Conv2d(69, 69 * 4, 1),
            nn.ReLU(True),
            nn.Conv2d(69 * 4, 69, 1),
            nn.Flatten(),
            nn.Softmax(dim=1)
        )
        self.final_out = nn.Sequential(nn.Conv2d(in_chs, in_chs, 1, bias=False),
                                 nn.BatchNorm2d(in_chs),
                                 nn.ReLU(inplace=True),
                                 nn.UpsamplingBilinear2d(scale_factor=4),
                                 nn.Conv2d(in_chs, 1, 1),
                                 nn.Sigmoid())

        self.initialize()
    def forward(self, feat1, feat2):
        fuse_feat1 = self.up1(self.Diff1(feat1[1], feat2[1]))
        fuse_feat2 = self.up2(self.Diff2(feat1[2], feat2[2]))
        fuse_feat3 = self.up3(self.Diff3(feat1[3], feat2[3]))
        fuse_feat4 = self.up4(self.Diff4(feat1[4], feat2[4]))
        fuse_feat = self.CBR(torch.cat([fuse_feat1, fuse_feat2, fuse_feat3, fuse_feat4], dim=1))
        gate = self.gate(fuse_feat)
        out = self.Diff(fuse_feat * gate)
        out = self.final_out(out)
        return out

    def initialize(self):
        weight_init(self)