from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
import numbers
import einops
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple
from mamba_ssm.modules.mamba_simple import Mamba
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


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
###################################embed and unembed###########################

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

def to_2tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, x)

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

def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        drop_path=0.,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        device=None,
        dtype=None,
        if_bimamba=False,
        bimamba_type="none",
        if_devide_out=False,
        init_layer_scale=None,
):
    if if_bimamba:
        bimamba_type = "v1"
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    # import ipdb; ipdb.set_trace()
    mixer_cls = partial(Mamba, layer_idx=layer_idx, bimamba_type=bimamba_type, if_devide_out=if_devide_out,
                        init_layer_scale=init_layer_scale, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block
class Block(nn.Module):
    def __init__(
            self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)

            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

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

class SFTA(nn.Module):
    def __init__(self, in_channel, out_channel, img_size, stride, is_fisrt_cls=False):
        super(SFTA, self).__init__()
        patch_size = 4
        embed_dim = 276
        depth = 4
        drop_path_rate = 0.1
        norm_epsilon = 1e-5
        rms_norm = False
        self.is_cls = is_fisrt_cls
        self.residual_in_fp32 = False
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
        # self.ref_fuse = nn.Sequential(
        #     conv3x3(out_channel + 69 * 4, out_channel, stride=1),
        #     nn.BatchNorm2d(out_channel),
        #     nn.ReLU()
        # )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        channels = out_channel
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon
        )
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        inter_dpr = [0.0] + dpr
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=None,
                    norm_epsilon=1e-05,
                    rms_norm=True,
                    residual_in_fp32=True,
                    fused_add_norm=True,
                    layer_idx=i,
                    if_bimamba=False,
                    bimamba_type='v2',
                    drop_path=inter_dpr[i],
                    if_devide_out=True,
                    init_layer_scale=None,
                )
                for i in range(depth)
            ]
        )

        if self.is_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, stride=stride, in_chans=channels, embed_dim=embed_dim)
        self.unpatch_embed = UnpatchEmbed(
            img_size=img_size, patch_size=patch_size, stride=stride, embed_dim=embed_dim, in_chans=channels)
        num_patches = self.patch_embed.num_patches
        self.pos_linear = nn.Linear(embed_dim * 3, embed_dim)
        self.differ = DifferenceAwareOps(1, channels, channels, img_size, stride)
    def forward(self, x, y, cls_token=None):
        _, _, H1, _ = x.size()
        _, _, H2, _ = y.size()
        if H1 != H2:
            y = self.up(y)
        x = self.down_channel(x)
        first_out = self.fuse(torch.cat((x, y), dim=1))
        # mamba block
        hidden_states = self.patch_embed(first_out)
        if self.is_cls:
            cls_token = self.cls_token.expand(hidden_states.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # hidden_states = torch.cat((cls_token, hidden_states), dim=1)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)
        out = layer_norm_fn(
            self.drop_path(hidden_states),
            self.norm_f.weight,
            self.norm_f.bias,
            eps=self.norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=self.residual_in_fp32,
        )
        # cls_token = out[:, 0, :].unsqueeze(1)
        out = out[:, :, :]
        out = self.unpatch_embed(out)
        out = self.differ(out, first_out)
        return out, cls_token

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
        self.sfa3 = SFTA(128, channels, img_size=img_sizes[2], stride=4, is_fisrt_cls=False)
        self.sfa4 = SFTA(64, channels, img_size=img_sizes[3], stride=4, is_fisrt_cls=False)

        self.up1 = nn.UpsamplingBilinear2d(scale_factor=32)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=16)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=8)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.class_cate = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channels * 4, channels * 64, 1),
            nn.ReLU(True),
            nn.Conv2d(channels * 64, 69 * 4, 1)
        )
        self.single_class_cate = nn.Sequential(
            nn.Conv2d(69, 69 * 4, 1),
            nn.ReLU(True),
            nn.Conv2d(69 * 4, 69, 1),
            nn.Flatten(),
            nn.Softmax(dim=1)
        )

        self.CBM = CombineBlockMamba(channels * 4, 1, img_sizes[4], stride=4)
    def forward(self, E1, E2, E3, E4):
        S5 = self.pyramid_pooling(E1)
        S4, cls = self.sfa1(E1, S5)
        S3, cls = self.sfa2(E2, S4, cls)
        S2, cls = self.sfa3(E3, S3, cls)
        S1, cate_pred = self.sfa4(E4, S2, cls)
        S4 = self.up1(S4)
        S3 = self.up2(S3)
        S2 = self.up3(S2)
        S1 = self.up4(S1)
        output = torch.cat([S4, S3, S2, S1], dim=1)
        B,H,D = cate_pred.size()
        cate_preds = cate_pred.view(B, D, 1, 1).chunk(4, dim=1)
        cate_pred1 = self.single_class_cate(cate_preds[0])
        cate_pred2 = self.single_class_cate(cate_preds[1])
        cate_pred3 = self.single_class_cate(cate_preds[2])
        cate_pred4 = self.single_class_cate(cate_preds[3])
        cate_pred = [cate_pred1, cate_pred2, cate_pred3, cate_pred4]
        output = self.CBM(output)
        return output

class CombineBlockMamba(nn.Module):
    def __init__(self, in_channel, out_channel, img_size, stride):
        super(CombineBlockMamba, self).__init__()
        patch_size = 4
        embed_dim = 224
        depth = 1
        drop_path_rate = 0.1
        norm_epsilon = 1e-5
        rms_norm = False
        self.residual_in_fp32 = False
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon
        )
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        inter_dpr = [0.0] + dpr
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=None,
                    norm_epsilon=1e-05,
                    rms_norm=True,
                    residual_in_fp32=True,
                    fused_add_norm=True,
                    layer_idx=i,
                    if_bimamba=False,
                    bimamba_type='v2',
                    drop_path=inter_dpr[i],
                    if_devide_out=True,
                    init_layer_scale=None,
                )
                for i in range(depth)
            ]
        )
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, stride=stride, in_chans=in_channel, embed_dim=embed_dim)
        self.unpatch_embed = UnpatchEmbed(
            img_size=img_size, patch_size=patch_size, stride=stride, embed_dim=embed_dim, in_chans=in_channel)
        num_patches = self.patch_embed.num_patches
        self.final_out = nn.Sequential(nn.Conv2d(in_channel, in_channel, 1, bias=False),
                                 nn.BatchNorm2d(in_channel),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(in_channel, 1, 1),
                                 nn.Sigmoid())
        self.differ = DifferenceAwareOps(1, in_channel, in_channel, img_size, stride)
    def forward(self, x):

        hidden_states = self.patch_embed(x)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)
        out = rms_norm_fn(
            self.drop_path(hidden_states),
            self.norm_f.weight,
            self.norm_f.bias,
            eps=self.norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=self.residual_in_fp32,
        )
        out = self.unpatch_embed(out)
        out = self.differ(out, x)
        out = self.final_out(out)
        return out

    def initialize(self):
        weight_init(self)
class DifferenceAwareOps(nn.Module):
    def __init__(self, num_frames, inchannel, outchannel, img_size, stride):
        super(DifferenceAwareOps, self).__init__()
        self.num_frames = num_frames
        patch_size = 4
        embed_dim = 112
        depth = 1
        drop_path_rate = 0.1
        norm_epsilon = 1e-5
        rms_norm = False
        self.residual_in_fp32 = False
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon
        )
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        inter_dpr = [0.0] + dpr
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=None,
                    norm_epsilon=1e-05,
                    rms_norm=True,
                    residual_in_fp32=True,
                    fused_add_norm=True,
                    layer_idx=i,
                    if_bimamba=False,
                    bimamba_type='v2',
                    drop_path=inter_dpr[i],
                    if_devide_out=True,
                    init_layer_scale=None,
                )
                for i in range(depth)
            ]
        )
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, stride=stride, in_chans=inchannel, embed_dim=embed_dim)
        self.unpatch_embed = UnpatchEmbed(
            img_size=img_size, patch_size=patch_size, stride=stride, embed_dim=embed_dim, in_chans=inchannel)
        num_patches = self.patch_embed.num_patches

        for t in self.parameters():
            nn.init.zeros_(t)
        self.final_relu = nn.ReLU(True)
        self.temp_strengthen = nn.Sequential(
            ConvBNReLU(inchannel, outchannel, 3, 1, 1, act_name=None)
        )
    def forward(self, x, first_x):
        B, C, H, W = x.shape

        unshifted_x_tmp = rearrange(x, "(b ) c h w -> b c h w")
        B, C, H, W = unshifted_x_tmp.shape
        shifted_x_tmp = torch.roll(unshifted_x_tmp, shifts=1, dims=-1)
        diff_q = shifted_x_tmp - unshifted_x_tmp  # B,C,H,W,T
        hidden_states = self.patch_embed(diff_q)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)
        shifted_x_tmp = rms_norm_fn(
            self.drop_path(hidden_states),
            self.norm_f.weight,
            self.norm_f.bias,
            eps=self.norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=self.residual_in_fp32,
        )
        shifted_x_tmp = self.unpatch_embed(shifted_x_tmp)
        out = self.temp_strengthen(x + shifted_x_tmp)
        out = self.final_relu(out + first_x)
        return out

    def initialize(self):
        weight_init(self)