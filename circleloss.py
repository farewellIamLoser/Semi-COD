from typing import Tuple

import torch
from torch import nn, Tensor
from einops import rearrange


def convert_label_to_similarity(normed_feature: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
    similarity_matrix = normed_feature.transpose(1, 0) @ normed_feature

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

class DifferenceAwareOps(nn.Module):
    def __init__(self, num_frames):
        super().__init__()
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

    def forward(self, x):
        if self.num_frames == 1:
            return x

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

class DifferenceFuseAwareOps(nn.Module):
    def __init__(self, num_frames):
        super().__init__()
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

    def forward(self, x, y):
        if self.num_frames == 1:
            return x

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


class CateBlock(nn.Module):
    def __init__(self, in_chs):
        super(CateBlock, self).__init__()
        self.Diff1 = DifferenceFuseAwareOps(2)
        self.Diff2 = DifferenceFuseAwareOps(2)
        self.Diff3 = DifferenceFuseAwareOps(2)
        self.Diff4 = DifferenceFuseAwareOps(2)
        self.Diff = DifferenceAwareOps(2)
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=8)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=1)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(4*in_chs, in_chs, 1),
            nn.ReLU(True),
            nn.Conv2d(in_chs, 4*in_chs, 1),
            nn.Softmax(dim=1),
        )
        self.softmax = nn.Softmax(dim=1)
        self.class_cate = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(4*in_chs, in_chs, 1),
            nn.ReLU(True),
            nn.Conv2d(in_chs, 69, 1),
            nn.Flatten(),
            nn.Softmax(dim=1),
        )

    def forward(self, feat1, feat2):
        fuse_feat1 = self.up1(self.Diff1(feat1[1], feat2[1]))
        fuse_feat2 = self.up2(self.Diff2(feat1[2], feat2[2]))
        fuse_feat3 = self.up3(self.Diff3(feat1[3], feat2[3]))
        fuse_feat4 = self.up4(self.Diff4(feat1[4], feat2[4]))
        fuse_feat = torch.cat([fuse_feat1, fuse_feat2, fuse_feat3, fuse_feat4], dim=1)
        gate = self.gate(fuse_feat)
        out = self.Diff(fuse_feat * gate)

        cate_pred = self.class_cate(fuse_feat)
        return out, cate_pred


if __name__ == "__main__":
    afeat1 = nn.functional.normalize(torch.rand(4, 64, 10, 10, requires_grad=True))
    afeat2 = nn.functional.normalize(torch.rand(4, 64, 20, 20, requires_grad=True))
    afeat3 = nn.functional.normalize(torch.rand(4, 64, 40, 40, requires_grad=True))
    afeat4 = nn.functional.normalize(torch.rand(4, 64, 80, 80, requires_grad=True))
    bfeat1 = nn.functional.normalize(torch.rand(4, 64, 10, 10, requires_grad=True))
    bfeat2 = nn.functional.normalize(torch.rand(4, 64, 20, 20, requires_grad=True))
    bfeat3 = nn.functional.normalize(torch.rand(4, 64, 40, 40, requires_grad=True))
    bfeat4 = nn.functional.normalize(torch.rand(4, 64, 80, 80, requires_grad=True))

    criterion = CateBlock(64)
    a, b = criterion([afeat1, afeat1, afeat2, afeat3, afeat4], [bfeat1, bfeat1, bfeat2, bfeat3, bfeat4])
    print('finish')