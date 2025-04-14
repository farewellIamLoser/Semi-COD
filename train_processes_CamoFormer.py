import torch.nn.functional as F
import torch
import loss
from feature_loss import *
from utils import ramps
from utils.tools import *
import testImage as ti
criterion = torch.nn.CrossEntropyLoss(weight=None, ignore_index=255, reduction='mean').cuda()
loss_lsc = FeatureLoss().cuda()
loss_lsc_kernels_desc_defaults = [{"weight": 1, "xy": 6, "rgb": 0.1}]
loss_lsc_radius = 5
l = 0.3

def get_current_consistency_weight(epoch, consistency=0.1, consistency_rampup=150):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * ramps.sigmoid_rampup(epoch, consistency_rampup)
    
def get_transform(ops=[0,1,2]):
    '''One of flip, translate, crop'''
    op = np.random.choice(ops)
    if op==0:
        flip = np.random.randint(0, 2)
        pp = Flip(flip)
    elif op==1:
        # pp = Translate(0.3)
        pp = Translate(0.15)
    elif op==2:
        pp = Crop(0.7, 0.7)
    return pp

def get_featuremap(h, x):
    w = h.weight
    b = h.bias
    c = w.shape[1]
    c1 = F.conv2d(x, w.transpose(0,1), padding=(1,1), groups=c)
    return c1, b

def unsymmetric_grad(x, y, calc, w1, w2):
    '''
    x: strong feature
    y: weak feature'''
    return calc(x, y.detach())*w1 + calc(x.detach(), y)*w2

# targeted at boundary: only p/n coexsits.
# learn features that focus on boundary prediction
# use feature vectors to guide pixel prediction 
# covariance to encourage the feature difference in the most decisive ones
def feature_loss(feature_map, pred, kr=4, norm=False, crtl_loss=True, w_ftp=0, topk=16, step_ratio=2):
    '''
    pred: n, 1, h, w'''
    # normalize feature map (but how?)
    if norm:
        fmap = feature_map / feature_map.std(dim=(-1,-2), keepdim=True).mean(dim=1, keepdim=True)
    else: fmap=feature_map
    # print(fmap.max(), fmap.min(), fmap.std(dim=(-1,-2)).max())

    n, c, h, w =fmap.shape
    # get local feature map
    ks = 2*kr
    assert h%ks==0 and w%ks==0
    # print('ks', ks)
    uf = lambda x: F.unfold(x, ks, padding = 0, stride=ks//step_ratio).permute(0,2,1).reshape(-1, x.shape[1], ks*ks) # N * no.blk, 64, 8*8
    fcmap = uf(fmap) 
    fcpred = uf(pred) # N', 1, 10*10
    # # get fg/bg confident coexisting block
    cfd_thres = .8
    exst = lambda x: (x>cfd_thres).sum(2, keepdim=True) > 0.3*ks*ks
    coexists = (exst(fcpred) & exst(1-fcpred))
    coexists = coexists[:, 0, 0] # N', 1, 1
    fcmap = fcmap[coexists]
    fcpred = fcpred[coexists]
    # print(fcmap.shape, fcpred.shape)
    if not len(fcmap):
        return 0, 0
    # minus mean
    mfcmap = fcmap - fcmap.mean(2, keepdim=True)
    mfcpred = fcpred - fcpred.mean(2, keepdim=True)
    # get most relevance in confident area bout saliency
    cov = mfcmap.matmul(mfcpred.permute(0, 2, 1)) # N', 64, 1
    sgnf_id = cov.abs().topk(topk, dim=1)[1].expand(-1,-1,ks*ks) # n', topk, 10*10
    sg_fcmap = fcmap.gather(dim=1, index=sgnf_id) # n', topk, 10*10
    # different potential calculation
    crf_k = lambda x: (-(x[:, :, None]-x[:, :, :, None])**2 * 0.5).sum(1, keepdim=True).exp() # n', 1, 100, 100
    pred_grvt = lambda x,y: (1-x)*y + x*(1-y) # (x-y).abs() # x*y + (1-x)*(1-y) - x*(1-y) - (1-x)*y
    ft_grvt = lambda x: 1-crf_k(x)
    # position
    xy = torch.stack(torch.meshgrid(torch.arange(ks, device=pred.device), torch.arange(ks, device=pred.device))) / 6
    xy = (xy).reshape(1,2, ks*ks).expand(len(sg_fcmap),-1,-1) # 1, 1, 100
    ffxy = crf_k(xy)
    if crtl_loss:
        # train the feature map without pred grad
        # L2 norm loss
        pmap = fcpred.detach()
        pmap = 0.5 - pred_grvt(pmap.unsqueeze(2), pmap.unsqueeze(-1)) # n', 1, 100, 100
        fpmap = ft_grvt(sg_fcmap) * ffxy
        ice = (pmap*fpmap).mean()
        # reversely, train the pred map
        # calculate CRF with confident point
        fffm = crf_k(sg_fcmap.detach())
        kernel = fffm*ffxy # n', 1, 10*10, 10*10
    else:
        ice = 0
        fffm = crf_k(sg_fcmap)
        kernel = fffm*ffxy # n', 1, 10*10, 10*10
        kernel[torch.eye(ks*ks, device=pred.device, dtype=bool).expand_as(kernel)] = 0

    pp = pred_grvt(fcpred[:,:,None], fcpred.unsqueeze(-1)) # n', 1, 100, 100
    if w_ftp==0:
        crf = (kernel * pp).mean()
    elif w_ftp==1:
        crf = (kernel.detach() * pp).mean() * (1+w_ftp)
    else:
        crf = unsymmetric_grad(kernel, pp, lambda x,y:(x*y).mean(), 1-w_ftp, 1+w_ftp)
    return crf, ice

def train_loss(data, mode, net, ctx, ft_dct, w_ft=.1, ft_st = 60, ft_fct=.5, ft_head=True, mtrsf_prob=1, ops=[0,1,2], w_l2g=0, l_me=0.1, me_st=50, me_all=False, multi_sc=0, l=0.3, sl=1):
    if ctx:
        epoch = ctx['epoch']
        global_step = ctx['global_step']
        sw = ctx['sw']
        t_epo = ctx['t_epo']

    if mode == 'stage1':
        image, mask, label = data
        pre_transform = get_transform([0, 1, 2])
        tr_image = pre_transform(image)
        tr_label = pre_transform(mask)
        image_scale = F.interpolate(tr_image, scale_factor=1, mode='bilinear', align_corners=True)
        tr_label = F.interpolate(tr_label, scale_factor=1, mode='bilinear', align_corners=True)
        out1 = net(image)
        out2 = net(image_scale)

        # loss computation
        out_s = pre_transform(out1)
        out_s = F.interpolate(out_s, scale_factor=1, mode='bilinear', align_corners=True)
        out_s_s = out2

        loss_intro1 = intro_loss(out1)
        loss_intro2 = intro_loss(out2)
        # loss_cate1 = CategoricalConsistency(cate1, label)
        # loss_cate2 = CategoricalConsistency(cate2, label)

        # loss_cc = FeatureConsistency(cate1, cate2)
        loss_ssc = SaliencyStructureConsistency(out_s_s, out_s.detach(), 0.85)

        all_loss1 = loss.single_bce(out1, mask)
        all_loss2 = loss.single_bce(out2, tr_label)
        # all_loss = all_loss2 + all_loss1 + loss_ssc + loss_cate1 + loss_cate2 + loss_intro1 + loss_intro2 + loss_cc
        all_loss = all_loss2 + all_loss1 + loss_ssc + loss_intro1 + loss_intro2
        # all_loss = all_loss2 + all_loss1

    elif mode == 'stage2':
        image, mask, Siamge, Smask, label = data
        out1, out2, O_feature, S_feature = net(image, Siamge, mode)

        # loss computation
        loss_intro1 = intro_loss(out1)
        loss_intro2 = intro_loss(out2)
        loss_cc = FeatureConsistency(S_feature, O_feature)
        loss_cate1 = CategoricalConsistency(O_feature, label)
        loss_cate2 = CategoricalConsistency(S_feature, label)
        all_loss1 = loss.single_bce(out1, mask)
        all_loss2 = loss.single_bce(out2, Smask)
        all_loss = all_loss1 + all_loss2 + loss_cc + loss_cate1 + loss_cate2 + loss_intro1 + loss_intro2
    elif mode == 'stage3':
        image, mask = data
        pre_transform = get_transform([0, 1, 2])
        tr_image = pre_transform(image)
        tr_label = pre_transform(mask)
        image_scale = F.interpolate(tr_image, scale_factor=1, mode='bilinear', align_corners=True)
        tr_label = F.interpolate(tr_label, scale_factor=1, mode='bilinear', align_corners=True)
        # out1, out2, cate1, cate2 = net(image, image_scale, mode)
        out1 = net(image)
        out2 = net(image_scale)
        # loss computation
        out_s = pre_transform(out1)
        out_s = F.interpolate(out_s, scale_factor=1, mode='bilinear', align_corners=True)
        out_s_s = out2

        # loss_intro1 = intro_loss(out1)
        # loss_intro2 = intro_loss(out2)
        # loss_cc = FeatureConsistency(cate1, cate2)
        loss_ssc = SaliencyStructureConsistency(out_s_s, out_s.detach(), 0.85)

        all_loss1 = loss.single_bce(out1, mask)
        all_loss2 = loss.single_bce(out2, tr_label)
        all_loss = all_loss2 + all_loss1 + loss_ssc
        # all_loss = all_loss2 + all_loss1 + loss_ssc + loss_intro1 + loss_intro2 + loss_cc


    return all_loss