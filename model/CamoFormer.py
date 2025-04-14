import torch
from .encoder.pvtv2_encoder import pvt_v2_b4
# from .encoder.vmanba_encoder import vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2

# from .decoder.decoder_vit import Decoder

# from .decoder.decoder_Camomamba import Decoder

# from .decoder.decoder_baseline_2 import Decoder
# from .decoder.decoder_baseline_5 import Decoder

# from model.old2Decoder.decoder_baseline import Decoder
# from .decoder.decoder_Camomamba_version2 import Decoder
# from model.old2Decoder.decoder_mamba import Decoder
# from .old2Decoder.decoder_baseline import Decoder
from model.old2Decoder.decoder_camo_mamba import Decoder
# from model.old2Decoder.decoder_vit_new import Decoder
import torch.nn as nn


def weight_init_backbone(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear): 
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.Sigmoid, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool1d, nn.Sigmoid, nn.Identity, nn.UpsamplingBilinear2d)):
            pass
        else:
            m.initialize()

def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            #nn.init.xavier_normal_(m.weight, gain=1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear): 
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            #nn.init.xavier_normal_(m.weight, gain=1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.Sigmoid, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool1d, nn.Sigmoid, nn.Identity)):
            pass
        else:
            m.initialize()

class CamoFormer(torch.nn.Module):
    def __init__(self, cfg, load_path=None):
        super(CamoFormer, self).__init__()
        self.cfg = cfg
        self.encoder = pvt_v2_b4()
        self.decoder = Decoder(64)

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def forward(self, x, ref_x, stage, shape=None, name=None):

        features = self.encoder(x)
        ref_features = self.encoder(ref_x)
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]

        ref_x1 = ref_features[0]
        ref_x2 = ref_features[1]
        ref_x3 = ref_features[2]
        ref_x4 = ref_features[3]

        OutPut2= self.decoder(ref_x1, ref_x2, ref_x3, ref_x4, x)
        OutPut1= self.decoder(x1, x2, x3, x4, x)
        if stage in ['stage1', 'stage2', 'stage3']:
            return OutPut1, OutPut2
        elif stage == 'eval':
            return OutPut1