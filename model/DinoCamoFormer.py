import torch
from .encoder.pvtv2_encoder import pvt_v2_b4
from .old2Decoder.decoder_camo_tea_stu_mamba import Decoder

import torch.nn as nn

import torch

class EMAUpdater:
    def __init__(self, model_student, model_teacher, momentum=0.996):
        """
        初始化 EMA 更新器
        :param model_student: 学生模型
        :param model_teacher: 教师模型
        :param momentum: EMA 动量参数，通常在 [0.99, 0.999] 之间
        """
        self.model_student = model_student
        self.model_teacher = model_teacher
        self.momentum = momentum
        self.model_teacher.eval()  # 教师模型通常不需要反向传播

    @torch.no_grad()
    def update_teacher(self):
        """
        用学生模型的参数更新教师模型（EMA）
        """
        student_dict = self.model_student.state_dict()
        teacher_dict = self.model_teacher.state_dict()
        for name, param_student in student_dict.items():
            if name in teacher_dict:
                param_teacher = teacher_dict[name]
                # 只更新浮点类型的参数，避免 long 类型的参数导致错误
                if param_teacher.dtype in (torch.float16, torch.float32, torch.float64):
                    teacher_dict[name].mul_(self.momentum).add_((1 - self.momentum) * param_student)
                # if param_teacher.dtype in (torch.long):
                #     print(name)
        self.model_teacher.load_state_dict(teacher_dict)


    # @torch.no_grad()
    # def update_student(self):
    #     """
    #     用教师模型的参数更新学生模型（EMA）
    #     """
    #     for param_student, param_teacher in zip(self.model_student.parameters(), self.model_teacher.parameters()):
    #         param_student.data.mul_(self.momentum).add_((1 - self.momentum) * param_teacher.data)

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
        self.studecoder = Decoder(64, 0)
        self.teadecoder = Decoder(64, 1)
        self.teadecoder.eval()
        for p in self.teadecoder.parameters():
            p.requires_grad = False

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

        OutPut2, OF2= self.studecoder(ref_x1, ref_x2, ref_x3, ref_x4)
        with torch.no_grad():
            OutPut1, OF1= self.teadecoder(x1, x2, x3, x4)
        if stage in ['stage1', 'stage2', 'stage3']:
            ema_updater = EMAUpdater(self.studecoder, self.teadecoder, momentum=0.996)
            ema_updater.update_teacher()
            return OutPut1, OutPut2, OF1, OF2
        elif stage == 'eval':
            return OutPut1