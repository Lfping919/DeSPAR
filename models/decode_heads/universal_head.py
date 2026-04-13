import torch
import torch.nn as nn
from models.components.prompt_ops import PromptGateGen

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.bn(self.conv(x))

class UniversalDecoder(nn.Module):
    """
    统一的级联解码器
    use_semantic_prompt=False: 执行 Stage 1 的基础特征相乘融合
    use_semantic_prompt=True:  执行 Stage 2 的 Prompt 门控自适应融合
    """
    def __init__(self, channel=32, dim=64, use_semantic_prompt=False):
        super().__init__()
        self.use_semantic_prompt = use_semantic_prompt
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample6 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_upsample2_1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2_2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_upsample2_3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv_concat4 = BasicConv2d(4 * channel, 4 * channel, 3, padding=1)

        self.conv4 = BasicConv2d(4 * channel, 4 * channel, 3, padding=1)
        self.conv5 = nn.Conv2d(4 * channel, 1, 1)

        # 仅在开启语义提示时实例化门控模块
        if self.use_semantic_prompt:
            self.gate2 = PromptGateGen(dim=dim, channel=channel, per_channel=False)
            self.gate3 = PromptGateGen(dim=dim, channel=channel, per_channel=False)
            self.gate4 = PromptGateGen(dim=dim, channel=channel, per_channel=False)

    def adaptive_fuse(self, a, b, alpha=None):
        if self.use_semantic_prompt and alpha is not None:
            # Stage 2: 门控融合
            return alpha * (a * b) + (1.0 - alpha) * 0.5 * (a + b)
        else:
            # Stage 1: 直接相乘
            return a * b

    def forward(self, x1, x2, x3, x4, cond=None):
        x1_1 = x1
        
        # 获取门控权重 (如果处于 Stage 1，则全为 None)
        alpha2 = self.gate2(cond) if self.use_semantic_prompt else None
        alpha3 = self.gate3(cond) if self.use_semantic_prompt else None
        alpha4 = self.gate4(cond) if self.use_semantic_prompt else None

        # 级联融合过程
        x2_1 = self.adaptive_fuse(self.conv_upsample1(self.upsample(x1)), x2, alpha2)
        
        ab_3 = self.adaptive_fuse(self.conv_upsample2(self.upsample(self.upsample(x1))),
                                  self.conv_upsample3(self.upsample(x2)), alpha3)
        x3_1 = self.adaptive_fuse(ab_3, x3, alpha3)

        ab_4 = self.adaptive_fuse(self.conv_upsample4(self.upsample(self.upsample(self.upsample(x1)))),
                                  self.conv_upsample5(self.upsample(self.upsample(x2))), alpha4)
        abc_4 = self.adaptive_fuse(ab_4, self.conv_upsample6(self.upsample(x3)), alpha4)
        x4_1 = self.adaptive_fuse(abc_4, x4, alpha4)

        # 后续拼接与输出
        x2_2 = self.conv_concat2(torch.cat((x2_1, self.conv_upsample2_1(self.upsample(x1_1))), 1))
        x3_2 = self.conv_concat3(torch.cat((x3_1, self.conv_upsample2_2(self.upsample(x2_2))), 1))
        x4_2 = self.conv_concat4(torch.cat((x4_1, self.conv_upsample2_3(self.upsample(x3_2))), 1))

        return self.conv5(self.conv4(x4_2))