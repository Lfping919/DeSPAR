import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.backbones.pvtv2_dgl import PVTv2_DGL_Encoder
from models.decode_heads.universal_head import UniversalDecoder, BasicConv2d
from models.components.prompt_ops import SemanticPromptBank, PC_FiLM, PGCA

class DeSPARNetwork(nn.Module):
    def __init__(self, 
                 use_semantic_prompt=False, 
                 class_num=None, 
                 prompt_dim=512,
                 channel=32, 
                 pretrained_pvtv2_pth='/home/lfp/projects/DeSPAR/weights/pvt_v2_b2.pth',
                 prompt_init_path=None):
        super().__init__()
        
        self.use_semantic_prompt = use_semantic_prompt
        self.prompt_dim = prompt_dim

        # 1. 初始化深度引导编码器
        self.encoder = PVTv2_DGL_Encoder()
        if pretrained_pvtv2_pth:
            self._load_pretrained_encoder(pretrained_pvtv2_pth)

        # 2. 通道归一化层
        self.norm1 = BasicConv2d(64, channel, 3, 1, 1)
        self.norm2 = BasicConv2d(128, channel, 3, 1, 1)
        self.norm3 = BasicConv2d(320, channel, 3, 1, 1)
        self.norm4 = BasicConv2d(512, channel, 3, 1, 1)

        # 3. 动态加载 Stage 2 语义组件
        if self.use_semantic_prompt:
            assert class_num is not None, "Stage 2 requires class_num."
            init_center = np.load(prompt_init_path) if prompt_init_path else None
            
            self.prompt_bank = SemanticPromptBank(num_classes=class_num, dim=prompt_dim, init_center=init_center)
            self.project_stage4 = nn.Linear(512, prompt_dim) if prompt_dim == 64 else nn.Identity()
            
            # FiLM 调制器
            self.film1 = PC_FiLM(dim=prompt_dim, out_channel=channel)
            self.film2 = PC_FiLM(dim=prompt_dim, out_channel=channel)
            self.film3 = PC_FiLM(dim=prompt_dim, out_channel=channel)
            self.film4 = PC_FiLM(dim=prompt_dim, out_channel=channel)
            
            # 交叉注意力细化
            self.crossattn3 = PGCA(channel=channel, dim=prompt_dim, num_queries=2, num_heads=1)
            self.crossattn4 = PGCA(channel=channel, dim=prompt_dim, num_queries=4, num_heads=2)

        # 4. 统一解码器
        self.decoder = UniversalDecoder(channel=channel, dim=prompt_dim, use_semantic_prompt=self.use_semantic_prompt)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

    def _load_pretrained_encoder(self, pth_path):
        save_model = torch.load(pth_path, map_location='cpu')
        model_dict = self.encoder.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.encoder.load_state_dict(model_dict)

    def forward(self, x, depth, batch_labels=None):
        # 1. 提取多尺度几何特征
        features = self.encoder(x, depth)
        x1, x2, x3, x4 = features[0], features[1], features[2], features[3]

        # 2. 通道归一化
        x1_nor = self.norm1(x1)
        x2_nor = self.norm2(x2)
        x3_nor = self.norm3(x3)
        x4_nor = self.norm4(x4)

        cond = None
        
        # 3. 语义提示介入 (仅 Stage 2)
        if self.use_semantic_prompt:
            if batch_labels is None:
                x4_gap = F.adaptive_avg_pool2d(x4, (1, 1)).flatten(1)
                proj = F.normalize(self.project_stage4(x4_gap), p=2, dim=1) if self.prompt_dim == 512 else self.project_stage4(x4_gap)
                cond, _ = self.prompt_bank.soft_mix(feat=proj, topk=2)
            else:
                cond = self.prompt_bank.get_prompt_by_label(batch_labels)
            
            # 应用通道调制
            x1_nor = self.film1.apply_film(x1_nor, cond)
            x2_nor = self.film2.apply_film(x2_nor, cond)
            x3_nor = self.film3.apply_film(x3_nor, cond)
            x4_nor = self.film4.apply_film(x4_nor, cond)

            # 应用空间交叉注意力
            x3_nor = self.crossattn3(x3_nor, cond)
            x4_nor = self.crossattn4(x4_nor, cond)

        # 4. 解码与上采样输出
        pred = self.decoder(x4_nor, x3_nor, x2_nor, x1_nor, cond)
        prediction = self.upsample_4(pred)

        # 为了兼容不同的 Loss 计算，返回 (预测图, Sigmoid图, 顶层特征)
        return prediction, torch.sigmoid(prediction), x4