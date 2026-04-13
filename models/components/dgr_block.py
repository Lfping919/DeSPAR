import torch
import torch.nn as nn

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, dilation=1, use_bn=True, use_relu=True):
        super().__init__()
        bias = not use_bn
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True) if use_relu else nn.Identity()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DepthScaleAdapter(nn.Module):
    def __init__(self, in_channels, r):
        super().__init__()
        self.pus = nn.PixelUnshuffle(r)
        
        self.trans1 = ConvBNReLU(in_ch=1, out_ch=8, kernel_size=1)
        self.conv1 = ConvBNReLU(in_ch=8, out_ch=8, kernel_size=3, stride=1, padding=1)

        out_ch = in_channels // 2
        self.trans2 = ConvBNReLU(in_ch=8 * (r * r), out_ch=2 * out_ch, kernel_size=1)
        self.conv2 = ConvBNReLU(in_ch=2 * out_ch, out_ch=out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, depth):
        # [B, 1, H, W] -> [B, out_ch, H/r, W/r]
        depth = self.conv1(self.trans1(depth)) 
        depth = self.pus(depth) 
        depth = self.conv2(self.trans2(depth)) 
        return depth

class GuidanceRefinementBlock(nn.Module):
    def __init__(self, channels, stride=1):
        super().__init__()
        self.channels = channels
        self.stride = stride
        self.kernel_size = 3
        self.group_channels = channels // 2
        self.groups = channels // self.group_channels
        kernel_num = (self.kernel_size ** 2) * self.groups

        self.reduce = ConvBNReLU(in_ch=channels, out_ch=channels // 4, kernel_size=1, stride=stride)
        self.span = ConvBNReLU(in_ch=channels // 4, out_ch=kernel_num, kernel_size=1, stride=stride, 
                               use_bn=False, use_relu=False)

        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)

        self.unfold = nn.Unfold(self.kernel_size, 1, (self.kernel_size - 1) // 2, stride)
        self.BConv = ConvBNReLU(in_ch=channels, out_ch=channels, kernel_size=3, stride=1, padding=1)

    def forward(self, guide_feat, target_feat):
        kernel = guide_feat + target_feat 
        if self.stride > 1:
            kernel = self.avgpool(kernel)

        kernel = self.span(self.reduce(kernel)) 
        
        b, c, h, w = kernel.shape
        # [B, G, 1, kernel_size^2, H, W]
        kernel = kernel.view(b, self.groups, self.kernel_size ** 2, h, w).unsqueeze(2)

        target_unfolded = self.unfold(target_feat).view(b, self.groups, self.group_channels,
                                                        self.kernel_size ** 2, h, w)
        
        out = (kernel * target_unfolded).sum(dim=3).view(b, self.channels, h, w)
        return self.BConv(out)

class DGRBlock(nn.Module):
    def __init__(self, in_channels=64, pus_r=4):
        super().__init__()
        self.out_ch = in_channels // 2
        
        # 特征降维与深度适配
        self.f_trans = ConvBNReLU(in_ch=in_channels, out_ch=self.out_ch, kernel_size=1)
        self.dsa = DepthScaleAdapter(in_channels=in_channels, r=pus_r)

        # 阶段1: FG-DRM
        # 用 RGB 特征 (fi) 引导 深度特征 (di) 去噪
        self.fg_drm = GuidanceRefinementBlock(channels=self.out_ch, stride=1)
        
        # 阶段2: DG-FEM 
        # 用去噪后的深度特征 (d_reg) 增强 RGB特征 (fi)
        self.dg_fem = GuidanceRefinementBlock(channels=self.out_ch, stride=1)

        # 特征升维恢复
        self.reverse_trans = ConvBNReLU(in_ch=self.out_ch, out_ch=in_channels, kernel_size=1)

    def forward(self, x, depth):
        # x: [B, C, H, W], depth: [B, 1, 352, 352]
        fi = self.f_trans(x)         # [B, C/2, H, W]
        di = self.dsa(depth)         # [B, C/2, H, W]

        # 双向协同学习
        d_reg = self.fg_drm(guide_feat=fi, target_feat=di) 
        f_enh = self.dg_fem(guide_feat=d_reg, target_feat=fi) 

        return self.reverse_trans(f_enh)