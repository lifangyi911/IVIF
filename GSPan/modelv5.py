from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os

# 确保basicsr在Python路径中
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
sys.path.append(project_root)

import warnings
import math
import copy
from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_, kaiming_normal_



from einops import rearrange


import warnings
import math
import copy
from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_, kaiming_normal_
# from basicsr.utils.registry import ARCH_REGISTRY
from einops import rearrange
from torch.utils.checkpoint import checkpoint

# from basicsr.archs.arch_util import trunc_normal_
import numpy as np
from LWN import _as_wavelet, get_filter_tensors, DWT, IDWT
# from pytorch_wavelets import DWTForward, DWTInverse

import pywt


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)

@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale

class EDSRNOUP(nn.Module):
    def __init__(self,
                 num_in_ch=5,
                 num_feat=64,
                 num_block=6,
                 res_scale=1):
        super(EDSRNOUP, self).__init__()

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        # self.body = nn.Sequential(*[
        #     SKBlock(num_feat) for _ in range(num_block)
        # ])
        self.body = make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat, res_scale=res_scale, pytorch_init=True)
        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)


    def forward(self, x):

        x = self.conv_first(x)
        res = self.conv_after_body(self.body(x))
        x = res + x

        return res


class SKBlock(nn.Module):
    def __init__(self, dim, reduction=16, min_dim=32):
        super(SKBlock, self).__init__()
        mid_dim = max(dim // reduction, min_dim)

        # 分支 1: 标准 3x3 卷积
        self.conv3 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(dim),
            nn.GELU()
        )

        # 分支 2: 5x5 卷积 (用空洞卷积 dilation=2 实现，感受野 5x5 但参数量同 3x3)
        self.conv5 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=2, dilation=2, bias=False),
            LayerNorm2d(dim),
            nn.GELU()
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, mid_dim, bias=False),
            nn.GELU(),
            nn.Linear(mid_dim, dim * 2, bias=False)  # 输出两个分支的权重
        )

        self.softmax = nn.Softmax(dim=1)

        # 最后加一个 1x1 卷积融合并做残差连接
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        res = x
        batch, c, h, w = x.shape

        # 1. 生成并行特征
        fea3 = self.conv3(x)
        fea5 = self.conv5(x)

        # 2. 融合特征用于计算权重
        fea_u = fea3 + fea5
        fea_s = self.gap(fea_u).view(batch, c)

        # 3. 计算注意力向量
        attn = self.fc(fea_s).view(batch, 2, c, 1, 1)  # (B, 2, C, 1, 1)
        attn = self.softmax(attn)

        # 4. 选择性加权融合
        out = fea3 * attn[:, 0, :, :, :] + fea5 * attn[:, 1, :, :, :]

        return res + self.proj(out)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            'mean is more than 2 std from [a, b] in nn.init.trunc_normal_. '
            'The distribution of values may be incorrect.',
            stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        low = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [low, up], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * low - 1, 2 * up - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution.

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py

    The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class SWT2D(nn.Module):
    """
    Stationary Wavelet Transform (SWT) in PyTorch (GPU), supports any wavelet from pywt.
    Input:  (b, c, h, w)
    Output: tuple of 4 tensors (LL, HL, LH, HH), each (b, c, h, w)
    """

    def __init__(self, channels, wavelet="haar"):
        super().__init__()
        self.channels = channels
        self.wavelet = pywt.Wavelet(wavelet)

        # 获取一维分解滤波器（dec_lo, dec_hi），并反转以匹配 conv 的行为
        dec_lo = torch.tensor(self.wavelet.dec_lo[::-1], dtype=torch.float32)
        dec_hi = torch.tensor(self.wavelet.dec_hi[::-1], dtype=torch.float32)

        # 构造二维滤波器（外积）
        LL = torch.outer(dec_lo, dec_lo)  # kH x kW
        HL = torch.outer(dec_hi, dec_lo)
        LH = torch.outer(dec_lo, dec_hi)
        HH = torch.outer(dec_hi, dec_hi)

        # 堆叠成 (4,1,kH,kW)
        base_kernels = torch.stack([LL, HL, LH, HH], dim=0).unsqueeze(1)  # (4,1,kH,kW)

        # 扩展到 (4*channels, 1, kH, kW) 以便 groups=channels
        kernels = base_kernels.repeat(channels, 1, 1, 1)  # (4c,1,kH,kW)

        # 把权重注册为 buffer（随 model.to(device) 移动，不参与梯度）
        self.register_buffer("weight", kernels)

        # 记录 kernel 尺寸，并计算不对称 pad
        self.kH, self.kW = LL.shape
        pad_top = (self.kH - 1) // 2
        pad_bottom = (self.kH - 1) - pad_top
        pad_left = (self.kW - 1) // 2
        pad_right = (self.kW - 1) - pad_left
        # F.pad 的 pad 顺序是 (left, right, top, bottom)
        self.pad_tuple = (pad_left, pad_right, pad_top, pad_bottom)

    def forward(self, x):
        """
        x: (b, c, h, w)
        returns: LL, HL, LH, HH each (b, c, h, w)
        """
        b, c, h, w = x.shape
        assert c == self.channels, f"channels mismatch: {c} vs {self.channels}"

        # 先做不对称 padding，再用 groups 卷积（padding=0）
        x_padded = F.pad(x, self.pad_tuple, mode='reflect')  # (b,c,h+pad_top+pad_bottom, w+pad_left+pad_right)
        out = F.conv2d(x_padded, self.weight, stride=1, padding=0, groups=self.channels)  # (b,4c,H_out,W_out)
        # out 的 H_out/W_out 应当等于输入 h,w
        LL, HL, LH, HH = torch.chunk(out, 4, dim=1)  # each (b,c,h,w)
        return LL, HL, LH, HH

class FourierUnit(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.conv = nn.Sequential(
            nn.BatchNorm2d(dim * 2),
            nn.Conv2d(dim * 2, dim * 2, kernel_size=1, groups=1),
            nn.GELU(),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        fft = torch.fft.rfft2(x, norm='ortho')
        fft = torch.stack((fft.real, fft.imag), dim=-1)              # (b, c, h, w//2+1, 2)
        fft = rearrange(fft, 'b c h w d -> b (c d) h w')
        fft = self.conv(fft)
        fft = rearrange(fft, 'b (c d) h w -> b c h w d', d=2)
        fft = torch.complex(fft[..., 0], fft[..., 1])
        out = torch.fft.irfft2(fft, s=(h, w), norm='ortho')
        return out


class SpectralSpatialBlock(nn.Module): #	Spectral-Spatial Block
    """单层 PW-FNet 风格 Block + 您的方向引导"""
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)

        # Global Fourier Mixer（PW-FNet 核心）
        self.global_mixer = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1),
            nn.GELU(),
            FourierUnit(dim * 2),
            nn.Conv2d(dim * 2, dim, 1),
        )

        # FFN（PW-FNet 原版）
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim * 2, 3, padding=1, groups=dim * 2),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, 1),
        )

        self.beta  = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        res = x
        x = self.norm1(x)
        x = self.global_mixer(x)
        x = x * self.beta + res

        res = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x * self.gamma + res
        return x


class WFCM(nn.Module): #Wavelet-Fourier context Module

    def __init__(self, dim, depths=[4,4], wavelet="db3"):
        super().__init__()

        # -------- Stage1 (空域) --------
        self.stage1 = nn.Sequential(*[
            SpectralSpatialBlock(dim) for _ in range(depths[0])
        ])

        # -------- DWT / iDWT --------
        wt_type = wavelet
        self.wavelet = _as_wavelet(wt_type)
        dec_lo, dec_hi, rec_lo, rec_hi = get_filter_tensors(
            wt_type, flip=True
        )
        self.dec_lo = nn.Parameter(dec_lo, requires_grad=True)
        self.dec_hi = nn.Parameter(dec_hi, requires_grad=True)
        self.rec_lo = nn.Parameter(rec_lo.flip(-1), requires_grad=True)
        self.rec_hi = nn.Parameter(rec_hi.flip(-1), requires_grad=True)

        self.dwt = DWT(self.dec_lo, self.dec_hi, wavelet=wt_type, level=1)
        self.idwt = IDWT(self.rec_lo, self.rec_hi, wavelet=wt_type, level=1)
        # self.dwt = DWTForward(wave=wavelet)
        # self.idwt = DWTInverse(wave=wavelet)

        # 4 个子带合到 2dim → 更丰富的频域特征
        self.freq_proj = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 2, 1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim * 2, 1)
        )

        # -------- Stage2 (频域金字塔) --------
        self.stage2 = nn.Sequential(*[
            SpectralSpatialBlock(dim * 2) for _ in range(depths[1])
        ])

        # 将 stage2 的输出拆成 4 个子带用于 iSWT
        self.reconstruct_proj = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 4, 1),
            # nn.GELU()
        )

        # -------- Skip Fusion --------
        self.fuse = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1)
        )

    def forward(self, x ):
        res = x

        # ------- Stage1 -------
        x = self.stage1(x)

        # ------- DWT：频域金字塔 -------
        LL, (HL, LH, HH) = self.dwt(x)

        x_freq = torch.cat([LL, HL, LH, HH], dim=1)  # (B, 4C, H, W)
        x_freq = self.freq_proj(x_freq)              # (B, 2C, H, W)


        # ------- Stage2 -------
        x_freq = self.stage2(x_freq)

        # ------- 生成逆变换的 4 个子带 -------
        subbands = self.reconstruct_proj(x_freq)  # (B,4C,H,W)
        LL2, HL2, LH2, HH2 = torch.chunk(subbands, 4, dim=1)



        # ------- iDWT 重建 -------
        x_up = self.idwt([LL2, (HL2, LH2, HH2)])       # (B,C,H,W)

        # ------- res + fuse -------
        # x = torch.cat([x_up, res], dim=1)
        x = x_up + res

        res = x
        x = self.fuse(x) + res

        return x

class SHFM(nn.Module): #Stationary High-Frequency Module
    def __init__(self, inchannel=64, channel=180):
        super().__init__()

        # 1. SWT分解
        self.swt = SWT2D(channels=inchannel, wavelet="haar")
        # self.ChannelGatedDirection = contextTensorOrientation(in_channels=inchannel)

        # 2. 增强高频处理（方案1）
        self.hl_processor = nn.Sequential(
            nn.Conv2d(inchannel, channel // 4, 3, padding=1),
            nn.ReLU(),
            ResidualBlock(channel // 4)
        )
        self.lh_processor = nn.Sequential(
            nn.Conv2d(inchannel, channel // 4, 3, padding=1),
            nn.ReLU(),
            ResidualBlock(channel // 4)
        )
        self.hh_processor = nn.Sequential(
            nn.Conv2d(inchannel, channel // 4, 3, padding=1),
            nn.ReLU(),
            ResidualBlock(channel // 4)
        )

        # 跨分量注意力融合
        self.subband_interaction = nn.Sequential(
            nn.Conv2d(3 * (channel // 4), 3 * (channel // 4), 3, padding=1, groups=3),  # 空间建模
            nn.GELU(),
            nn.Conv2d(3 * (channel // 4), 3 * (channel // 4), 1),  # 通道/子带间建模
        )

        # 3. 渐进式高频净化（方案3）
        self.stage1_denoise = nn.Sequential(
            nn.Conv2d(3 * (channel // 4), channel, 3, padding=1),
            # nn.InstanceNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, padding=1)
        )

        self.stage2_enhance = nn.Sequential(
            ResidualBlock(channel),
            nn.Conv2d(channel, channel, 1)
        )

        # 可学习的噪声阈值
        self.noise_threshold = nn.Parameter(torch.tensor(0.4))
        self.mask_temp = nn.Parameter(torch.tensor(5.0))
        self.hf_scale = nn.Parameter(torch.tensor(0.3))
        # 数据统计
        self.noise_mask_mean = 0.0
        self.stats = {
            'hf_magnitude_mean': 0.0,
            'noise_threshold_value': 0.0,
            'noise_mask_mean': 0.0,
            'noise_mask_std': 0.0,
            # 'noise_mask_min':0.0,
            'mask_temp': 0.0,
            'hf_scale': 0.0,
            # 'noise_mask_max':0.0,
            'fully_suppressed': 0.0,
            'fully_retained': 0.0,

            # 'hf_enhanced_mean':0.0
        }

    def forward(self, srcs):
        """
        srcs: (b, inchannel, h, w)
        返回: 增强的高频特征 (b, channel, h, w)
        """
        # 1. SWT分解
        LL, HL, LH, HH = self.swt(srcs)

        # eps = 1e-8
        # gx_raw = (HL)  # (B, C, H, W)
        # gy_raw = (LH)  # (B, C, H, W)
        # hh_raw = (HH)  # (B, C, H, W)

        # dx,dy,conf = self.ChannelGatedDirection(srcs)
        #
        #
        # self.stats['dx_mean'] = torch.mean(dx).item()
        # self.stats['dx_std'] = torch.std(dx).item()
        # self.stats['dy_mean'] = torch.mean(dy).item()
        # self.stats['dy_std'] = torch.std(dy).item()
        #
        #
        # self.stats['conf_mean'] = torch.mean(conf).item()
        # self.stats['conf_std'] = torch.std(conf).item()



        # 2. 分别处理三个高频分量
        hl_feat = self.hl_processor(HL)  # (b, channel//4, h, w)
        lh_feat = self.lh_processor(LH)  # (b, channel//4, h, w)
        hh_feat = self.hh_processor(HH)  # (b, channel//4, h, w)

        # 3. 跨分量注意力融合
        hf_concat = torch.cat([hl_feat, lh_feat, hh_feat], dim=1)  # (b, 3c, h/2, w/2)
        hf_interacted = self.subband_interaction(hf_concat)

        # 4. 渐进式高频净化
        # 第一阶段：基础降噪
        hf_denoised = self.stage1_denoise(hf_interacted)  # (b, channel, h, w)

        # 基于幅度的噪声抑制
        hf_magnitude = torch.abs(hf_denoised)
        # self.stats['noise_threshold_value'] = self.noise_threshold.item()
        self.stats['mask_temp'] = self.mask_temp.item()
        self.stats['hf_scale'] = self.hf_scale.item()
        # self.stats['hf_magnitude_mean'] = torch.mean(hf_magnitude).item()
        temp = torch.clamp(self.mask_temp, min=0.1, max=6.0)
        noise_mask = 0.5 * (1.0 + torch.tanh(temp * (hf_magnitude - self.noise_threshold)))

        # self.stats['noise_mask_mean'] = torch.mean(noise_mask).item()
        # self.stats['noise_mask_std'] = torch.std(noise_mask).item()
        # self.stats['noise_mask_min'] = torch.min(noise_mask).item()
        # self.stats['noise_mask_max'] = torch.max(noise_mask).item()

        hf_cleaned = noise_mask * hf_denoised + (1 - noise_mask) * (0.05 * hf_denoised)
        fully_suppressed = torch.mean((noise_mask < 0.1).float()).item()
        fully_retained = torch.mean((noise_mask > 0.9).float()).item()
        # self.stats['fully_suppressed'] = fully_suppressed
        # self.stats['fully_retained'] = fully_retained

        # 第二阶段：结构增强
        hf_enhanced = self.stage2_enhance(hf_cleaned)  # (b, channel, h, w)
        hf_enhanced = hf_enhanced + hf_cleaned
        hf_enhanced = self.hf_scale * hf_enhanced
        # self.stats['hf_enhanced_mean'] = torch.mean(hf_enhanced).item()

        # 返回三者：增强高频 + 方向向量场 + 幅值图
        # return hf_enhanced, (dx, dy), conf

        return hf_enhanced


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        return out + res


def window_partition(x, window_size):
    # x is the feature from net_g
    b, c, h, w = x.shape
    windows = rearrange(x, 'b c (h_count dh) (w_count dw) -> (b h_count w_count) (dh dw) c', dh=window_size,
                        dw=window_size)
    # h_count = h // window_size
    # w_count = w // window_size
    # windows = x.reshape(b,c,h_count, window_size, w_count, window_size)
    # windows = windows.permute(0,1,2,4,3,5) #b,c,h_count,w_count,window_size,window_size
    # windows = windows.reshape(b,c,h_count*w_count, window_size * window_size)
    # windows = windows.permute(0,2,3,1) #b,h_count*w_count, window_size*window_size,c
    # windows = windows.reshape(-1, window_size*window_size, c)

    return windows


def with_pos_embed(tensor, pos):
    return tensor if pos is None else tensor + pos


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, act_layer=nn.ReLU):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            'mean is more than 2 std from [a, b] in nn.init.trunc_normal_. '
            'The distribution of values may be incorrect.',
            stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        low = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [low, up], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * low - 1, 2 * up - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution.

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py

    The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)



class WindowCrossAttn(nn.Module):
    def __init__(self, inchanel=64, dim=180, num_heads=6, window_size=12, num_gs_seed=2304):
        super(WindowCrossAttn, self).__init__()
        self.inchanel = inchanel
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.num_gs_seed = num_gs_seed
        self.num_gs_seed_sqrt = int(math.sqrt(num_gs_seed))

        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # # define a parameter table of relative position bias
        # self.relative_position_bias_table = nn.Parameter(
        #     torch.zeros((2 * self.num_gs_seed_sqrt - 1) * (2 * self.num_gs_seed_sqrt - 1), num_heads))
        # trunc_normal_(self.relative_position_bias_table, std=.02)

        # # get pair-wise relative position index for each token inside the window
        coords_source = (np.indices((self.num_gs_seed_sqrt, self.num_gs_seed_sqrt)) + 0.5) * self.window_size
        coords_tgt = (np.indices((self.window_size, self.window_size)) + 0.5) * self.num_gs_seed_sqrt
        coords_delta = coords_source.reshape(2, -1)[:, :, np.newaxis] - coords_tgt.reshape(2, -1)[:, np.newaxis, :]
        shape = coords_delta.shape
        indexed_coords_delta = coords_delta.flatten()
        indexed_coords_delta = list(map(list(set(sorted(indexed_coords_delta))).index, indexed_coords_delta))
        indexed_coords_delta = np.array(indexed_coords_delta).reshape(shape)

        indexed_coords_delta[0] *= indexed_coords_delta.max()
        indexed_coords_delta = torch.from_numpy(indexed_coords_delta.sum(0))
        self.register_buffer('relative_position_index', indexed_coords_delta)
        # assign closed feature bigger initial weight
        relative_position_bias_table = torch.zeros(
            (2 * max(self.num_gs_seed_sqrt, window_size) - 1) * (2 * max(self.num_gs_seed_sqrt, window_size) - 1),
            num_heads)
        # for source_pix, closest_tgt_pix in enumerate((coords_delta ** 2).sum(0).argmin(1)):
        #     rel_pos_idx = indexed_coords_delta[source_pix][closest_tgt_pix]
        #     relative_position_bias_table[rel_pos_idx] = 2
        self.relative_position_bias_table = nn.Parameter(relative_position_bias_table)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.qhead = nn.Linear(dim, dim, bias=True)
        self.khead = nn.Linear(dim, dim, bias=True)
        self.vhead = nn.Linear(dim, dim, bias=True)

        self.proj = nn.Linear(dim, dim)

        # softmax
        self.softmax = nn.Softmax(dim=-1)

        # direction basis per head: shape (num_heads, 2, head_dim)
        # B_{h,0}, B_{h,1} are learnable basis vectors used as:
        # d_h(theta) = cos2θ * B_{h,0} + sin2θ * B_{h,1}
        # self.dir_basis = nn.Parameter(torch.randn(num_heads, 2, head_dim) * 0.02)

        # independent q_dir projection is NOT required here because modulation uses q/k directly,
        # but we keep a small linear if you want to map GS into same scale (optional)
        # (we'll not use q_dir_proj for projection to dir space; modulation uses q/k and dir_basis)
        # head-wise learnable alpha
        # self.alpha = nn.Parameter(torch.ones(num_heads) * float(0.05))
        # self.last_modulation_range = (0.0, 0.0)

    def forward(self, gs, feat, dir_windows=None):
        # gs shape: b*h_count*w_count, num_gs, c    the input gs here should already include pos embedding and scale embedding
        # feat shape: b*h_count*w_count, dh*dw, c    dh=dw=window_size
        b_, num_gs, c = gs.shape
        b_, n, c = feat.shape

        q = self.qhead(gs)  # b_, num_gs_, c
        q = q.reshape(b_, num_gs, self.num_heads, c // self.num_heads)
        q = q.permute(0, 2, 1, 3)  # b_, num_heads, n, c // num_heads

        k = self.khead(feat)  # b_, n_, c
        k = k.reshape(b_, n, self.num_heads, c // self.num_heads)
        k = k.permute(0, 2, 1, 3)  # b_, num_heads, n, c // num_heads

        v = self.vhead(feat)  # b_, n_, c
        v = v.reshape(b_, n, self.num_heads, c // self.num_heads)
        v = v.permute(0, 2, 1, 3)  # b_, num_heads, n, c // num_heads

        q = q * self.scale
        attn_logits = (q @ k.transpose(-2, -1))  # (B_win, H, Nq, Nk)

        # # ==========================
        # #      Direction Module
        # # ==========================
        # if dir_windows is not None:
        #     dx = dir_windows[..., 0]  # (B,1,H,W)
        #     dy = dir_windows[..., 1]
        #     conf = dir_windows[..., 2]
        #
        #     # combine direction → (B,2,H,W)
        #     dir_map = torch.cat([dx, dy], dim=1)
        #
        #     # window partition → (Bwin,Nk,2)
        #     dir_tokens = window_partition(dir_map, self.window_size)
        #     conf_tokens = window_partition(conf, self.window_size).squeeze(-1)  # (Bwin,Nk)
        #
        #     # normalize direction vectors
        #     dir_tokens = F.normalize(dir_tokens, dim=-1)  # (Bwin,Nk,2)
        #
        #     # project into head subspace
        #     # dir_head: (Bwin, H, Nk, head_dim)
        #     dir_head = torch.einsum("bjd, hdc -> bhjc", dir_tokens, self.dir_basis)
        #     dir_head = F.normalize(dir_head, dim=-1)
        #
        #     # direction similarity modulation
        #     q_proj = torch.einsum("bhnc, bhjc -> bhjn", q, dir_head)  # (Bwin,H,Nq,Nk)
        #     # k_proj = torch.einsum("bhmc,bhkc->bhmk", k, dir_head)
        #     k_proj = torch.einsum("bhmc, bhmc -> bhm", k, dir_head).unsqueeze(2)  # (Bwin,H,Nq?,Nk)
        #
        #     dir_sim = q_proj * k_proj  # (Bwin,H,Nq,Nk)
        #     dir_sim = torch.tanh(dir_sim)
        #
        #     # conf gate (broadcast to heads & queries)
        #     conf_gate = conf_tokens.unsqueeze(1).unsqueeze(2)  # (Bwin,1,1,Nk)
        #     conf_gate = conf_gate.expand(-1, self.num_heads, num_gs, -1)  # (Bwin,H,Nq,Nk)
        #
        #     # modulation strength
        #     alpha = self.alpha.view(1, self.num_heads, 1, 1)
        #     modulation = (1.0+alpha * dir_sim * conf_gate)
        #
        #     attn_logits = attn_logits * (1 + alpha * dir_sim * conf_gate)
        #     self.last_modulation_range = (modulation.min().item(),modulation.max().item())
        #
        # # ==========================

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.num_gs_seed_sqrt * self.num_gs_seed_sqrt, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        # print(f"attn.norm(), {attn[0].norm()} \t relative_position_bias.norm(): {relative_position_bias.norm()}")
        attn = attn_logits + relative_position_bias.unsqueeze(0)  # * 100
        attn = self.softmax(attn)
        # always focus self
        # attn = attn * 0 +torch.eye(144, device=attn.device)[None,None]
        x = (attn @ v).transpose(1, 2).reshape(b_, num_gs, c)
        x = self.proj(x)

        return x




class WindowCrossAttnLayer(nn.Module):
    def __init__(self, dim=180, num_heads=6, window_size=12, shift_size=0, num_gs_seed=2308):
        super(WindowCrossAttnLayer, self).__init__()

        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.shift_size = shift_size
        self.window_size = window_size

        self.window_cross_attn = WindowCrossAttn(dim=dim, num_heads=num_heads, window_size=window_size,
                                                 num_gs_seed=num_gs_seed)

        self.mlp_crossattn_feature = MLP(in_features=dim, hidden_features=dim, out_features=dim)

    def forward(self, x, query_pos, feat, scale_embedding, dir_windows=None):
        # gs shape: b*h_count*w_count, num_gs, c
        # query_pos shape: b*h_count*w_count, num_gs, c
        # feat shape: b,c,h,w
        # scale_embedding shape: b*h_count*w_count, 1, c



        ###cross attention for Q,K,V
        resi = x
        x = self.norm3(x)
        if self.shift_size > 0:
            shift_feat = torch.roll(feat, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
            # shift_dir_windows = torch.roll(dir_windows, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        else:
            shift_feat = feat
            # shift_dir_windows = dir_windows
        shift_feat = window_partition(shift_feat, self.window_size)  # b*h_count*w_count, dh*dw, c  dh=dw=window_size



        x = self.window_cross_attn(with_pos_embed(x, query_pos),shift_feat)
        x = resi + x

        ###FFN
        resi = x
        x = self.norm4(x)
        x = self.mlp_crossattn_feature(x)
        x = resi + x

        return x


class WindowCrossAttnBlock(nn.Module):
    def __init__(self, dim=180, window_size=12, num_heads=6, num_layers=4, num_gs_seed=2308):
        super(WindowCrossAttnBlock, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.norm = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([
            WindowCrossAttnLayer(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2,
                num_gs_seed=num_gs_seed) for i in range(num_layers)
        ])

    def forward(self, x, query_pos, feat, scale_embedding, dir_windows=None):
        resi = x
        x = self.norm(x)
        for block in self.blocks:
            x = block(x, query_pos, feat, scale_embedding, dir_windows=dir_windows)
        x = self.mlp(x)
        x = resi + x
        return x


class GSSelfAttn(nn.Module):
    def __init__(self, dim=180, num_heads=6, num_gs_seed_sqrt=12):
        super(GSSelfAttn, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_gs_seed_sqrt = num_gs_seed_sqrt

        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.proj = nn.Linear(dim, dim)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.num_gs_seed_sqrt - 1) * (2 * self.num_gs_seed_sqrt - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.num_gs_seed_sqrt)
        coords_w = torch.arange(self.num_gs_seed_sqrt)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.num_gs_seed_sqrt - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.num_gs_seed_sqrt - 1
        relative_coords[:, :, 0] *= 2 * self.num_gs_seed_sqrt - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer('relative_position_index', relative_position_index)

        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.softmax = nn.Softmax(dim=-1)

        self.qhead = nn.Linear(dim, dim, bias=True)
        self.khead = nn.Linear(dim, dim, bias=True)
        self.vhead = nn.Linear(dim, dim, bias=True)


    def forward(self, gs):
        # gs shape: b*h_count*w_count, num_gs, c
        # pos shape: b*h_count*w_count, num_gs, c
        b_, num_gs, c = gs.shape

        q = self.qhead(gs)
        q = q.reshape(b_, num_gs, self.num_heads, c // self.num_heads)
        q = q.permute(0, 2, 1, 3)  # b_, num_heads, n, c // num_heads

        k = self.khead(gs)
        k = k.reshape(b_, num_gs, self.num_heads, c // self.num_heads)
        k = k.permute(0, 2, 1, 3)  # b_, num_heads, n, c // num_heads

        v = self.vhead(gs)
        v = v.reshape(b_, num_gs, self.num_heads, c // self.num_heads)
        v = v.permute(0, 2, 1, 3)  # b_, num_heads, n, c // num_heads

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # b_, num_heads, num_gs, n

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.num_gs_seed_sqrt * self.num_gs_seed_sqrt, self.num_gs_seed_sqrt * self.num_gs_seed_sqrt,
            -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)

        attn = (attn @ v).transpose(1, 2).reshape(b_, num_gs, c)
        attn = self.proj(attn)

        return attn


class GSSelfAttnLayer(nn.Module):
    def __init__(self, dim=180, num_heads=6, num_gs_seed_sqrt=12, shift_size=0):
        super(GSSelfAttnLayer, self).__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)


        self.gs_self_attn = GSSelfAttn(dim=dim, num_heads=num_heads, num_gs_seed_sqrt=num_gs_seed_sqrt)

        self.mlp_selfattn = MLP(in_features=dim, hidden_features=dim, out_features=dim)

        self.num_gs_seed_sqrt = num_gs_seed_sqrt
        self.shift_size = shift_size



    def forward(self, gs, pos, h_count, w_count, scale_embedding):
        # gs shape:b*h_count*w_count, num_gs_seed, channel
        # pos shape: b*h_count*w_count, num_gs_seed, channel
        # scale_embedding shape: b*h_count*w_count, 1, channel



        resi = gs
        gs = self.norm1(gs)

        #### shift gs
        if self.shift_size > 0:
            shift_gs = rearrange(gs, '(b m n) (h w) c -> b (m h) (n w) c', m=h_count, n=w_count,
                                 h=self.num_gs_seed_sqrt, w=self.num_gs_seed_sqrt)
            shift_gs = torch.roll(shift_gs, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shift_gs = rearrange(shift_gs, 'b (m h) (n w) c -> (b m n) (h w) c', m=h_count, n=w_count,
                                 h=self.num_gs_seed_sqrt, w=self.num_gs_seed_sqrt)
        else:
            shift_gs = gs

        #### gs self attention
        gs = self.gs_self_attn(shift_gs)

        #### shift gs back
        if self.shift_size > 0:
            shift_gs = rearrange(gs, '(b m n) (h w) c -> b (m h) (n w) c', m=h_count, n=w_count,
                                 h=self.num_gs_seed_sqrt, w=self.num_gs_seed_sqrt)
            shift_gs = torch.roll(shift_gs, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            shift_gs = rearrange(shift_gs, 'b (m h) (n w) c -> (b m n) (h w) c', m=h_count, n=w_count,
                                 h=self.num_gs_seed_sqrt, w=self.num_gs_seed_sqrt)
        else:
            shift_gs = gs

        gs = shift_gs + resi

        # FFN
        resi = gs
        gs = self.norm2(gs)
        gs = self.mlp_selfattn(gs)
        gs = gs + resi
        return gs


class SpectralReferentialCrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        # self.norm_kv = nn.LayerNorm(dim)
        self.norm_ffn = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim)
        ) # 典型的Transformer FFN

    def forward(self, x_q, x_kv):
        # x_q: gs (B, N_gs, C), pos_q: gs_pos (B, N_gs, C)
        # x_kv: gs_ori (B, N_gs, C)

        # MHA Block
        resi = x_q
        q = self.norm_q(x_q)
        # kv = self.norm_kv(x_kv)
        # K/V from gs_ori (no pos_embed on K/V for simplicity, as it shares position with Q)
        attn_out, _ = self.cross_attn(query=q, key=x_kv, value=x_kv)
        x_q = resi + attn_out # Residual connection for MHA

        # FFN Block
        resi = x_q
        x_q = self.ffn(self.norm_ffn(x_q))
        out = resi + x_q

        return out


class ChannelCrossAttention(nn.Module):
    """
    针对全色锐化优化的通道级交叉注意力。
    计算的是通道间的相关性 (C x C)，而不是像素间的相关性 (N x N)。
    """

    def __init__(self, dim, num_heads=6, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Linear(dim, dim, bias=bias)
        self.k = nn.Linear(dim, dim, bias=bias)
        self.v = nn.Linear(dim, dim, bias=bias)

        self.project_out = nn.Linear(dim, dim, bias=bias)

    def forward(self, x_q, x_kv):
        b, n, c = x_q.shape

        # 1. 生成 Q, K, V
        # x_q 来自 MS 流 (主流)，x_kv 来自 PAN 流 (参考流)
        q = self.q(x_q)
        k = self.k(x_kv)
        v = self.v(x_kv)

        # 2. 维度变换：将通道拆分到多头，并准备进行 C x C 乘法
        # (b, n, c) -> (b, head, c/head, n)
        q = rearrange(q, 'b n (h d) -> b h d n', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h d n', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h d n', h=self.num_heads)

        # 3. L2 归一化：增强训练稳定性，防止数值爆炸 (重要！)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # 4. 计算通道相关性矩阵 (C x C)
        # (b, h, d, n) @ (b, h, n, d) -> (b, h, d, d)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        # 5. 聚合信息
        # (b, h, d, d) @ (b, h, d, n) -> (b, h, d, n)
        out = (attn @ v)

        # 6. 转回原始形状
        out = rearrange(out, 'b h d n -> b n (h d)')
        out = self.project_out(out)
        return out


class ChannelInteractionModule(nn.Module):
    """
    完整的交互块：Norm -> CC-Attn -> Norm -> FFN
    """

    def __init__(self, dim, num_heads=6, ffn_expansion=2):
        super().__init__()
        # 注意力路径
        self.norm1_q = nn.LayerNorm(dim)
        self.norm1_kv = nn.LayerNorm(dim)
        self.attn = ChannelCrossAttention(dim, num_heads)

        # FFN 路径
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_expansion),
            nn.GELU(),
            nn.Linear(dim * ffn_expansion, dim)
        )

    def forward(self, x_q, x_kv):
        """
        x_q: MS 特征流 [B, N, C]
        x_kv: PAN 特征流 [B, N, C]
        """
        # 第一阶段：通道交叉注意力注入 (注入 PAN 的细节到 MS)
        # 使用残差连接保持原始光谱信息
        x_q = x_q + self.attn(self.norm1_q(x_q), self.norm1_kv(x_kv))

        # 第二阶段：非线性特征精炼
        x_q = x_q + self.ffn(self.norm2(x_q))

        return x_q

class GSSelfAttnBlock(nn.Module):
    def __init__(self, dim=180, num_heads=6, num_selfattn_layers=4, num_gs_seed_sqrt=12):
        super(GSSelfAttnBlock, self).__init__()

        self.num_gs_seed_sqrt = int(num_gs_seed_sqrt)

        self.mlp1 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([
            GSSelfAttnLayer(
                dim=dim,
                num_heads=num_heads,
                num_gs_seed_sqrt=num_gs_seed_sqrt,
                shift_size=0 if i % 2 == 0 else num_gs_seed_sqrt // 2
            ) for i in range(num_selfattn_layers)
        ])
        self.blocks_ori = nn.ModuleList([
            GSSelfAttnLayer(
                dim=dim,
                num_heads=num_heads,
                num_gs_seed_sqrt=num_gs_seed_sqrt,
                shift_size=0 if i % 2 == 0 else num_gs_seed_sqrt // 2
            ) for i in range(num_selfattn_layers)
        ])

        # self.streamfusion1 = ChannelInteractionModule(dim, num_heads)
        # self.streamfusion2 = ChannelInteractionModule(dim, num_heads)
        self.streamfusion1 = SpectralReferentialCrossAttention(dim, num_heads)
        self.streamfusion2 = SpectralReferentialCrossAttention(dim, num_heads)
        # self.sc_cross_attn = StreamFusionModule(dim, num_heads)
        # self.scale_injection1 = nn.Parameter(torch.zeros(1))  # 初始化为0，让网络慢慢打开交互
        # self.scale_injection2 = nn.Parameter(torch.zeros(1))

    def forward(self, gs, gs_ori, pos, h_count, w_count, scale_embedding):
        resi = gs
        resi_ori = gs_ori
        gs = self.norm1(gs)
        gs_ori = self.norm2(gs_ori)
        for block in self.blocks:
            gs = block(gs, pos, h_count, w_count, scale_embedding)
        for block in self.blocks_ori:
            gs_ori = block(gs_ori, pos, h_count, w_count, scale_embedding)
        gs_fuse = self.streamfusion1(x_q=gs, x_kv=gs_ori)

        gs_ori_fuse = self.streamfusion2(x_q=gs_ori, x_kv=gs)

        gs_fuse = self.mlp1(gs_fuse)
        gs_fuse = gs_fuse + resi
        gs_ori_fuse = self.mlp2(gs_ori_fuse)
        gs_ori_fuse = gs_ori_fuse + resi_ori



        return gs_fuse, gs_ori_fuse

class DSIABlock(nn.Module): # Dual-stream Interaction Attention Block
    def __init__(self, channel,num_gs_seed_sqrt,window_size,num_heads,num_crossattn_layers,num_gs_seed,
                 num_crossattn_blocks,num_crossattn_ori_layers,num_crossattn_ori_blocks,num_selfattn_layers,num_selfattn_blocks,use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.window_size = window_size
        self.window_crossattn_ori_blocks = nn.ModuleList([
            WindowCrossAttnBlock(dim=channel,
                                 window_size=window_size,
                                 num_heads=num_heads,
                                 num_layers=num_crossattn_layers,
                                 num_gs_seed=num_gs_seed) for i in range(num_crossattn_blocks)
        ])
        self.window_crossattn_blocks = nn.ModuleList([
            WindowCrossAttnBlock(dim=channel,
                                 window_size=window_size,
                                 num_heads=num_heads,
                                 num_layers=num_crossattn_ori_layers,
                                 num_gs_seed=num_gs_seed) for i in range(num_crossattn_ori_blocks)
        ])
        # self.streamfusion = SpectralReferentialCrossAttention(channel, num_heads)

        self.gs_selfattn_blocks = nn.ModuleList([
            GSSelfAttnBlock(dim=channel,
                            num_heads=num_heads,
                            num_selfattn_layers=num_selfattn_layers,
                            num_gs_seed_sqrt=num_gs_seed_sqrt
                            ) for i in range(num_selfattn_blocks)
        ])

    def forward(self, query, query_pos,query_context, feat, feat_context,scale_embedding,h,w):
        """
        x_content: 主流 query (B, N, C) - Local/RGB focus
        x_context: 辅流 query_context (B, N, C) - Global/Freq focus
        """


        for block in self.window_crossattn_blocks:
            if self.use_checkpoint:
                query = checkpoint(block, query, query_pos, feat, scale_embedding)
            else:
                query = block(query, query_pos, feat, scale_embedding)  # b*h_count*w_count, num_gs_seed, channel
        # resi1 = query



        for block in self.window_crossattn_ori_blocks:
            if self.use_checkpoint:
                query_context = checkpoint(block, query_context, query_pos, feat_context, scale_embedding)
            else:
                query_context = block(query_context, query_pos, feat_context,
                                  scale_embedding)  # b*h_count*w_count, num_gs_seed, channel

        skip_query = query
        skip_query_context = query_context

        # query = self.streamfusion(x_q=query, x_kv=query_context)

        for i, block in enumerate(self.gs_selfattn_blocks):
            if self.use_checkpoint:
                query, query_context = checkpoint(block, query, query_context, query_pos, h // self.window_size,
                                              w // self.window_size, scale_embedding)
            else:
                query, query_context = block(query, query_context, query_pos, h // self.window_size, w // self.window_size,
                                         scale_embedding)

        query = query + skip_query
        query_context = query_context + skip_query_context

        return query, query_context

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

# @ARCH_REGISTRY.register()
class STDSfea2gsV2B(nn.Module):
    def __init__(self, spectral_num=4,sensor=None, inchannel=64, channel=60, num_heads=6, num_crossattn_blocks=1, num_crossattn_layers=2, num_crossattn_ori_blocks=1, num_crossattn_ori_layers=2,
                 num_selfattn_blocks = 4, num_selfattn_layers = 4 ,depths=[4,4],num_dsiablocks=1,
                 num_gs_seed=64, gs_up_factor=1.0, window_size=8, img_range=1.0, shuffle_scale1 = 2, shuffle_scale2 = 1, use_checkpoint = False):
        super(STDSfea2gsV2B, self).__init__()
        self.channel = channel
        self.nhead = num_heads
        self.gs_up_factor = gs_up_factor
        self.num_gs_seed = num_gs_seed
        self.window_size = window_size
        self.num_selfattn_blocks = num_selfattn_blocks
        self.img_range = img_range
        self.use_checkpoint = use_checkpoint
        self.spectral_num = spectral_num
        self.sensor = sensor

        self.num_gs_seed_sqrt = int(math.sqrt(num_gs_seed))
        self.gs_up_factor_sqrt = int(math.sqrt(gs_up_factor))

        self.shuffle_scale1 = shuffle_scale1
        self.shuffle_scale2 = shuffle_scale2

        self.encoder_pan = EDSRNOUP(num_in_ch=2,num_feat=inchannel)
        self.encoder_lms = EDSRNOUP(num_in_ch=spectral_num,num_feat=inchannel)

        # shared gaussian embedding and its pos embedding
        # self.enhanced_swt = SHFM(inchannel, channel)
        # self.gaussian_refiner = WFCM(dim=inchannel, depths = depths)

        # HF -> query 映射
        # self.hf_in_norm = nn.InstanceNorm2d(channel, affine=False)  # 注意通道数变化
        # self.pool_mha = nn.MultiheadAttention(embed_dim=channel, num_heads=6, batch_first=True)
        self.content_embedding = nn.Parameter(torch.randn(self.num_gs_seed, channel))
        self.direction_embedding = nn.Parameter(torch.randn(self.num_gs_seed, channel))
        # self.hf_norm = nn.LayerNorm(channel)
        # self.hf_query_proj = nn.Sequential(
        #     nn.Conv2d(channel, channel, 1),  # 1x1卷积，保持维度
        #     nn.ReLU(inplace=True),
        #     ResidualBlock(channel)
        # )
        self.hf_stats = {
            'base_feat_mean': 0.0,
            'hf_enhanced_mean': 0.0,
            'fusion_gate_mean': 0.0,

        }

        self.pos_embedding = nn.Parameter(torch.randn(self.num_gs_seed, channel), requires_grad=True)
        trunc_normal_(self.content_embedding, std=.02)
        trunc_normal_(self.direction_embedding, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)
        # ---- context-tensor / sampling helpers ----
        # project input (srcs) -> intensity for ST (if srcs is feature map)
        # self.st_proj = nn.Conv2d(inchannel, 1, kernel_size=1, bias=True)

        # small smoothing kernel implemented via avg pooling (or conv)
        # we'll use avg_pool2d in forward for smoothing

        # global scale for mapping lambda -> sigma (learnable)
        # self.register_parameter('sigma_global_scale', nn.Parameter(torch.tensor(1.0)))  # alpha in formulas

        # residual scale for sigma and rho predictions (learnable small multipliers)
        # self.register_parameter('sigma_res_scale', nn.Parameter(torch.tensor(0.1)))
        # self.register_parameter('rho_res_scale', nn.Parameter(torch.tensor(0.1)))

        # small eps for numerical stability


        self.img_feat_proj = nn.Sequential(
            nn.Conv2d(inchannel, channel, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, 1, 1)
        )
        self.img_feat_context_proj = nn.Sequential(
            nn.Conv2d(inchannel, channel, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, 1, 1)
        )



        # self.ffn = nn.Sequential(
        #         nn.LayerNorm(channel),
        #         nn.Linear(channel, channel),
        #         nn.GELU(),
        #         nn.Linear(channel, channel)
        #     )


        self.dsiablocks = nn.ModuleList([
            DSIABlock( channel,self.num_gs_seed_sqrt,window_size,num_heads,num_crossattn_layers,
                     num_gs_seed,num_crossattn_blocks,num_crossattn_ori_layers,num_crossattn_ori_blocks,num_selfattn_layers,num_selfattn_blocks,use_checkpoint) for i in range(num_dsiablocks)
        ])
        # self.skip_norm = LayerNorm2d(channel)

        # 你的融合模块 (输入通道是 2*channel)
        # self.global_residual_fusion = nn.Sequential(
        #     nn.Conv2d(channel * 2, channel, 1, 1, 0),
        #     nn.GELU(),
        #     nn.Conv2d(channel, channel, 3, 1, 1) # 再加一层 3x3 增加一点空间融合能力
        # )

        # self.final_fusion_module = FrequencyModulationFusion(channel)


        # GS sigma_x, sigma_y
        self.mlp_block_sigma = nn.Sequential(
            nn.Linear(channel, channel),
            nn.ReLU(),
            nn.Linear(channel, channel * 4),
            nn.ReLU(),
            nn.Linear(channel * 4, int(2 * gs_up_factor))
        )

        # GS rho
        self.mlp_block_rho = nn.Sequential(
            nn.Linear(channel, channel),
            nn.ReLU(),
            nn.Linear(channel, channel * 4),
            nn.ReLU(),
            nn.Linear(channel * 4, int(1 * gs_up_factor))
        )

        # GS alpha
        self.mlp_block_alpha = nn.Sequential(
            nn.Linear(channel, channel),
            nn.ReLU(),
            nn.Linear(channel, channel * 4),
            nn.ReLU(),
            nn.Linear(channel * 4, int(1 * gs_up_factor))
        )

        # GS RGB values
        self.mlp_block_rgb = nn.Sequential(
            nn.Linear(channel, channel),
            nn.ReLU(),
            nn.Linear(channel, channel * 4),
            nn.ReLU(),
            nn.Linear(channel * 4, int(self.spectral_num * gs_up_factor))
        )

        # GS mean_x, mean_y
        self.mlp_block_mean = nn.Sequential(
            nn.Linear(channel, channel),
            nn.ReLU(),
            nn.Linear(channel, channel * 4),
            nn.ReLU(),
            nn.Linear(channel * 4, int(2 * gs_up_factor))
        )
        self.conv_after_body_Fusion = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, 1, 1)
        )


        self.UPNet = nn.Sequential(
            nn.Conv2d(channel, channel * self.shuffle_scale1 * self.shuffle_scale1, 3, 1, 1),
            # nn.ReLU(),
            nn.PixelShuffle(self.shuffle_scale1),
            nn.Conv2d(channel, channel * self.shuffle_scale2 * self.shuffle_scale2, 3, 1, 1),
            nn.PixelShuffle(self.shuffle_scale2)
        )

    @staticmethod
    def get_N_reference_points(h, w, device='cuda'):
        # step_y = 1/(h+1)
        # step_x = 1/(w+1)
        step_y = 1 / h
        step_x = 1 / w
        ref_y, ref_x = torch.meshgrid(torch.linspace(step_y / 2, 1 - step_y / 2, h, dtype=torch.float32, device=device),
                                      torch.linspace(step_x / 2, 1 - step_x / 2, w, dtype=torch.float32, device=device))
        reference_points = torch.stack((ref_x.reshape(-1), ref_y.reshape(-1)), -1)
        reference_points = reference_points[None, :, None]
        return reference_points




    def forward(self, lr_u, pan, scale=1):
        '''
        using deformable detr decoder for cross attention
        Args:
            query: (batch_size, num_query, dim)
            query_pos: (batch_size, num_query, dim)
            srcs: (batch_size, dim, h1, w1)
        '''

        # x = torch.concat([pan, lr_u], dim=1)
        pan_hp = pan - F.avg_pool2d(pan, kernel_size=5, stride=1, padding=2)
        pan_input = torch.cat([pan, pan_hp], dim=1)
        feat_pan = self.encoder_pan(pan_input)
        feat_lms = self.encoder_lms(lr_u)

        b, c, h, w = feat_lms.shape  ###srcs is pad to the size that could be divided by window_size

        # 1. 增强的SWT处理
        # hf_enhanced= self.enhanced_swt(feat_lms)  # hf_enhanced:(b,channel,h,w); dir_map:(b,3,h,w); mag_map:(b,h,w)
        # feat_context = self.gaussian_refiner(feat_pan)



        # 2. 生成query（基于增强的高频信息）
        # proj = self.hf_query_proj(hf_enhanced)  # (b, channel, h, w)
        # proj_windows = window_partition(proj, self.window_size)  # (N, L, channel)
        #
        # N, L, C = proj_windows.shape
        # seed_q = self.gs_embedding.unsqueeze(0).repeat(N, 1, 1)  # (N, num_gs_seed, channel)
        #
        # hf_query_res, attn_w = self.pool_mha(query=seed_q, key=proj_windows, value=proj_windows)
        # hf_query = seed_q + hf_query_res
        # query = self.hf_norm(hf_query)  # (N, num_gs_seed, channel)
        query = self.content_embedding.unsqueeze(0).unsqueeze(1).repeat(b, (h // self.window_size) * (w // self.window_size),1, 1)  # b, h_count*w_count, num_gs_seed, channel
        query = query.reshape(b * (h // self.window_size) * (w // self.window_size), -1,self.channel)  # b*h_count*w_count, num_gs_seed, channel

        # 3. 改进特征融合：让高频信息也影响feat
        feat = self.img_feat_proj(feat_lms)  # (b, channel, h, w)

        # skip_feat = feat  # 用于大跨度skip连接
        feat_context = self.img_feat_context_proj(feat_pan)

        query_context = self.direction_embedding.unsqueeze(0).unsqueeze(1).repeat(b, (h // self.window_size) * (
                w // self.window_size), 1, 1)  # b, h_count*w_count, num_gs_seed, channel
        query_context = query_context.reshape(b * (h // self.window_size) * (w // self.window_size), -1,
                                  self.channel)  # b*h_count*w_count, num_gs_seed, channel


        scale_embedding = None

        query_pos = self.pos_embedding.unsqueeze(0).unsqueeze(1).repeat(b, (h // self.window_size) * (
                w // self.window_size), 1, 1)  # b, h_count*w_count, num_gs_seed, channel

        query_pos = query_pos.reshape(b * (h // self.window_size) * (w // self.window_size), -1,
                                      self.channel)  # b*h_count*w_count, num_gs_seed, channel
        # self.hf_stats['feat'] = torch.mean(torch.abs(feat)).item()
        # self.hf_stats['feat_context'] = torch.mean(torch.abs(feat_context)).item()


        # hf = torch.cat([HL, LH, HH], dim=1)  # (b,3c,h,w)
        # hf = self.hf_in_norm(hf)
        # proj = self.hf_proj(hf)

        # feat = self.img_feat_proj(srcs)  # b*channel*h*w
        # feat_hf = torch.cat([feat, proj], dim=1)
        # feat = self.feat_fuse(feat_hf)
        # fuse_gate = self.hf_fuse_attn(torch.cat([feat, proj], dim=1))
        # feat = feat + fuse_gate * proj



        for i, block in enumerate(self.dsiablocks):
            if self.use_checkpoint:
                query, query_context = checkpoint(block, query, query_pos,query_context, feat, feat_context,scale_embedding,h,w)
            else:
                query, query_context = block(query, query_pos,query_context, feat, feat_context,scale_embedding,h,w)

        query = rearrange(query, '(b m n) (h w) c -> b c (m h) (n w)',
                          m=h // self.window_size, n=w // self.window_size, h=self.num_gs_seed_sqrt)
        query_context = rearrange(query_context, '(b m n) (h w) c -> b c (m h) (n w)',
                                  m=h // self.window_size, n=w // self.window_size, h=self.num_gs_seed_sqrt)
        query_map = self.conv_after_body_Fusion(torch.cat([query, query_context], 1))
        # query_context = query_context + resi_ori
        # query_context_map = rearrange(query_context, '(b m n) (h w) c -> b c (m h) (n w)',
        #                           m=h // self.window_size, n=w // self.window_size, h=self.num_gs_seed_sqrt)
        #
        # # 2. 【核心修改】: Concatenate + Fusion
        # # 将内容特征和方向特征拼接
        # merged_feat = torch.cat([query_map, query_context_map], dim=1)  # (B, 2C, H_gs, W_gs)
        #
        # # 融合降维
        # fused_query = self.final_fusion(merged_feat)  # (B, C, H_gs, W_gs)
        #大跨度skip
        # skip_feat_normed = self.skip_norm(skip_feat)
        #
        # # 4. Concat + Fuse
        # query_map = self.global_residual_fusion(torch.cat([skip_feat_normed, query_map], dim=1))
        # 3. UPNet 上采样
        query = self.UPNet(query_map)  # (B, C, H_up, W_up)

        query = query.permute(0, 2, 3, 1)



        # query = rearrange(query, '(b m n) (h w) c -> b m h n w c', m=h // self.window_size, n=w // self.window_size,
        #                   h=self.num_gs_seed_sqrt)

        query_sigma = self.mlp_block_sigma(query).reshape(b, -1, 2) # b, h_count*w_count*H*W, 2
        query_rho = self.mlp_block_rho(query).reshape(b, -1, 1)

        query_alpha = self.mlp_block_alpha(query).reshape(b, -1, 1)
        query_rgb = self.mlp_block_rgb(query).reshape(b, -1, self.spectral_num)
        query_mean = self.mlp_block_mean(query).reshape(b, -1, 2)
        # limit_cells = 6.0会崩溃
        if self.sensor == 'QB':
            limit_cells = 5.0 # qb有限制，其他没有
            query_mean = torch.tanh(query_mean) * limit_cells

        query_mean = query_mean / torch.tensor(
            [self.num_gs_seed_sqrt * (w // self.window_size) * self.shuffle_scale1 * self.shuffle_scale2,
             self.num_gs_seed_sqrt * (h // self.window_size) * self.shuffle_scale1 * self.shuffle_scale2])[
            None, None].to(query_mean.device)  # b, h_count*w_count*num_gs_seed, 2

        reference_offset = self.get_N_reference_points(
            self.num_gs_seed_sqrt * (h // self.window_size) * self.shuffle_scale1 * self.shuffle_scale2,
            self.num_gs_seed_sqrt * (w // self.window_size) * self.shuffle_scale1 * self.shuffle_scale2, pan.device)
        query_mean = query_mean + reference_offset.reshape(1, -1, 2)
        # query_mean = torch.clamp(query_mean, 0.001, 0.999)
        # self.hf_stats['query_sigx_max'] = torch.max(query_sigma[..., 0:1]).item()
        # self.hf_stats['query_sigy_max'] = torch.max(query_sigma[..., 1:2]).item()
        # self.hf_stats['query_rho_max'] = torch.max(query_rho).item()
        # self.hf_stats['query_rho_min'] = torch.min(query_rho).item()

        query = torch.cat([query_sigma, query_rho, query_alpha, query_rgb, query_mean],
                          dim=-1)  # b, h_count*w_count*num_gs_seed, 9

        return query


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 假设 float32 (4 bytes)
    size_mb = total * 4 / (1024 ** 2)
    return total, trainable, size_mb

if __name__ == '__main__':
    pan = torch.randn(1, 1, 64, 64).cuda()
    lr_u = torch.randn(1, 4, 64, 64).cuda()


    model = STDSfea2gsV2B().cuda()
    import time

    total, trainable, size_mb = count_parameters(model)
    print(f"模型: {model.__class__.__name__}")
    print(f"总参数量: {total:,}")
    print(f"可训练参数量: {trainable:,}")
    print(f"估计模型大小 (float32): {size_mb:.2f} MB")

    output = model(lr_u,pan,  )
    loss = output.sum()
    loss.backward()

    # 检查哪些参数没有梯度
    print("检查未使用的参数:")
    for i, (name, param) in enumerate(model.named_parameters()):
        if param.grad is None:
            print(f"❌ 参数 {i}: {name} - 没有梯度!")
        else:
            print(f"✅ 参数 {i}: {name} - 有梯度")