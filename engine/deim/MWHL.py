import torch
from torch import nn
import pywt
from torch.autograd import Function

from torch.nn.modules.utils import _pair, _quadruple
import torch.nn.functional as F

# -------------------------- 对应结构图【ResBlock】残差特征增强模块 --------------------------
class ResidualLeakBlock(nn.Module):
    """
    带LeakyReLU的残差卷积块
    对应结构图(a)中的ResBlock模块，核心功能：对融合后的下采样特征做增强，缓解梯度消失
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 输入投影卷积，统一通道维度
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
        # 残差主体卷积，5×3卷积捕捉多尺度特征
        self.body = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.proj(x)
        residual = x
        x = self.body(x)
        # 残差连接，避免下采样带来的信息丢失
        out = self.relu(x + residual)
        return out

# -------------------------- 可微二维离散小波变换（DWT）核心实现 --------------------------
class DWT_Function(Function):
    """
    基于PyTorch Autograd的可微DWT前向/反向传播实现
    核心功能：将输入特征解耦为LL(低频结构)、LH(水平高频)、HL(垂直高频)、HH(对角高频)四个子带，支持端到端训练
    """
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        x = x.contiguous()
        # 保存反向传播所需的滤波器与输入形状
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape
        dim = x.shape[1]
        # 分组卷积实现小波分解，步长2实现2倍下采样
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        # 四个子带沿通道拼接
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            B, C, H, W = ctx.shape
            dx = dx.view(B, 4, -1, H // 2, W // 2)
            dx = dx.transpose(1, 2).reshape(B, -1, H // 2, W // 2)
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            # 转置卷积实现反向传播的上采样
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=C)
        return dx, None, None, None, None

# -------------------------- 可微二维逆离散小波变换（IDWT）核心实现 --------------------------
class IDWT_Function(Function):
    """可微IDWT实现，与DWT配对实现小波域的特征重建"""
    @staticmethod
    def forward(ctx, x, filters):
        ctx.save_for_backward(filters)
        ctx.shape = x.shape
        B, _, H, W = x.shape
        x = x.view(B, 4, -1, H, W).transpose(1, 2)
        C = x.shape[1]
        x = x.reshape(B, -1, H, W)
        filters = filters.repeat(C, 1, 1, 1)
        # 转置卷积实现逆小波变换，2倍上采样
        x = torch.nn.functional.conv_transpose2d(x, filters, stride=2, groups=C)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors[0]
            B, C, H, W = ctx.shape
            C = C // 4
            dx = dx.contiguous()
            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return dx, None

# -------------------------- IDWT封装模块 --------------------------
class IDWT_2D(nn.Module):
    """2D逆小波变换封装，支持自定义小波基"""
    def __init__(self, wave):
        super(IDWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)
        # 构建二维重构滤波器
        w_ll = rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)
        w_ll = w_ll.unsqueeze(0).unsqueeze(1)
        w_lh = w_lh.unsqueeze(0).unsqueeze(1)
        w_hl = w_hl.unsqueeze(0).unsqueeze(1)
        w_hh = w_hh.unsqueeze(0).unsqueeze(1)
        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
        self.register_buffer('filters', filters)
        self.filters = self.filters.to(dtype=torch.float32)

    def forward(self, x):
        return IDWT_Function.apply(x, self.filters)

# -------------------------- DWT封装模块 对应结构图【DWT】模块 --------------------------
class DWT_2D(nn.Module):
    """2D离散小波变换封装，对应结构图(a)中的DWT模块，支持自定义小波基"""
    def __init__(self, wave):
        super(DWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])
        # 构建二维分解滤波器
        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)
        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))
        self.w_ll = self.w_ll.to(dtype=torch.float32)
        self.w_lh = self.w_lh.to(dtype=torch.float32)
        self.w_hl = self.w_hl.to(dtype=torch.float32)
        self.w_hh = self.w_hh.to(dtype=torch.float32)

    def forward(self, x):
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)

class MixPool2d(nn.Module):
    """ 将最大池化层和均值池化层混和的混合池化层

    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """

    def __init__(self, kernel_size=2, stride=2, padding=0, same=False):
        super(MixPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        x = F.pad(x, self._padding(x), mode='reflect')
        mean = x.mean()  # 全局图片均值
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,))
        x1 = x.mean(dim=-1)[0]
        x2 = x.max(dim=-1)[0]

        x = torch.where(x1 > mean, x1, x2)
        return x2

# -------------------------- MWHL核心模块 完整对应结构图(a)全流程 --------------------------
class MWHL(nn.Module):
    """
    MWHL: Max pooling-Wavelet Hybrid Layer 最大池化-小波混合下采样层
    完整对应结构图(a)全流程，核心功能：双分支互补实现2倍下采样，同时保留语义特征与结构细节
    输入：[B, C, H, W]，输出：[B, C, H/2, W/2]
    """
    def __init__(self, channel=32, wave = 'haar'):
        super().__init__()
        # 小波变换模块 对应结构图DWT
        self.dwt = DWT_2D(wave)
        self.channel = channel
        # 最大池化模块 对应结构图MaxPool
        self.maxpool = nn.MaxPool2d(2,2)
        self.mixpool = MixPool2d(2,2)
        # 1×1卷积降维，融合双分支特征
        self.conv_D = nn.Conv2d(channel*2, channel, kernel_size=1, stride=1, padding=0)
        # 残差增强模块 对应结构图ResBlock
        self.ResBlock = ResidualLeakBlock(channel, channel)

    def forward(self, x):
        # 步骤1：小波变换分解，得到4个子带，对应结构图DWT分支
        e_dwt = self.dwt(x)
        # 拆分4个子带，取LL低频结构分量（尺寸C×H/2×W/2）
        e_ll, e_lh, e_hl, e_hh = e_dwt.split(self.channel, 1)
        # 步骤2：最大池化下采样，得到语义特征（尺寸C×H/2×W/2），对应结构图MaxPool分支
        e_down = self.maxpool(x)
        # 步骤3：双分支特征拼接，对应结构图C拼接环节
        e2 = self.conv_D(torch.cat([e_ll, e_down], dim=1))
        # 步骤4：残差块特征增强，对应结构图ResBlock环节
        out = self.ResBlock(e2)
        return out

# 模块测试代码
if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(1, 64, 32, 32).to(device)
    model = MWHL(64).to(device)
    y = model(x)

    print("输入特征维度：", x.shape)
    print("输出特征维度：", y.shape)