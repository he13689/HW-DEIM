"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
"""

import copy
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_activation

from ..core import register

__all__ = ['HybridEncoder']


class ConvNormLayer_fuse(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size - 1) // 2 if padding is None else padding
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            groups=g,
            padding=padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)
        self.ch_in, self.ch_out, self.kernel_size, self.stride, self.g, self.padding, self.bias = \
            ch_in, ch_out, kernel_size, stride, g, padding, bias

    def forward(self, x):
        if hasattr(self, 'conv_bn_fused'):
            y = self.conv_bn_fused(x)
        else:
            y = self.norm(self.conv(x))
        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv_bn_fused'):
            self.conv_bn_fused = nn.Conv2d(
                self.ch_in,
                self.ch_out,
                self.kernel_size,
                self.stride,
                groups=self.g,
                padding=self.padding,
                bias=True)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv_bn_fused.weight.data = kernel
        self.conv_bn_fused.bias.data = bias
        self.__delattr__('conv')
        self.__delattr__('norm')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor()

        return kernel3x3, bias3x3

    def _fuse_bn_tensor(self):
        kernel = self.conv.weight
        running_mean = self.norm.running_mean
        running_var = self.norm.running_var
        gamma = self.norm.weight
        beta = self.norm.bias
        eps = self.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size - 1) // 2 if padding is None else padding
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            groups=g,
            padding=padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


# TODO, add activation for cv1 following YOLOv10
# self.cv1 = Conv(c1, c2, 1, 1)
# self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)
class SCDown(nn.Module):
    def __init__(self, c1, c2, k, s, act=None):
        super().__init__()
        self.cv1 = ConvNormLayer_fuse(c1, c2, 1, 1)
        self.cv2 = ConvNormLayer_fuse(c2, c2, k, s, c2)

    def forward(self, x):
        return self.cv2(self.cv1(x))


class VGGBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        self.__delattr__('conv1')
        self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=False,
                 act="silu",
                 bottletype=VGGBlock):
        super(CSPLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            bottletype(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer_fuse(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_2 = self.conv2(x)
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        return self.conv3(x_1 + x_2)


class RepNCSPELAN4(nn.Module):
    # csp-elan
    def __init__(self, c1, c2, c3, c4, n=3,
                 bias=False,
                 act="silu"):
        super().__init__()
        self.c = c3 // 2
        self.cv1 = ConvNormLayer_fuse(c1, c3, 1, 1, bias=bias, act=act)
        self.cv2 = nn.Sequential(CSPLayer(c3 // 2, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock), ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act))
        self.cv3 = nn.Sequential(CSPLayer(c4, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock), ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act))
        self.cv4 = ConvNormLayer_fuse(c3 + (2 * c4), c2, 1, 1, bias=bias, act=act)

    def forward_chunk(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


# transformer
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class DSConv(nn.Module):
    """The Basic Depthwise Separable Convolution."""
    def __init__(self, c_in, c_out, k=3, s=1, p=None, d=1, bias=False):
        super().__init__()
        if p is None:
            p = (d * (k - 1)) // 2
        self.dw = nn.Conv2d(
            c_in, c_in, kernel_size=k, stride=s,
            padding=p, dilation=d, groups=c_in, bias=bias
        )
        self.pw = nn.Conv2d(c_in, c_out, 1, 1, 0, bias=bias)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return self.act(self.bn(x))


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization."""
        return self.act(self.conv(x))


class DSBottleneck(nn.Module):
    """
    An improved bottleneck block using depthwise separable convolutions (DSConv).

    This class implements a lightweight bottleneck module that replaces standard convolutions with depthwise
    separable convolutions to reduce parameters and computational cost.

    Attributes:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to use a residual shortcut connection. The connection is only added if c1 == c2. Defaults to True.
        e (float, optional): Expansion ratio for the intermediate channels. Defaults to 0.5.
        k1 (int, optional): Kernel size for the first DSConv layer. Defaults to 3.
        k2 (int, optional): Kernel size for the second DSConv layer. Defaults to 5.
        d2 (int, optional): Dilation for the second DSConv layer. Defaults to 1.

    Methods:
        forward: Performs a forward pass through the DSBottleneck module.

    Examples:
        >>> import torch
        >>> model = DSBottleneck(c1=64, c2=64, shortcut=True)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([2, 64, 32, 32])
    """
    def __init__(self, c1, c2, shortcut=True, e=0.5, k1=3, k2=5, d2=1):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = DSConv(c1, c_, k1, s=1, p=None, d=1)
        self.cv2 = DSConv(c_, c2, k2, s=1, p=None, d=d2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class DSC3k(C3):
    """
    An improved C3k module using DSBottleneck blocks for lightweight feature extraction.

    This class extends the C3 module by replacing its standard bottleneck blocks with DSBottleneck blocks,
    which use depthwise separable convolutions.

    Attributes:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of DSBottleneck blocks to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connections within the DSBottlenecks. Defaults to True.
        g (int, optional): Number of groups for grouped convolution (passed to parent C3). Defaults to 1.
        e (float, optional): Expansion ratio for the C3 module's hidden channels. Defaults to 0.5.
        k1 (int, optional): Kernel size for the first DSConv in each DSBottleneck. Defaults to 3.
        k2 (int, optional): Kernel size for the second DSConv in each DSBottleneck. Defaults to 5.
        d2 (int, optional): Dilation for the second DSConv in each DSBottleneck. Defaults to 1.

    Methods:
        forward: Performs a forward pass through the DSC3k module (inherited from C3).

    Examples:
        >>> import torch
        >>> model = DSC3k(c1=128, c2=128, n=2, k1=3, k2=7)
        >>> x = torch.randn(2, 128, 64, 64)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([2, 128, 64, 64])
    """
    def __init__(
        self,
        c1,
        c2,
        n=1,
        shortcut=True,
        g=1,
        e=0.5,
        k1=3,
        k2=5,
        d2=1
    ):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)

        self.m = nn.Sequential(
            *(
                DSBottleneck(
                    c_, c_,
                    shortcut=shortcut,
                    e=1.0,
                    k1=k1,
                    k2=k2,
                    d2=d2
                )
                for _ in range(n)
            )
        )


class FuseModule(nn.Module):
    """
    A module to fuse multi-scale features for the HyperACE block.

    This module takes a list of three feature maps from different scales, aligns them to a common
    spatial resolution by downsampling the first and upsampling the third, and then concatenates
    and fuses them with a convolution layer.

    Attributes:
        c_in (int): The number of channels of the input feature maps.
        channel_adjust (bool): Whether to adjust the channel count of the concatenated features.

    Methods:
        forward: Fuses a list of three multi-scale feature maps.

    """

    def __init__(self, c_in, channel_adjust):
        super(FuseModule, self).__init__()
        self.downsample = nn.AvgPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        if channel_adjust:
            self.conv_out = Conv(4 * c_in, c_in, 1)
        else:
            self.conv_out = Conv(3 * c_in, c_in, 1)

    def forward(self, x):
        x1_ds = self.downsample(x[0])
        x3_up = self.upsample(x[2])
        x_cat = torch.cat([x1_ds, x[1], x3_up], dim=1)
        out = self.conv_out(x_cat)
        return out


class AdaHyperedgeGen(nn.Module):
    """
    Generates an adaptive hyperedge participation matrix from a set of vertex features.

    This module implements the Adaptive Hyperedge Generation mechanism. It generates dynamic hyperedge prototypes
    based on the global context of the input nodes and calculates a continuous participation matrix (A)
    that defines the relationship between each vertex and each hyperedge.

    Attributes:
        node_dim (int): The feature dimension of each input node.
        num_hyperedges (int): The number of hyperedges to generate.
        num_heads (int, optional): The number of attention heads for multi-head similarity calculation. Defaults to 4.
        dropout (float, optional): The dropout rate applied to the logits. Defaults to 0.1.
        context (str, optional): The type of global context to use ('mean', 'max', or 'both'). Defaults to "both".

    Methods:
        forward: Takes a batch of vertex features and returns the participation matrix A.
    """

    def __init__(self, node_dim, num_hyperedges, num_heads=4, dropout=0.1, context="both"):
        super().__init__()
        self.num_heads = num_heads
        self.num_hyperedges = num_hyperedges
        self.head_dim = node_dim // num_heads
        self.context = context

        # 定义一个可学习的超图向量
        self.prototype_base = nn.Parameter(torch.Tensor(num_hyperedges, node_dim))
        nn.init.xavier_uniform_(self.prototype_base)
        if context in ("mean", "max"):
            self.context_net = nn.Linear(node_dim, num_hyperedges * node_dim)
        elif context == "both":
            self.context_net = nn.Linear(2 * node_dim, num_hyperedges * node_dim)
        else:
            raise ValueError(
                f"Unsupported context '{context}'. "
                "Expected one of: 'mean', 'max', 'both'."
            )

        self.pre_head_proj = nn.Linear(node_dim, node_dim)

        self.dropout = nn.Dropout(dropout)
        self.scaling = math.sqrt(self.head_dim)

    def forward(self, X):
        B, N, D = X.shape
        if self.context == "mean":
            context_cat = X.mean(dim=1)
        elif self.context == "max":
            context_cat, _ = X.max(dim=1)
        else:
            avg_context = X.mean(dim=1)
            max_context, _ = X.max(dim=1)
            context_cat = torch.cat([avg_context, max_context], dim=-1)
        # 获得超图偏置，超图就是图中点的一个高维特征含义
        prototype_offsets = self.context_net(context_cat).view(B, self.num_hyperedges, D)
        # self.prototype_base是基础的超图特征，是初始化的，是可学习参数
        prototypes = self.prototype_base.unsqueeze(0) + prototype_offsets

        X_proj = self.pre_head_proj(X)  # 做了一个线性学习
        # 多头特征
        X_heads = X_proj.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        # 多头超图
        proto_heads = prototypes.view(B, self.num_hyperedges, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # reshape，将头数和batch合并
        X_heads_flat = X_heads.reshape(B * self.num_heads, N, self.head_dim)
        proto_heads_flat = proto_heads.reshape(B * self.num_heads, self.num_hyperedges, self.head_dim).transpose(1, 2)

        # lagits = B * self.num_heads, N, self.num_hyperedges
        logits = torch.bmm(X_heads_flat, proto_heads_flat) / self.scaling
        # 每个特征的头求平均，就构建成了边
        logits = logits.view(B, self.num_heads, N, self.num_hyperedges).mean(dim=1)

        # 防过拟合
        logits = self.dropout(logits)

        # 最后用softmax将内积转为权重项（归一化）
        return F.softmax(logits, dim=1)


class AdaHGConv(nn.Module):
    """
    Performs the adaptive hypergraph convolution.

    This module contains the two-stage message passing process of hypergraph convolution:
    1. Generates an adaptive participation matrix using AdaHyperedgeGen.
    2. Aggregates vertex features into hyperedge features (vertex-to-edge).
    3. Disseminates hyperedge features back to update vertex features (edge-to-vertex).
    A residual connection is added to the final output.

    Attributes:
        embed_dim (int): The feature dimension of the vertices.
        num_hyperedges (int, optional): The number of hyperedges for the internal generator. Defaults to 16.
        num_heads (int, optional): The number of attention heads for the internal generator. Defaults to 4.
        dropout (float, optional): The dropout rate for the internal generator. Defaults to 0.1.
        context (str, optional): The context type for the internal generator. Defaults to "both".

    Methods:
        forward: Performs the adaptive hypergraph convolution on a batch of vertex features.

    """

    def __init__(self, embed_dim, num_hyperedges=16, num_heads=4, dropout=0.1, context="both"):
        super().__init__()
        self.edge_generator = AdaHyperedgeGen(embed_dim, num_hyperedges, num_heads, dropout, context)
        self.edge_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )
        self.node_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

    def forward(self, X):
        # 产生了边，即每个点和我高级特征（超图）之间的关系
        A = self.edge_generator(X)

        # 根据边，使用点特征点来更新超图，重算超图特征
        He = torch.bmm(A.transpose(1, 2), X)
        He = self.edge_proj(He)  # 线性学习

        # 用超图来更新点特征
        X_new = torch.bmm(A, He)
        X_new = self.node_proj(X_new)  # 线性学习

        return X_new + X  # 和原特征相加，经过超图特征后的特征


class AdaHGComputation(nn.Module):
    """
    A wrapper module for applying adaptive hypergraph convolution to 4D feature maps.

    This class makes the hypergraph convolution compatible with standard CNN architectures. It flattens a
    4D input tensor (B, C, H, W) into a sequence of vertices (tokens), applies the AdaHGConv layer to
    model high-order correlations, and then reshapes the output back into a 4D tensor.

    Attributes:
        embed_dim (int): The feature dimension of the vertices (equivalent to input channels C).
        num_hyperedges (int, optional): The number of hyperedges for the underlying AdaHGConv. Defaults to 16.
        num_heads (int, optional): The number of attention heads for the underlying AdaHGConv. Defaults to 8.
        dropout (float, optional): The dropout rate for the underlying AdaHGConv. Defaults to 0.1.
        context (str, optional): The context type for the underlying AdaHGConv. Defaults to "both".

    Methods:
        forward: Processes a 4D feature map through the adaptive hypergraph computation layer.

    """

    def __init__(self, embed_dim, num_hyperedges=16, num_heads=8, dropout=0.1, context="both"):
        super().__init__()
        self.embed_dim = embed_dim
        self.hgnn = AdaHGConv(
            embed_dim=embed_dim,
            num_hyperedges=num_hyperedges,
            num_heads=num_heads,
            dropout=dropout,
            context=context
        )

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.hgnn(tokens)
        x_out = tokens.transpose(1, 2).view(B, C, H, W)
        return x_out


class C3AH(nn.Module):
    """
    A CSP-style block integrating Adaptive Hypergraph Computation (C3AH).

    The input feature map is split into two paths.
    One path is processed by the AdaHGComputation module to model high-order correlations, while the other
    serves as a shortcut. The outputs are then concatenated to fuse features.

    Attributes:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        e (float, optional): Expansion ratio for the hidden channels. Defaults to 1.0.
        num_hyperedges (int, optional): The number of hyperedges for the internal AdaHGComputation. Defaults to 8.
        context (str, optional): The context type for the internal AdaHGComputation. Defaults to "both".

    Methods:
        forward: Performs a forward pass through the C3AH module.

    """

    def __init__(self, c1, c2, e=1.0, num_hyperedges=8, context="both"):
        super().__init__()
        c_ = int(c2 * e)
        assert c_ % 16 == 0, "Dimension of AdaHGComputation should be a multiple of 16."
        num_heads = c_ // 16
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = AdaHGComputation(embed_dim=c_,
                                  num_hyperedges=num_hyperedges,
                                  num_heads=num_heads,
                                  dropout=0.1,
                                  context=context)
        self.cv3 = Conv(2 * c_, c2, 1)

    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.cv2(x)
        return self.cv3(torch.cat((self.m(y1), y2), 1))


class HyperGraph(nn.Module):
    """
    Hypergraph-based Adaptive Correlation Enhancement (HyperACE).

    This is the core module of YOLOv13, designed to model both global high-order correlations and
    local low-order correlations. It first fuses multi-scale features, then processes them through parallel
    branches: two C3AH branches for high-order modeling and a lightweight DSConv-based branch for
    low-order feature extraction.

    Attributes:
        c1 (int): Number of input channels for the fuse module.
        c2 (int): Number of output channels for the entire block.
        n (int, optional): Number of blocks in the low-order branch. Defaults to 1.
        num_hyperedges (int, optional): Number of hyperedges for the C3AH branches. Defaults to 8.
        dsc3k (bool, optional): If True, use DSC3k in the low-order branch; otherwise, use DSBottleneck. Defaults to True.
        shortcut (bool, optional): Whether to use shortcuts in the low-order branch. Defaults to False.
        e1 (float, optional): Expansion ratio for the main hidden channels. Defaults to 0.5.
        e2 (float, optional): Expansion ratio within the C3AH branches. Defaults to 1.
        context (str, optional): Context type for C3AH branches. Defaults to "both".
        channel_adjust (bool, optional): Passed to FuseModule for channel configuration. Defaults to True.

    Methods:
        forward: Performs a forward pass through the HyperACE module.

    """

    def __init__(self, c1, c2, n=1, num_hyperedges=8, dsc3k=True, shortcut=False, e1=0.5, e2=1, context="both", channel_adjust=False):
        super().__init__()
        self.c = int(c2 * e1)
        self.cv1 = Conv(c1, 3 * self.c, 1, 1)
        self.cv2 = Conv((4 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            DSC3k(self.c, self.c, 2, shortcut, k1=3, k2=7) if dsc3k else DSBottleneck(self.c, self.c, shortcut=shortcut) for _ in range(n)
        )
        self.fuse = FuseModule(c1, channel_adjust)
        self.branch1 = C3AH(self.c, self.c, e2, num_hyperedges, context)
        self.branch2 = C3AH(self.c, self.c, e2, num_hyperedges, context)

    def forward(self, X):
        # input list, output [1 256 40 40]， 将输入的多尺度特征图进行fuse融合
        x = self.fuse(X)
        y = list(self.cv1(x).chunk(3, 1))
        out1 = self.branch1(y[1])
        out2 = self.branch2(y[1])
        y.extend(m(y[-1]) for m in self.m)
        y[1] = out1
        y.append(out2)
        return self.cv2(torch.cat(y, 1))


@register()
class HybridEncoder(nn.Module):
    __share__ = ['eval_spatial_size', ]

    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward=1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None,
                 version='dfine',
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides

        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            proj = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                ('norm', nn.BatchNorm2d(hidden_dim))
            ]))

            self.input_proj.append(proj)

        # encoder transformer
        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_act
        )

        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
        ])

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            # TODO, add activation for those lateral convs
            if version == 'dfine':
                self.lateral_convs.append(ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1))
            else:
                self.lateral_convs.append(ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                RepNCSPELAN4(hidden_dim * 2, hidden_dim, hidden_dim * 2, round(expansion * hidden_dim // 2), round(3 * depth_mult), act=act) \
                    if version == 'dfine' else CSPLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion, bottletype=VGGBlock)
            )

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                nn.Sequential(SCDown(hidden_dim, hidden_dim, 3, 2, act=act)) \
                    if version == 'dfine' else ConvNormLayer_fuse(hidden_dim, hidden_dim, 3, 2, act=act)
            )
            self.pan_blocks.append(
                RepNCSPELAN4(hidden_dim * 2, hidden_dim, hidden_dim * 2, round(expansion * hidden_dim // 2), round(3 * depth_mult), act=act) \
                    if version == 'dfine' else CSPLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion, bottletype=VGGBlock)
            )

        self.hyper_graph = HyperGraph(hidden_dim, hidden_dim*3)

        self.downsample = nn.AvgPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)
                # self.register_buffer(f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        """
        """
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def defuse(self, feats):
        feats_list = list(feats.chunk(3, 1))
        feats_list[-1] = self.downsample(feats_list[-1])
        feats_list[0] = self.upsample(feats_list[0])
        return feats_list

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # 加入hypervce [256 80 80， 256 40 40， 256 20 20] -> 256 40 40
        hg = self.hyper_graph(proj_feats)

        hg_list = self.defuse(hg)
        # 将超图加到原有features中
        proj_feats[0] += hg_list[0]
        proj_feats[1] += hg_list[1]
        proj_feats[2] += hg_list[2]

        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)

                memory: torch.Tensor = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        # broadcasting and fusion
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest')
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_height], dim=1))
            outs.append(out)

        print(len(outs))
        # 1 256 80 80
        # 1 256 40 40
        # 1 256 20 20
        return outs
