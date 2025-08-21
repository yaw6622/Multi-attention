#This file contains an old version codes of MAI Network, which includes branches of 3 attention mechanisms and a regular decoder. The code of AFFN is separately put in the file AFFN.py. We will
#update the new version of MAI Network with AFFN included ASAP.

import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple
import torch.nn.functional as F
import math
from My_function.module import Attention, PreNorm, FeedForward
from einops import rearrange, repeat
import torchvision.models as models
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_channels=3, embed_dim=768):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0]//2, patch_size[1]//2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):  # x (b, c, h, w)
        x = self.proj(x)   # x (b, embed, h/4, w/4) default
        _, _, H, W = x.shape #H,W ：横纵向分割小块的个数
        x = x.flatten(2).transpose(1, 2) # x (b, h*w/16, embed)
        x = self.norm(x) # x (b, h*w/16, embed) (b, n, c)
        return x, H, W

# class PatchEmbed(nn.Module):
#     r""" Image to Patch Embedding
#     Args:
#         patch_size (int): Patch token size. Default: 4.
#         in_chans (int): Number of input image channels. Default: 3.
#         embed_dim (int): Number of linear projection output channels. Default: 96.
#         norm_layer (nn.Module, optional): Normalization layer. Default: None
#     """
#
#     def __init__(self, img_size=224, patch_size=16, stride=16, in_channels=3, embed_dim=768, norm_layer=None,
#                  flatten=None):
#         super(PatchEmbed, self).__init__()
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)
#         self.img_size = img_size
#         self.patch_size = patch_size
#
#         self.grid_size = ((img_size[0] - patch_size[0]) // stride + 1, (img_size[0] - patch_size[0]) // stride + 1)
#         self.num_patches = self.grid_size[0] * self.grid_size[1]
#
#         self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=stride)
#         self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
#
#     def forward(self, x):
#         x = self.proj(x)  #B,d,h,w
#         x = x.permute(0, 2, 3, 1)  #B,h,w,d
#         x = self.norm(x)
#         # x = x.permute(0, 3, 1, 2)
#         return x

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class Final_PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()

        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale*self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//self.dim_scale)
        x= self.norm(x)


        return x

class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            # d_state="auto", # 20240109
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        # B, L, C = x.shape
        # K = 4
        #
        # # 将输入张量调整为适合后续计算的形状
        # x_hwwh = x.view(B, 1, C, L).expand(B, 2, C, L)  # 扩展为 2 个通道
        # xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, c, l)
        #
        # x_dbl = torch.einsum("b k c l, k c d -> b k d l", xs, self.x_proj_weight)
        # dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        # dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)
        #
        # xs = xs.view(B, -1, L)  # (b, k * c, l)
        # dts = dts.contiguous().view(B, -1, L)  # (b, k * d, l)
        # Bs = Bs.view(B, K, -1, L)  # (b, k, d_state, l)
        # Cs = Cs.view(B, K, -1, L)  # (b, k, d_state, l)
        # Ds = self.Ds.view(-1)  # (k * d)
        # As = -torch.exp(self.A_logs).view(-1, self.d_state)  # (k * d, d_state)
        # dt_projs_bias = self.dt_projs_bias.view(-1)  # (k * d)
        #
        # out_y = self.selective_scan(
        #     xs, dts,
        #     As, Bs, Cs, Ds, z=None,
        #     delta_bias=dt_projs_bias,
        #     delta_softplus=True,
        #     return_last_state=False,
        # ).view(B, K, -1, L)
        #
        # inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        # wh_y = out_y[:, 1].view(B, -1, L)
        # invwh_y = inv_y[:, 1].view(B, -1, L)
        #
        # return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y
        # self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
    #     B, L, C = x.shape
    #
    #     xz = self.in_proj(x)
    #     x, z = xz.chunk(2, dim=-1)  # (b, l, d)
    #
    #     x = x.permute(0, 2, 1).contiguous()  # 转换为 (b, d, l)
    #     x = self.act(self.conv2d(x.unsqueeze(2)))  # 需要 unsqueeze 以适应卷积层的输入
    #     y1, y2, y3, y4 = self.forward_core(x)
    #
    #     y = y1 + y2 + y3 + y4
    #     y = y.permute(0, 2, 1).contiguous().view(B, L, -1)
    #     y = self.out_norm(y)
    #     y = y * F.silu(z)
    #     out = self.out_proj(y)
    #
    #     if self.dropout is not None:
    #         out = self.dropout(out)

        # return out
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer=nn.LayerNorm,
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x

class DWConv(nn.Module):  # 使用卷积获取位置信息，融合在mix-ffn里
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):  # x (b, n, c)
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)  # x (b, c, h, w)
        x = self.dwconv(x)  # x(b, dim, h, w)
        x = x.flatten(2).transpose(1, 2)  # x (b, n, dim)

        return x

class EfficientSelfAttention(nn.Module):  # 
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5  # sqrt


        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.sr_ratio = sr_ratio

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)  # reduction
            self.norm = nn.LayerNorm(dim)

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x, H, W):  # x (b, n, c)
        B, N, C = x.shape
        # q (b, n, c) -> (b, n, heads, c/heads) -> (b, heads, n, c/heads) 产生多个头
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)


        if self.sr_ratio > 1:  # reduction 维度收缩
            # x_ (b, c, n) -> (b, c, h, w)
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            # x_ (b, c, h/r, w/r) -> (b, c, h*w/R) -> (b, h*w/R, c)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)  #
            x_ = self.norm(x_)
            # kv (b, n/R, c*2) -> (b, n/R, 2, heads, c/heads) -> (2, b, heads, n/R, c/heads)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            # kv (b, h*w, c*2) -> (b, n, 2, heads, c/heads) -> (2, b, heads, n, c/heads)
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        # k, v (b, heads, n/R, c/heads)
        k, v = kv[0], kv[1]

        # attn (b, heads, n, c/heads) * (b, heads, c/heads, n/R) -> (b, heads, n, n/R)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # x (b, heads, n, n/R) * (b, heads, n/R, c/heads) -> (b, heads, n, c/heads)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # x (b, n, heads, c/heads) -> (b, n, c)

        x = self.proj(x)
        x = self.proj_drop(x)

        # x (b, n, c)
        return x

class FFN(nn.Module):
    def __init__(self, in_features, hidden_feaures=None, out_features=None, act_layer=nn.GELU(), drop=0.):
        super().__init__()

        out_features = out_features or in_features
        hidden_feaures = hidden_feaures or in_features
        self.fc1 = nn.Linear(in_features, hidden_feaures)
        self.dwconv = DWConv(hidden_feaures)
        self.act = act_layer
        self.fc2 = nn.Linear(hidden_feaures, out_features)
        self.drop = nn.Dropout(drop)


    def forward(self, x, H, W):  # x (b, n, in_c)
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x  # x (b, n, out_c)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU(), norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)  # (b, n, c) -> (b, n, c)

        self.attn = EfficientSelfAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                           attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)

        self.norm2 = norm_layer(dim)
        self.mlp = FFN(in_features=dim, hidden_feaures=(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        # 将多分支结构的子路径"随机删除"
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, H, W):  # x (b, n, c)
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))  # x (b, n, c)
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x  # (b, n, c)

class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.atrous_block1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.atrous_block6 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18)
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv1x1 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.atrous_block1(x)
        x2 = self.atrous_block6(x)
        x3 = self.atrous_block12(x)
        x4 = self.atrous_block18(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.conv1x1(x)
        x = self.bn(x)
        return self.relu(x)

class AdaptATT(nn.Module):
    def __init__(self, img_size=[224,224], patch_size=4, stride=4, in_chans=3, Transformer_depth=[1],Mamba_depth=[4], num_classes=1000, embed_dim=[128], d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True, resolution_indicator = 0.5, heads = [2], dim_head = 16, scale_dim = 4,
                 mlp_ratios = [4],use_checkpoint=False, qkv_bias=False, qk_scale=None, sr_ratios=[8],**kwargs):
        super(AdaptATT, self).__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim[0]
        self.patch_size = patch_size
        self.img_size = img_size
        self.hidden_dim = [128]
        self.num_patches_1d = img_size[0] // patch_size
        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_channels=in_chans, embed_dim=embed_dim[0])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, Mamba_depth[0])]
        inter_dpr = [0.0] + dpr
        self.MambaBlock = nn.ModuleList([
            VSSBlock(
                hidden_dim=self.embed_dim,
                drop_path=inter_dpr[i] if isinstance(inter_dpr, list) else drop_path_rate,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop_rate,
                d_state=d_state,
            )
            for i in range(Mamba_depth[0])])
        self.final_up = Final_PatchExpand2D(dim=self.embed_dim, dim_scale=self.patch_size, norm_layer=norm_layer)
        self.Mamba_final_conv = nn.Conv2d(self.embed_dim // self.patch_size, self.num_classes, 1)
        dpr2 = [x.item() for x in torch.linspace(0, drop_path_rate, Transformer_depth[0])]
        cur = 0
        self.Trans_Block = nn.ModuleList(
            [
                TransformerBlock(dim=embed_dim[0], num_heads=heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
                      qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr2[cur + i],
                      norm_layer=norm_layer, sr_ratio=sr_ratios[0])
                for i in range(Transformer_depth[0])
            ]
        )
        self.norm1 = norm_layer(embed_dim[0])
        self.dropout = nn.Dropout2d(drop_rate)
        self.Trans_proj = nn.Linear(embed_dim[0], self.hidden_dim[0])
        resnet = models.resnet18(pretrained=True)
        resnet.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
        )
        self.downsample = resnet.layer3
        self.EMA_uplevel = EMA(channels=128, factor=32)
        self.ASPP_uplevel = ASPP(in_channels=128, out_channels=128)
        self.EMA = EMA(channels=256, factor=32)
        self.ASPP = ASPP(in_channels=256, out_channels=128)
        self.upsample1 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.upsample2 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.weight_EMA_Transformer = nn.Parameter(torch.Tensor(1))
        self.MLP_Predict = nn.Conv2d(128, num_classes, kernel_size=1)
        self.weight_Mamba_other = nn.Parameter(torch.Tensor(1))
        # self.if_abs_pos_embed = True
        # self.resolution_indicator = resolution_indicator
        # self.Transformer_depth = depth
        # self.heads = heads
        # self.dim_head = dim_head
        # self.scale_dim = scale_dim
        # self.dropout = 0.
        # self.emb_dropout = 0.
        # self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, stride=stride, in_channels=in_chans,
        #                               embed_dim=self.embed_dim,
        #                               norm_layer=norm_layer if patch_norm else None)
        # self.num_patches = self.patch_embed.num_patches
        # if self.if_abs_pos_embed:
        #     self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))
        #     self.pos_drop = nn.Dropout(p=drop_rate)
        # self.ADPT_cls_token = nn.Parameter(resolution_indicator * torch.ones(1, num_classes, embed_dim))
        # self.transformer = Transformer(self.embed_dim, self.Transformer_depth, self.heads, self.dim_head,
        #                                      self.embed_dim * self.scale_dim, self.dropout)
        # self.dropout = nn.Dropout(self.emb_dropout)
        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(self.embed_dim),
        #     nn.Linear(self.embed_dim, self.num_classes * self.patch_size ** 2)
        # )








    def forward(self, x):
        B = x.shape[0]
        MambaBranch, H, W = self.patch_embed1(x)  # x (b, 3, 224, 224) -> (b, embed[0], 56, 56) -> (b, 3136, embed[0])
        TransformerBranch = MambaBranch #(b, H(56)* W(56), embed[0])
        MambaBranch = MambaBranch.view(B, H, W, self.embed_dim) #B,H,W,d
        for blk in self.MambaBlock:
            MambaBranch = blk(MambaBranch)
        MambaBranch = self.final_up(MambaBranch)
        MambaBranch = MambaBranch.permute(0, 3, 1, 2)
        MambaBranch = self.Mamba_final_conv(MambaBranch)  #B,Num_class, 224,224

        for i, blk in enumerate(self.Trans_Block):
            TransformerBranch = blk(TransformerBranch, H, W)
        TransformerBranch = self.norm1(TransformerBranch)
        TransformerBranch = TransformerBranch.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # 深拷贝
        TransformerBranch = TransformerBranch.flatten(2).transpose(1, 2)  # x (b, n, c)
        TransformerBranch = self.Trans_proj(TransformerBranch)  # x (b, n, hid)
        TransformerBranch = TransformerBranch.permute(0, 2, 1).reshape(B, -1, H, W)  # x (b, hid, 56(H/4), 56(H/4))
        TransformerBranch = self.dropout(TransformerBranch)

        EMABranch = x
        EMABranch = self.backbone(EMABranch) # B*128*28*28--(H/8)
        up_level_EMABranch = self.EMA_uplevel(EMABranch)  # B*128*28*28--(H/8)
        up_level_EMABranch = self.ASPP_uplevel(up_level_EMABranch)  # B*128*28*28--(H/8)
        low_level_Branch = self.downsample(EMABranch)  # B*256*14*14--(H/16)
        low_level_Branch = self.EMA(low_level_Branch)  # B*256*14*14--(H/16)
        low_level_Branch = self.ASPP(low_level_Branch)  # B*128*14*14--(H/16)
        EMABranch = self.upsample1(low_level_Branch)  # B*128*28*28--(H/8)
        EMABranch = EMABranch + up_level_EMABranch
        EMABranch = self.upsample2(EMABranch)  # B*128*56*56--(H/16)

        MergeFeature_EMA_Transformer = self.weight_EMA_Transformer * EMABranch + (1 - self.weight_EMA_Transformer) * TransformerBranch
        MergeFeature_EMA_Transformer = F.interpolate(MergeFeature_EMA_Transformer, size=(self.img_size[0], self.img_size[1]), mode='bilinear',
                                       align_corners=True)
        MergeFeature_EMA_Transformer = self.MLP_Predict(MergeFeature_EMA_Transformer)
        output = (1-self.weight_Mamba_other) * MergeFeature_EMA_Transformer + self.weight_Mamba_other * MambaBranch

        # x = self.patch_embed(x)
        # B, H, W, D = x.shape
        # #positional embedding
        # if self.if_abs_pos_embed:
        #     # x = x.view(B, D, H * W)
        #     x = x.view(B, H * W, D)
        #     x = x + self.pos_embed
        #     x = self.pos_drop(x)
        #     x = x.view(B, H, W, D)
        # # Transformer Branch
        # Branch_Trans = x.view(B, H*W, D)
        # Adpt_tokens = repeat(self.ADPT_cls_token, '() N d -> b N d', b=B)
        # Branch_Trans = torch.cat((Adpt_tokens, Branch_Trans), dim=1)
        # Branch_Trans = self.transformer(Branch_Trans)
        # weight_trans = Branch_Trans[:, :self.num_classes]  #B, # of class, d
        # Branch_Trans = Branch_Trans[:, self.num_classes:]  #B, # of H*W, d
        # Branch_Trans = self.mlp_head(Branch_Trans.reshape(-1, self.embed_dim))  #B, # of H*W, C*h*w
        # Branch_Trans = Branch_Trans.reshape(B, self.num_patches, self.patch_size ** 2, self.num_classes)
        # Branch_Trans = Branch_Trans.reshape(B, H * self.patch_size* W * self.patch_size, self.num_classes)
        # Branch_Trans = Branch_Trans.reshape(B, H * self.patch_size, W * self.patch_size, self.num_classes)
        # Branch_Trans = Branch_Trans.permute(0, 3, 1, 2)
        # out = Branch_Trans


        # VMamba Branch
        # Branch_Mamba = x
        # for blk in self.MambaBlock:
        #     Branch_Mamba = Branch_Mamba(x)
        # # x = self.MambaBlock(x)
        # x = self.final_up(x)
        # x = x.permute(0,3,1,2)
        # out = self.final_conv(x)

        # return output, self.weight_Mamba_other, self.weight_EMA_Transformer

        return output, self.weight_EMA_Transformer, self.weight_Mamba_other
