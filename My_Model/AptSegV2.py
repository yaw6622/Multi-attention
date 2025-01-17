# EMA版本

import torch

import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple
import torch.nn.functional as F
import torchvision.models as models
import math
from My_function.module import Attention, PreNorm, FeedForward
from einops import rearrange, repeat
# from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

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


class EfficientSelfAttention(nn.Module):  # 感觉基本借鉴了PVT的注意力实现
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


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU(), norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)  # (b, n, c) -> (b, n, c)

        self.attn = EfficientSelfAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                           attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)

        self.norm2 = norm_layer(dim)
        self.mlp = FFN(in_features=dim, hidden_feaures=(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        # DropPath和Dropout思想类似，将多分支结构的子路径"随机删除"
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, H, W):  # x (b, n, c)
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))  # x (b, n, c)
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x  # (b, n, c)

class LightweightConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LightweightConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


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

class AptSegV2(nn.Module):
    def __init__(self, img_size=[224,224], patch_size=16, stride=16, in_chans=3, depth=[4], num_classes=10, embed_dim=[96],
                 d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, mlp_ratios = [4], qkv_bias=False, qk_scale=None,
                 norm_layer=nn.LayerNorm, patch_norm=True, resolution_indicator = 0.5, heads = [2], dim_head = 16, scale_dim = 4,
                 use_checkpoint=False, sr_ratios=[8], **kwargs):
        super(AptSegV2, self).__init__()
        self.num_classes = num_classes
        self.depths = depth
        self.hidden_dim = [128]
        self.img_size = img_size
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_channels=in_chans, embed_dim=embed_dim[0])
        cur = 0
        self.block1 = nn.ModuleList(
            [
                Block(dim=embed_dim[0], num_heads=heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
                      qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                      norm_layer=norm_layer, sr_ratio=sr_ratios[0])
                for i in range(depth[0])
            ]
        )
        self.norm1 = norm_layer(embed_dim[0])
        self.proj = nn.Linear(embed_dim[0], self.hidden_dim[0])
        self.linear_pred = nn.Conv2d(self.hidden_dim[0], num_classes, kernel_size=1)  # 也是通过1x1卷积实现的mlp
        self.dropout = nn.Dropout2d(drop_rate)
        self.EMA = EMA(channels = 256, factor=32)
        self.ASPP = ASPP(in_channels=256, out_channels=128)
        self.EMA_uplevel = EMA(channels=128, factor=32)
        self.ASPP_uplevel = ASPP(in_channels=128, out_channels=128)
        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            # resnet.layer3,
            # resnet.layer4,
        )
        self.downsample =resnet.layer3
        self.upsample1 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.upsample2 =nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1)
        # self.reduce_conv = nn.Conv2d(2048, 256, kernel_size=1)#if resnet101,中间会是2048通道
        self.final_conv = nn.Conv2d(128, num_classes, kernel_size=1)
        self.MLP_Predict = nn.Conv2d(128, num_classes, kernel_size=1)
        # self.upconv =nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=4, padding=0)

        self.weight_EMA = nn.Parameter(torch.Tensor(1))






    def forward(self, x):
        B = x.shape[0]
        # EMA Branch   design 1 ResNet 18 backbone,EMA+ASPP
        RegularBranch = x
        RegularBranch = self.backbone(RegularBranch)
        # RegularBranch = self.reduce_conv(RegularBranch)
        up_level_RegularBranch = self.EMA_uplevel(RegularBranch) #B*128*28*28--(H/8)
        up_level_RegularBranch = self.ASPP_uplevel(up_level_RegularBranch) #B*128*28*28--(H/8)
        RegularBranch = self.downsample(RegularBranch) #B*256*14*14--(H/16)

        RegularBranch = self.EMA(RegularBranch) #B*256*14*14--(H/16)
        RegularBranch = self.ASPP(RegularBranch) #B*128*14*14--(H/16)

        # RegularBranch = self.final_conv(RegularBranch)
        # RegularBranch = F.interpolate(RegularBranch, size=(224, 224), mode='bilinear', align_corners=False)
        RegularBranch = self.upsample1(RegularBranch) #B*128*28*28--(H/8)
        RegularBranch = RegularBranch + up_level_RegularBranch
        RegularBranch = self.upsample2(RegularBranch) #B*128*56*56--(H/16)




        # out = []
        TransformerBranch, H, W = self.patch_embed1(x)
        # x, H, W = self.patch_embed1(x)  # x (b, 3, 224, 224) -> (b, embed[0], 56, 56) -> (b, 3136, embed[0])
        for i, blk in enumerate(self.block1):
            TransformerBranch = blk(TransformerBranch, H, W)
        TransformerBranch = self.norm1(TransformerBranch)
        # # x (b, 3136, embed[0]) -> (b, embed[0], 56, 56)
        TransformerBranch = TransformerBranch.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # 深拷贝
        # out.append(x)
        TransformerBranch = TransformerBranch.flatten(2).transpose(1, 2)  # x (b, n, c)
        TransformerBranch = self.proj(TransformerBranch)  # x (b, n, hid)
        TransformerBranch = TransformerBranch.permute(0,2,1).reshape(B, -1, H, W)  #存疑，原来是_c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])
        TransformerBranch = self.dropout(TransformerBranch)
        # TransformerBranch = self.linear_pred(TransformerBranch)
        Merge_features = self.weight_EMA * RegularBranch + (1-self.weight_EMA) * TransformerBranch
        # Merge_features = torch.cat([TransformerBranch, RegularBranch], dim=1)
        Merge_features = F.interpolate(Merge_features, size=(self.img_size[0], self.img_size[1]), mode='bilinear',
                               align_corners=True)
        output = self.MLP_Predict(Merge_features)

        # TransformerBranch = F.interpolate(TransformerBranch, size=(self.img_size[0], self.img_size[1]), mode='bilinear', align_corners=True)
        # Merge_features = torch.cat([TransformerBranch, RegularBranch], dim=1)



        # output= RegularBranch+TransformerBranch

        return output