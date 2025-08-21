import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- 基础模块 ----
class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None: p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()
    def forward(self, x): return self.act(self.bn(self.conv(x)))

# ---- 尺度与通道对齐 ----
class ScaleAlign(nn.Module):
    def __init__(self, in_ch, out_ch, target_stride=4, in_stride=4):
        super().__init__()
        self.proj = ConvBNAct(in_ch, out_ch, k=1, act=True)
        self.scale = in_stride // target_stride  # >1 means down, <1 means up
        # 我们默认把所有特征对齐到 1/4 尺度（target_stride=4）
    def forward(self, x, out_hw):
        x = self.proj(x)
        x = F.interpolate(x, size=out_hw, mode='bilinear', align_corners=False)
        return x

# ---- 跨尺度门控交互（轻量）----
class CrossScaleGatedInteraction(nn.Module):
    """
    输入 list of tensors [x1, x2, x3, x4] (同尺度同通道)，输出同维度的交互增强后的列表。
    用 GAP 得到每路摘要，经 1x1 映射成门值，彼此相乘实现“看别人再调自己”。
    """
    def __init__(self, ch, hidden=64):
        super().__init__()
        self.to_gate = nn.Sequential(
            nn.Conv2d(ch, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.GroupNorm(hidden),
            nn.Sigmoid()
        )
    def forward(self, feats):
        # feats: list of [B,C,H,W]
        pooled = [F.adaptive_avg_pool2d(f, 1) for f in feats]    # 每路摘要
        gates  = [self.to_gate(p) for p in pooled]               # 每路门
        out = []
        for i, fi in enumerate(feats):
            # 用其他路的门的均值来调制当前路（排除自己）
            others = [g for j, g in enumerate(gates) if j != i]
            if len(others) > 0:
                mod = torch.stack(others, dim=0).mean(0)
                out.append(fi * (1.0 + mod))   # 残差式调制，稳定训练
            else:
                out.append(fi)
        return out

# ---- 动态尺度路由（权重自适应）----
class DynamicScaleRouter(nn.Module):
    def __init__(self, ch, hidden=64, num_paths=4):
        super().__init__()
        self.scorers = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(ch, hidden, 1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, 1, 1, bias=True)
            ) for _ in range(num_paths)
        ])
        self.softmax = nn.Softmax(dim=1)
    def forward(self, feats):
        # feats: list of [B,C,H,W]
        scores = []
        for i, f in enumerate(feats):
            s = self.scorers[i](f)   # [B,1,1,1]
            scores.append(s)
        S = torch.cat(scores, dim=1)         # [B,4,1,1]
        W = self.softmax(S)                  # soft routing weights
        fused = 0
        for i, f in enumerate(feats):
            w = W[:, i:i+1]                  # [B,1,1,1]
            fused = fused + f * w
        return fused, W.squeeze(-1).squeeze(-1)  # 返回权重以便可视化



# ---- 总体解码器 ----
class AFFDecoder(nn.Module):
    """
    输入：
      x4_t:  [B, Ct,  H/4,  W/4]  (Transformer分支)
      x4_e:  [B, Ce,  H/4,  W/4]  (EMA分支)
      x8_m:  [B, C8,  H/8,  W/8]  (Mamba分支)
      x16_m: [B, C16, H/16, W/16] (Mamba分支)
    输出：
      logits: [B, num_classes, H, W]
    """
    def __init__(self, Ct, Ce, C8, C16, num_classes, C=256, aspp_out=256):
        super().__init__()
        # 对齐到 1/4 尺度 + 通道对齐
        self.align_t  = ScaleAlign(Ct,  C, target_stride=4, in_stride=4)
        self.align_e  = ScaleAlign(Ce,  C, target_stride=4, in_stride=4)
        self.align_8  = ScaleAlign(C8,  C, target_stride=4, in_stride=8)
        self.align_16 = ScaleAlign(C16, C, target_stride=4, in_stride=16)

        # 跨尺度门控交互
        self.csgi = CrossScaleGatedInteraction(C, hidden=C//4)

        # 路由融合
        self.router = DynamicScaleRouter(C, hidden=C//4, num_paths=4)

        # 上下文与细化
        self.aspp   = ASPP(C, aspp_out, atrous_rates=(1,3,6,9))
        self.cbam   = CBAM(aspp_out, r=16)

        # 预测头（1/4 尺度）
        self.head_1_4 = nn.Sequential(
            ConvBNAct(aspp_out, aspp_out, k=3),
            nn.Conv2d(aspp_out, num_classes, 1, bias=True)
        )

        # 可选：边界分支（辅助监督）
        self.boundary = nn.Sequential(
            ConvBNAct(aspp_out, aspp_out//2, k=3),
            nn.Conv2d(aspp_out//2, 1, 1, bias=True)
        )

    def forward(self, x4_t, x4_e, x8_m, x16_m, out_hw):
        """
        out_hw: 原图 (H, W)，用于最终上采样
        """
        h4, w4 = out_hw[0]//4, out_hw[1]//4

        # 1) 对齐到 1/4 & 同通道
        t = self.align_t(x4_t, (h4, w4))
        e = self.align_e(x4_e, (h4, w4))
        m8 = self.align_8(x8_m, (h4, w4))
        m16 = self.align_16(x16_m, (h4, w4))

        feats = [t, e, m8, m16]

        # 2) 跨尺度门控交互
        feats = self.csgi(feats)

        # 3) 动态尺度路由融合
        fused, weights = self.router(feats)  # weights: [B,4]

        # 4) 上下文聚合 + 细化
        ctx = self.aspp(fused)
        ctx = self.cbam(ctx)

        # 5) 预测与上采样
        logits_1_4 = self.head_1_4(ctx)
        logits = F.interpolate(logits_1_4, size=out_hw, mode='bilinear', align_corners=False)

        # 可返回边界图用于辅助损失（可选）
        boundary_1_4 = self.boundary(ctx)
        boundary = F.interpolate(boundary_1_4, size=out_hw, mode='bilinear', align_corners=False)

        return logits, logits_1_4, boundary, weights
