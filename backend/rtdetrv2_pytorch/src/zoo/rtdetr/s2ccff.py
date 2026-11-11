"""S2-CCFF (Cross-Scale Feature Fusion with S2) for RT-DETR / RT-DETRv2.

Re-implementation of the S2-CCFF module from PF-DETR
(Liu et al., 2025, doi:10.20944/preprints202410.1980.v2), built from Fig. 3-6 and Eq. 1-4.
The paper has no public code. PPA and NWD are NOT implemented here.

Global branch (DCAM / FSAM) follows OKNet (Cui et al., AAAI 2024, "Omni-Kernel Network
for Image Restoration"): DCAM = FCA + SCA, FSAM = FGM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import register
from .hybrid_encoder import HybridEncoder, ConvNormLayer, CSPRepLayer


__all__ = ['SPDConv', 'DCAM', 'FSAM', 'OmniKernel', 'CSPOK', 'CSPOKFusion', 'S2CCFFHybridEncoder']


class SPDConv(nn.Module):
    """Space-to-depth (scale 2) followed by a non-strided Conv + BN + act (SPD-Conv, Sunkara & Luo 2022).
    (B, C, H, W) -> (B, out_channels, H/2, W/2)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, scale=2, act='silu'):
        super().__init__()
        self.scale = scale
        self.conv = ConvNormLayer(in_channels * scale * scale, out_channels, kernel_size, 1, act=act)

    def space_to_depth(self, x):
        s = self.scale
        h, w = x.shape[-2:]
        if h % s or w % s:  # odd size: pad bottom/right so no pixel is dropped
            x = F.pad(x, (0, (s - w % s) % s, 0, (s - h % s) % s))
        # same ordering as the SPD-Conv paper: f(0,0), f(1,0), f(0,1), f(1,1)
        return torch.cat([x[..., i::s, j::s] for j in range(s) for i in range(s)], dim=1)

    def forward(self, x):
        return self.conv(self.space_to_depth(x))


def _fp32_fft_region(fn):
    """cuFFT does not support fp16 for non power-of-two sizes -> always run the FFT branch in fp32."""
    def wrapper(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            return fn(self, x.float())
    return wrapper


class DCAM(nn.Module):
    """Dual-domain Channel Attention Module (OKNet: FCA followed by SCA).
    FCA: channel weights from GAP -> 1x1 conv, applied in the frequency domain (FFT -> x -> IFFT -> |.|)
    SCA: channel weights from GAP -> 1x1 conv, applied in the spatial domain.
    """
    def __init__(self, dim):
        super().__init__()
        self.fca_pool = nn.AdaptiveAvgPool2d(1)
        self.fca_conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sca_pool = nn.AdaptiveAvgPool2d(1)
        self.sca_conv = nn.Conv2d(dim, dim, 1, bias=True)

    @_fp32_fft_region
    def forward(self, x):
        att = self.fca_conv(self.fca_pool(x))
        x_fft = torch.fft.fft2(x, norm='backward')
        x_fca = torch.abs(torch.fft.ifft2(att * x_fft, dim=(-2, -1), norm='backward'))
        att = self.sca_conv(self.sca_pool(x_fca))
        return att * x_fca


class FSAM(nn.Module):
    """Frequency-based Spatial Attention Module (OKNet: FGM).
    Two 1x1 convs; one path goes to the frequency domain and is modulated by the other,
    IFFT -> |.|, then fused with the input via learnable alpha / beta.
    """
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    @_fp32_fft_region
    def forward(self, x):
        x1 = self.conv1(x)
        x2_fft = torch.fft.fft2(self.conv2(x), norm='backward')
        out = torch.abs(torch.fft.ifft2(x1 * x2_fft, dim=(-2, -1), norm='backward'))
        return out * self.alpha + x * self.beta


class OmniKernel(nn.Module):
    """Omni-Kernel Module (OKM) as in Fig. 6a / Eq. 1-3 of PF-DETR.
    y = C1x1(X1)
    O_local  = DW1x1(y)                                   (Eq. 1)
    O_large  = DW1x61(y) + DW61x61(y) + DW61x1(y)         (Eq. 2, kernel 63 -> 61 per paper)
    O_global = FSAM(DCAM(y))                              (Eq. 3)
    out      = C1x1(O_local + O_large + O_global)         (inner term of Eq. 4)
    """
    def __init__(self, dim, large_kernel=61):
        super().__init__()
        k, p = large_kernel, large_kernel // 2
        self.in_conv = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.GELU())
        self.dw_11 = nn.Conv2d(dim, dim, 1, groups=dim)
        self.dw_1k = nn.Conv2d(dim, dim, (1, k), padding=(0, p), groups=dim)
        self.dw_kk = nn.Conv2d(dim, dim, k, padding=p, groups=dim)
        self.dw_k1 = nn.Conv2d(dim, dim, (k, 1), padding=(p, 0), groups=dim)
        self.dcam = DCAM(dim)
        self.fsam = FSAM(dim)
        self.act = nn.ReLU()
        self.out_conv = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        y = self.in_conv(x)
        o_local = self.dw_11(y)
        o_large = self.dw_1k(y) + self.dw_kk(y) + self.dw_k1(y)
        o_global = self.fsam(self.dcam(y))
        out = self.act(o_local + o_large + o_global.to(o_local.dtype))
        return self.out_conv(out)


class CSPOK(nn.Module):
    """CSP + Omni-Kernel (Fig. 6a, Eq. 4).
    Conv1x1 -> split channels 1/4 (X1) | 3/4 (X2) -> OKM(X1) -> concat with X2 -> Conv1x1.
    """
    def __init__(self, in_channels, out_channels, ok_ratio=0.25, large_kernel=61, act='silu'):
        super().__init__()
        self.c1 = max(1, int(round(out_channels * ok_ratio)))
        self.c2 = out_channels - self.c1
        self.cv1 = ConvNormLayer(in_channels, out_channels, 1, 1, act=act)
        self.okm = OmniKernel(self.c1, large_kernel=large_kernel)
        self.cv2 = ConvNormLayer(out_channels, out_channels, 1, 1, act=act)

    def forward(self, x):
        x1, x2 = self.cv1(x).split([self.c1, self.c2], dim=1)
        return self.cv2(torch.cat([self.okm(x1), x2], dim=1))


class CSPOKFusion(nn.Module):
    """CSPOK-Fusion (Fig. 5): concat (done by caller) -> CSPOK -> RT-DETR Fusion block.
    The Fusion block is exactly RT-DETR's CSPRepLayer: two 1x1 branches, one through N x RepVggBlock, added.
    """
    def __init__(self, in_channels, out_channels, num_blocks=3, expansion=1.0, act='silu',
                 ok_ratio=0.25, large_kernel=61):
        super().__init__()
        self.cspok = CSPOK(in_channels, out_channels, ok_ratio=ok_ratio, large_kernel=large_kernel, act=act)
        self.fusion = CSPRepLayer(out_channels, out_channels, num_blocks, expansion=expansion, act=act)

    def forward(self, x):
        return self.fusion(self.cspok(x))


@register()
class S2CCFFHybridEncoder(HybridEncoder):
    """HybridEncoder (AIFI + CCFF) with the S3 top-down fusion replaced by CSPOK-Fusion that also
    receives SPDConv(S2). Input: 4 backbone levels [S2, S3, S4, S5] (strides 4/8/16/32).
    Output: 3 levels (strides 8/16/32), identical interface to HybridEncoder -> the decoder is unchanged.

    Parameter names of everything except `spd` and `fpn_blocks.1` are identical to HybridEncoder,
    so a COCO-pretrained RT-DETR checkpoint can still be loaded for all shared parts.
    `use_encoder_idx` indexes the 3 levels [S3, S4, S5] (2 = S5, as in RT-DETR).
    """
    def __init__(self,
                 in_channels=[64, 128, 256, 512],
                 feat_strides=[4, 8, 16, 32],
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
                 version='v2',
                 spd_kernel_size=3,
                 ok_ratio=0.25,
                 large_kernel=61):
        assert len(in_channels) == 4 and len(feat_strides) == 4, 'expects [S2, S3, S4, S5]'
        super().__init__(in_channels=in_channels[1:], feat_strides=feat_strides[1:], hidden_dim=hidden_dim,
                         nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, enc_act=enc_act,
                         use_encoder_idx=use_encoder_idx, num_encoder_layers=num_encoder_layers,
                         pe_temperature=pe_temperature, expansion=expansion, depth_mult=depth_mult, act=act,
                         eval_spatial_size=eval_spatial_size, version=version)
        self.s2_channels = in_channels[0]
        self.spd = SPDConv(in_channels[0], hidden_dim, kernel_size=spd_kernel_size, act=act)
        # fpn_blocks[0] fuses at S4, fpn_blocks[-1] fuses at S3 -> replace the S3 one
        self.fpn_blocks[-1] = CSPOKFusion(hidden_dim * 3, hidden_dim, round(3 * depth_mult),
                                          expansion=expansion, act=act, ok_ratio=ok_ratio,
                                          large_kernel=large_kernel)

    def forward(self, feats):
        assert len(feats) == len(self.in_channels) + 1, f'expected {len(self.in_channels) + 1} levels'
        s2, feats = feats[0], feats[1:]
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # AIFI (unchanged)
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None)
                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        # top-down (S5 -> S4 unchanged, S4 -> S3 uses CSPOK-Fusion with SPDConv(S2))
        n = len(self.in_channels)
        inner_outs = [proj_feats[-1]]
        for idx in range(n - 1, 0, -1):
            feat_heigh = self.lateral_convs[n - 1 - idx](inner_outs[0])
            inner_outs[0] = feat_heigh
            feat_low = proj_feats[idx - 1]
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest')
            if idx == 1:
                s2_feat = self.spd(s2)
                if s2_feat.shape[-2:] != feat_low.shape[-2:]:
                    s2_feat = F.interpolate(s2_feat, size=feat_low.shape[-2:], mode='nearest')
                fused_in = torch.concat([upsample_feat, feat_low, s2_feat], dim=1)
            else:
                fused_in = torch.concat([upsample_feat, feat_low], dim=1)
            inner_outs.insert(0, self.fpn_blocks[n - 1 - idx](fused_in))

        # bottom-up (unchanged)
        outs = [inner_outs[0]]
        for idx in range(n - 1):
            downsample_feat = self.downsample_convs[idx](outs[-1])
            outs.append(self.pan_blocks[idx](torch.concat([downsample_feat, inner_outs[idx + 1]], dim=1)))
        return outs
