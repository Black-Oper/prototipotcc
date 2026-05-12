import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model

try:
    from torchvision.ops import DeformConv2d
    HAS_DEFORM_CONV = True
except ImportError:
    HAS_DEFORM_CONV = False

# ---------------------------------------------------------------------------
# Arquitetura: PyramidVSR — Multi-Scale Coarse-to-Fine Video Super-Resolution
#
# Filosofia de design:
#   Qualidade máxima dentro do envelope de tempo real. Alinhamento
#   deformável em múltiplas escalas (coarse-to-fine) captura tanto
#   movimentos amplos quanto deslocamentos sub-pixel finos.
#
# Fundamentação:
#   - Fuoli et al., 2023 (DAP — Deformable Attention Pyramid)
#       Pirâmide de atenção deformável em múltiplas escalas para
#       alinhamento temporal eficiente. DAP-128 atinge >26 FPS.
#
#   - Miyazaki et al., 2024 (RSRDCN)
#       Alinhamento piramidal com convolução deformável em escala
#       multi-nível. Extração de features em múltiplas escalas via
#       encoder piramidal. 30x mais rápido que patch-matching.
#
#   - Jo et al., 2018 (DUF)
#       Filtros dinâmicos + residual para detalhes de alta frequência.
#       Qualidade SOTA mas lento. PyramidVSR busca qualidade similar
#       com arquitetura mais eficiente.
#
#   - Caballero et al., 2017
#       Compensação de movimento multi-escala integrada end-to-end.
#
#   - Shi et al., 2016 (ESPCN)
#       Sub-pixel shuffle no espaço LR.
#
#   - Huang et al., 2025 (LightVSR)
#       Agregação de features multi-escala para preservar detalhes finos.



class SEBlock(nn.Module):
    """Squeeze-and-Excitation para atenção por canal."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.PReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.excitation(self.squeeze(x))


class ResidualBlock(nn.Module):
    """Bloco residual com SE-attention."""

    def __init__(self, channels, use_attention=True):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.attention = SEBlock(channels) if use_attention else nn.Identity()

    def forward(self, x):
        return x + self.attention(self.body(x))


class _PyramidDeformAligner(nn.Module):
    """Alinhamento deformável piramidal coarse-to-fine (Fuoli 2023, Miyazaki 2024).

    Opera em 3 escalas: 1/4, 1/2, 1x. Offsets estimados na escala mais
    grosseira são propagados e refinados nas escalas mais finas, permitindo
    capturar movimentos amplos (escala coarse) e ajustes sub-pixel (fine).
    """

    def __init__(self, hidden_dim):
        super().__init__()
        # Downsampling para criar pirâmide
        self.down_2x = nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1)
        self.down_4x = nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1)

        # Predição de offsets em cada escala
        self.offset_coarse = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(hidden_dim, 2 * 3 * 3, 3, padding=1),
        )
        self.offset_mid = nn.Sequential(
            nn.Conv2d(hidden_dim * 2 + 2 * 3 * 3, hidden_dim, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(hidden_dim, 2 * 3 * 3, 3, padding=1),
        )
        self.offset_fine = nn.Sequential(
            nn.Conv2d(hidden_dim * 2 + 2 * 3 * 3, hidden_dim, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(hidden_dim, 2 * 3 * 3, 3, padding=1),
        )

        # Convolução deformável final na escala original
        self.deform_conv = DeformConv2d(hidden_dim, hidden_dim, 3, padding=1)

    def forward(self, feat, prev_state):
        """Alinhamento coarse-to-fine em 3 escalas."""
        # Pirâmide do frame atual
        feat_2x = self.down_2x(feat)
        feat_4x = self.down_4x(feat_2x)

        # Pirâmide do estado anterior
        prev_2x = self.down_2x(prev_state)
        prev_4x = self.down_4x(prev_2x)

        # Escala coarse (1/4): captura movimentos amplos
        offset_c = self.offset_coarse(torch.cat([feat_4x, prev_4x], dim=1))

        # Escala mid (1/2): refina com offsets propagados
        offset_c_up = F.interpolate(offset_c, size=feat_2x.shape[2:],
                                    mode='bilinear', align_corners=False) * 2
        offset_m = self.offset_mid(
            torch.cat([feat_2x, prev_2x, offset_c_up], dim=1)
        )

        # Escala fine (1x): ajuste sub-pixel final
        offset_m_up = F.interpolate(offset_m, size=feat.shape[2:],
                                    mode='bilinear', align_corners=False) * 2
        offset_f = self.offset_fine(
            torch.cat([feat, prev_state, offset_m_up], dim=1)
        )

        # Convolução deformável com offsets refinados
        return self.deform_conv(prev_state, offset_f)


class _PyramidConvAligner(nn.Module):
    """Fallback: alinhamento piramidal sem convolução deformável.
    Usa convoluções padrão em múltiplas escalas."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.down_2x = nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1)

        self.align_coarse = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.PReLU(),
        )
        self.align_fine = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
        )

    def forward(self, feat, prev_state):
        feat_2x = self.down_2x(feat)
        prev_2x = self.down_2x(prev_state)
        coarse = self.align_coarse(torch.cat([feat_2x, prev_2x], dim=1))
        coarse_up = F.interpolate(coarse, size=feat.shape[2:],
                                  mode='bilinear', align_corners=False)
        return self.align_fine(torch.cat([feat + coarse_up, prev_state], dim=1))


class ConvGRU(nn.Module):
    """Unidade recorrente convolucional para fusão temporal."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.conv_reset = nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1)
        self.conv_update = nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1)
        self.conv_candidate = nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1)

    def forward(self, feat, prev_state):
        combined = torch.cat([feat, prev_state], dim=1)
        reset = torch.sigmoid(self.conv_reset(combined))
        update = torch.sigmoid(self.conv_update(combined))
        candidate = torch.tanh(
            self.conv_candidate(torch.cat([feat, reset * prev_state], dim=1))
        )
        return (1 - update) * prev_state + update * candidate


@register_model("PyramidVSR", interface="recurrent")
class PyramidVSR(nn.Module):
    """
    Multi-Scale Coarse-to-Fine Video Super-Resolution.

    Pipeline (todo no espaço LR):
        Frame LR ──► Extração de Features ──► Alinhamento Piramidal
        (3 escalas: 1/4 → 1/2 → 1x) ──► ConvGRU (fusão temporal) ──►
        Refinamento Residual+SE ──► Sub-pixel Shuffle + Skip ──► Frame SR

    Alinhamento em múltiplas escalas captura movimentos amplos (escala
    coarse) e deslocamentos sub-pixel (escala fine), resultando na
    maior qualidade de reconstrução entre as arquiteturas propostas.

    Args:
        scale_factor: Fator de upscaling (padrão: 2)
        channels: Canais da imagem (padrão: 3 para RGB)
        hidden_dim: Dimensão dos feature maps internos (padrão: 64)
        num_res_blocks: Quantidade de blocos residuais (padrão: 5)
    """

    def __init__(self, scale_factor=2, channels=3, hidden_dim=64, num_res_blocks=5):
        super().__init__()
        self.scale_factor = scale_factor
        self.hidden_dim = hidden_dim
        self.num_res_blocks = num_res_blocks

        # 1. Extração de Features (espaço LR)
        self.feat_extract = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, 5, padding=2),
            nn.PReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.PReLU(),
        )

        # 2. Alinhamento Piramidal — Fuoli 2023 / Miyazaki 2024
        if HAS_DEFORM_CONV:
            self.aligner = _PyramidDeformAligner(hidden_dim)
        else:
            print("[AVISO] torchvision.ops.DeformConv2d não disponível. "
                  "Usando alinhamento piramidal por convolução padrão.")
            self.aligner = _PyramidConvAligner(hidden_dim)

        # 3. Fusão Temporal via ConvGRU
        self.fusion = ConvGRU(hidden_dim)

        # 4. Refinamento com SE-attention
        self.refine = nn.Sequential(
            *[ResidualBlock(hidden_dim, use_attention=(i % 2 == 1))
              for i in range(num_res_blocks)]
        )

        # 5. Reconstrução Sub-pixel — Shi et al., 2016
        self.upsample = nn.Sequential(
            nn.Conv2d(hidden_dim, channels * (scale_factor ** 2), 3, padding=1),
            nn.PixelShuffle(scale_factor),
        )

        self._initialize_weights()

    def forward(self, x, prev_state=None):
        """
        Args:
            x: Frame LR atual (B, C, H, W)
            prev_state: Estado oculto do frame anterior (B, hidden_dim, H, W) ou None

        Returns:
            sr: Frame SR reconstruído (B, C, H*scale, W*scale)
            state: Estado oculto atualizado para o próximo frame
        """
        # 1. Extração de features no espaço LR
        feat = self.feat_extract(x)

        # 2. Alinhamento piramidal coarse-to-fine
        if prev_state is None:
            prev_state = torch.zeros_like(feat)

        aligned = self.aligner(feat, prev_state)

        # 3. Fusão temporal recorrente via ConvGRU
        state = self.fusion(feat, aligned)

        # 4. Refinamento com skip connection global
        refined = self.refine(state) + feat

        # 5. Upsampling sub-pixel + skip bicúbico
        residual = self.upsample(refined)
        base = F.interpolate(x, scale_factor=self.scale_factor,
                             mode='bicubic', align_corners=False)
        sr = residual + base

        return sr, state

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
