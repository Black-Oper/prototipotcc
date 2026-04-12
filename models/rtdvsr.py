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
# Arquitetura: RTDVSR — Real-Time Deformable Video Super-Resolution
#
# Fundamentação:
#   - Shi et al., 2016 (ESPCN)
#       Extração de features inteiramente no espaço LR + sub-pixel shuffle.
#       Reduz complexidade computacional e de memória em ordens de magnitude.
#
#   - Fuoli et al., 2023 (DAP)
#       Arquitetura recorrente estritamente causal com alinhamento deformável.
#       Permite inferência online (streaming) sem depender de frames futuros.
#       DAP-128 atinge >26 FPS com qualidade competitiva.
#
#   - Miyazaki et al., 2024 (RSRDCN)
#       Convolução deformável para alinhamento de features, substituindo
#       optical flow e patch-matching. 30x mais rápido que SRNTT.
#
#   - Caballero et al., 2017
#       Fusão espaço-temporal no espaço LR. 3-5 frames é o ponto ótimo;
#       acima disso, ruído temporal degrada o desempenho.
#
#   - Huang et al., 2025 (LightVSR)
#       Design leve com ~3.5M params atingindo ~28 FPS.
#       Blocos residuais e atenção simplificada (SE-attention).
#
#   - Wang et al., 2019 / Xi et al., 2025 (Surveys)
#       Pós-amostragem (post-upsampling) é a estratégia mais eficiente.
#       Redes leves são o futuro para equilíbrio qualidade/velocidade.
#
#   - Wang et al., 2025 (HarmoQ / Outlier-Aware PTQ)
#       Quantização pós-treinamento pode acelerar até 75x. Arquitetura
#       deve ser "quantization-friendly" (evitar outliers extremos).
# ---------------------------------------------------------------------------


class SEBlock(nn.Module):
    """Squeeze-and-Excitation para atenção por canal (Huang et al., 2025).

    Recalibra os feature maps por canal, priorizando canais com
    informação de textura e movimento mais relevantes.
    """

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
        scale = self.excitation(self.squeeze(x))
        return x * scale


class ResidualBlock(nn.Module):
    """Bloco residual leve com SE-attention opcional para refinamento no espaço LR."""

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


class ConvGRU(nn.Module):
    """Unidade recorrente convolucional para fusão temporal (Fuoli et al., 2023).

    Mantém um estado oculto que acumula informação temporal de forma
    controlada via gates de reset e update, garantindo coerência
    temporal em sequências longas sem degradação de qualidade.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.conv_reset = nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1)
        self.conv_update = nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1)
        self.conv_candidate = nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1)

    def forward(self, feat, prev_state):
        """
        Args:
            feat: Features do frame atual (B, hidden_dim, H, W)
            prev_state: Estado oculto alinhado do frame anterior

        Returns:
            new_state: Estado oculto atualizado
        """
        combined = torch.cat([feat, prev_state], dim=1)

        reset = torch.sigmoid(self.conv_reset(combined))
        update = torch.sigmoid(self.conv_update(combined))

        candidate = torch.tanh(
            self.conv_candidate(torch.cat([feat, reset * prev_state], dim=1))
        )

        new_state = (1 - update) * prev_state + update * candidate
        return new_state


class _DeformableAligner(nn.Module):
    """Alinhamento temporal via convolução deformável (Fuoli 2023, Miyazaki 2024).

    Prediz offsets a partir da concatenação [features_atuais, estado_anterior]
    e aplica convolução deformável no estado anterior para alinhá-lo ao frame
    corrente. Substitui optical flow com custo muito menor.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.offset_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(hidden_dim, 2 * 3 * 3, 3, padding=1),  # 18 offsets para kernel 3x3
        )
        self.deform_conv = DeformConv2d(hidden_dim, hidden_dim, 3, padding=1)

    def forward(self, feat, prev_state):
        offsets = self.offset_conv(torch.cat([feat, prev_state], dim=1))
        return self.deform_conv(prev_state, offsets)


class _ConvAligner(nn.Module):
    """Fallback: alinhamento via convolução padrão quando DeformConv2d não
    está disponível. Menos eficaz, mas funcional."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.align = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
        )

    def forward(self, feat, prev_state):
        return self.align(torch.cat([feat, prev_state], dim=1))


@register_model("RTDVSR", interface="recurrent")
class RTDVSR(nn.Module):
    """
    Real-Time Deformable Video Super-Resolution.

    Arquitetura recorrente causal com alinhamento implícito via convolução
    deformável e fusão temporal via ConvGRU. Todo o processamento ocorre
    no espaço de baixa resolução até o estágio final de sub-pixel shuffle.

    Pipeline (todo no espaço LR):
        Frame LR ──► Extração de Features ──► Alinhamento Deformável ──►
        ConvGRU (fusão temporal) ──► Refinamento Residual+SE ──►
        Sub-pixel Shuffle + Skip Bicúbico ──► Frame SR

    O estado oculto (hidden state) carrega informação temporal dos frames
    anteriores, permitindo inferência causal (online) sem buffering.

    Args:
        scale_factor: Fator de upscaling (padrão: 2)
        channels: Canais da imagem (padrão: 3 para RGB)
        hidden_dim: Dimensão dos feature maps internos (padrão: 64)
        num_res_blocks: Quantidade de blocos residuais de refinamento (padrão: 4)
    """

    def __init__(self, scale_factor=2, channels=3, hidden_dim=64, num_res_blocks=4):
        super().__init__()
        self.scale_factor = scale_factor
        self.hidden_dim = hidden_dim
        self.num_res_blocks = num_res_blocks

        # 1. Extração de Features (espaço LR) — Shi et al., 2016
        self.feat_extract = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, 5, padding=2),
            nn.PReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.PReLU(),
        )

        # 2. Alinhamento Temporal — Fuoli 2023 / Miyazaki 2024
        if HAS_DEFORM_CONV:
            self.aligner = _DeformableAligner(hidden_dim)
        else:
            print("[AVISO] torchvision.ops.DeformConv2d não disponível. "
                  "Usando alinhamento por convolução padrão (qualidade reduzida).")
            self.aligner = _ConvAligner(hidden_dim)

        # 3. Fusão Temporal via ConvGRU — Fuoli et al., 2023
        self.fusion = ConvGRU(hidden_dim)

        # 4. Refinamento com SE-attention — Huang et al., 2025 (LightVSR)
        self.refine = nn.Sequential(
            *[ResidualBlock(hidden_dim, use_attention=(i % 2 == 1))
              for i in range(num_res_blocks)]
        )

        # 5. Reconstrução Sub-pixel — Shi et al., 2016 (ESPCN)
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

        # 2. Alinhamento temporal via convolução deformável
        if prev_state is None:
            prev_state = torch.zeros_like(feat)

        aligned = self.aligner(feat, prev_state)

        # 3. Fusão temporal recorrente via ConvGRU
        state = self.fusion(feat, aligned)

        # 4. Refinamento com skip connection global
        refined = self.refine(state) + feat

        # 5. Upsampling sub-pixel + skip bicúbico (Wang et al., 2019)
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
