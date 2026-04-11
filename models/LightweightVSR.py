import torch
import torch.nn as nn

from .registry import register_model

try:
    from torchvision.ops import DeformConv2d
    HAS_DEFORM_CONV = True
except ImportError:
    HAS_DEFORM_CONV = False

# ---------------------------------------------------------------------------
# Arquitetura: Lightweight Recurrent VSR
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
#       Blocos residuais e atenção simplificada.
#
#   - Wang et al., 2019 / Xi et al., 2025 (Surveys)
#       Pós-amostragem (post-upsampling) é a estratégia mais eficiente.
#       Redes leves são o futuro para equilíbrio qualidade/velocidade.
#
#   - Wang et al., 2025 (HarmoQ / Outlier-Aware PTQ)
#       Quantização pós-treinamento pode acelerar até 75x. Arquitetura
#       deve ser "quantization-friendly" (evitar outliers extremos).
# ---------------------------------------------------------------------------


class ResidualBlock(nn.Module):
    """Bloco residual leve para refinamento de features no espaço LR."""

    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.body(x)


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


@register_model("LightweightVSR", interface="recurrent")
class LightweightVSR(nn.Module):
    """
    Super-Resolução de Vídeo Leve com Alinhamento Temporal Recorrente.

    Pipeline (todo no espaço LR):
        Frame LR ──► Extração de Features ──► Alinhamento Temporal ──►
        Fusão ──► Refinamento Residual ──► Sub-pixel Shuffle ──► Frame SR

    O estado oculto (hidden state) carrega informação temporal dos frames
    anteriores, permitindo inferência causal (online) sem buffering.

    Args:
        scale_factor: Fator de upscaling (padrão: 2)
        channels: Canais da imagem (padrão: 3 para RGB)
        hidden_dim: Dimensão dos feature maps internos (padrão: 64)
        num_res_blocks: Quantidade de blocos residuais de refinamento (padrão: 6)
    """

    def __init__(self, scale_factor=2, channels=3, hidden_dim=64, num_res_blocks=6):
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
        )

        # 2. Alinhamento Temporal — Fuoli 2023 / Miyazaki 2024
        if HAS_DEFORM_CONV:
            self.aligner = _DeformableAligner(hidden_dim)
        else:
            print("[AVISO] torchvision.ops.DeformConv2d não disponível. "
                  "Usando alinhamento por convolução padrão (qualidade reduzida).")
            self.aligner = _ConvAligner(hidden_dim)

        # 3. Fusão Temporal — Caballero et al., 2017
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 1),  # 1x1 para redução de canais
            nn.PReLU(),
        )

        # 4. Refinamento — Huang et al., 2025 (LightVSR)
        self.refine = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(num_res_blocks)]
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

        # 3. Fusão: features atuais + features temporais alinhadas
        fused = self.fusion(torch.cat([feat, aligned], dim=1))

        # 4. Refinamento com skip connection global
        refined = self.refine(fused) + feat

        # 5. Upsampling sub-pixel
        sr = self.upsample(refined)

        return sr, refined  # refined vira prev_state do próximo frame

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
