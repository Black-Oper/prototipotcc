import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model

# ---------------------------------------------------------------------------
# Arquitetura: ShuffleVSR — Ultra-Lightweight Video Super-Resolution
#
# Filosofia de design:
#   Velocidade máxima com qualidade aceitável. Projetada para cenários
#   onde o orçamento computacional é extremamente restrito (mobile, edge).
#   Sacrifica capacidade de modelagem temporal em favor de throughput.
#
# Fundamentação:
#   - Shi et al., 2016 (ESPCN)
#       Sub-pixel shuffle no espaço LR como upsampling eficiente.
#
#   - Huang et al., 2025 (LightVSR)
#       Validação de que arquiteturas com ~3.5M params atingem ~28 FPS.
#       ShuffleVSR reduz ainda mais via convoluções separáveis.
#
#   - Xi et al., 2025 (Survey)
#       Redes leves (lightweight) são o futuro para equilíbrio
#       qualidade/velocidade. Estratégias de pós-amostragem essenciais.
#
#   - Wang et al., 2025 (HarmoQ)
#       Design quantization-friendly: sem operações que gerem outliers
#       extremos nas ativações, facilitando PTQ agressiva (INT4/INT8).
# ---------------------------------------------------------------------------


class DepthwiseSeparableConv(nn.Module):
    """Convolução separável em profundidade: reduz FLOPs em ~k² vezes
    comparado a convolução padrão, onde k é o tamanho do kernel."""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class ShuffleBlock(nn.Module):
    """Bloco residual com convolução separável e channel shuffle.

    Channel shuffle redistribui informação entre grupos de canais,
    compensando a falta de interação inter-grupo das convoluções
    depthwise sem custo computacional adicional.
    """

    def __init__(self, channels, groups=4):
        super().__init__()
        self.groups = groups
        self.body = nn.Sequential(
            DepthwiseSeparableConv(channels, channels),
            nn.PReLU(),
            DepthwiseSeparableConv(channels, channels),
        )

    def forward(self, x):
        out = x + self.body(x)
        # Channel shuffle
        B, C, H, W = out.shape
        out = out.view(B, self.groups, C // self.groups, H, W)
        out = out.transpose(1, 2).contiguous()
        out = out.view(B, C, H, W)
        return out


@register_model("ShuffleVSR", interface="recurrent")
class ShuffleVSR(nn.Module):
    """
    Ultra-Lightweight Video Super-Resolution via Depthwise Separable Convolutions.

    Pipeline (todo no espaço LR):
        Frame LR ──► Extração Separável ──► Concat Temporal ──►
        Fusão 1x1 ──► ShuffleBlocks ──► Sub-pixel Shuffle + Skip ──► Frame SR

    Prioriza throughput máximo (FPS) sobre qualidade de reconstrução.
    Ideal para deploy em dispositivos com recursos limitados ou como
    baseline de velocidade para comparação experimental.

    Args:
        scale_factor: Fator de upscaling (padrão: 2)
        channels: Canais da imagem (padrão: 3 para RGB)
        hidden_dim: Dimensão dos feature maps internos (padrão: 48)
        num_blocks: Quantidade de ShuffleBlocks (padrão: 4)
    """

    def __init__(self, scale_factor=2, channels=3, hidden_dim=48, num_blocks=4):
        super().__init__()
        self.scale_factor = scale_factor
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        # 1. Extração de Features — convoluções separáveis (espaço LR)
        self.feat_extract = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, 5, padding=2),  # primeira camada padrão
            nn.PReLU(),
            DepthwiseSeparableConv(hidden_dim, hidden_dim),
            nn.PReLU(),
        )

        # 2. Fusão Temporal — concat + redução 1x1 (mínimo overhead)
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 1),
            nn.PReLU(),
        )

        # 3. Refinamento — ShuffleBlocks com channel shuffle
        self.refine = nn.Sequential(
            *[ShuffleBlock(hidden_dim) for _ in range(num_blocks)]
        )

        # 4. Reconstrução Sub-pixel — Shi et al., 2016
        self.upsample = nn.Sequential(
            nn.Conv2d(hidden_dim, channels * (scale_factor ** 2), 3, padding=1),
            nn.PixelShuffle(scale_factor),
        )

        self._initialize_weights()

    def forward(self, x, prev_state=None):
        """
        Args:
            x: Frame LR atual (B, C, H, W)
            prev_state: Features do frame anterior (B, hidden_dim, H, W) ou None

        Returns:
            sr: Frame SR reconstruído (B, C, H*scale, W*scale)
            state: Features atuais para o próximo frame
        """
        # 1. Extração de features no espaço LR
        feat = self.feat_extract(x)

        # 2. Fusão temporal simples
        if prev_state is None:
            prev_state = torch.zeros_like(feat)

        fused = self.fusion(torch.cat([feat, prev_state], dim=1))

        # 3. Refinamento com skip connection global
        refined = self.refine(fused) + feat

        # 4. Upsampling sub-pixel + skip bicúbico
        residual = self.upsample(refined)
        base = F.interpolate(x, scale_factor=self.scale_factor,
                             mode='bicubic', align_corners=False)
        sr = residual + base

        return sr, feat

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
